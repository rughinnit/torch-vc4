"""aten implementations for the vc4 subset backend.

Every entry point here is registered at ``PrivateUse1`` and is reached only
when torch has decided the op runs on vc4.  The compute is on the QPU; the
Python/ARM side only

  * reshapes/views (metadata), and
  * moves bytes (channel padding, phase images, weight permutes, concat,
    nearest resampling, compaction) and materialises constants,

all through ``torch.ops.torch_vc4.*`` (csrc/torch_vc4.cpp).  There is no CPU
maths and no CPU fallback: a shape or attribute the engine cannot do raises.

Batch size 1 only: the engine runs one image per launch and this backend does
not hide a host loop over the batch.
"""

import math

import torch

T = torch.ops.torch_vc4

_LIBS = []


def _keep(lib):
    _LIBS.append(lib)
    return lib


__all__ = [
    "register",
    "conv2d",
    "conv2d_backward",
    "channel_sums",
    "fold_bn_into_conv",
]


# --------------------------------------------------------------------- checks

def f32(t, what):
    if t.device.type != "vc4":
        raise RuntimeError(
            "torch_vc4: %s: expected a vc4 tensor, got %s" % (what, t.device))
    if t.dtype != torch.float32:
        raise RuntimeError("torch_vc4: %s: only float32 is supported, got %s"
                           % (what, t.dtype))
    if not t.is_contiguous():
        raise RuntimeError(
            "torch_vc4: %s: needs a contiguous tensor; this backend has no "
            "strided kernels (slicing/permute give metadata-only views that "
            "compute ops reject)" % what)
    return t


def _align16(v):
    return (v + 15) & ~15


def _pair(v):
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise RuntimeError("torch_vc4: expected a pair, got %r" % (v,))
        return int(v[0]), int(v[1])
    return int(v), int(v)


# ------------------------------------------------------------------- conv2d

def phase_q(pad):
    """Phase-conv padding for the stride-2 decomposition (docs/ONNX.md §6).

    For a stride-2, 3x3 conv with padding ``pad``, tap ``k`` reads input row
    ``2i + k - pad``, i.e. parity ``a = (k - pad) mod 2`` at phase offset
    ``d = (k - pad - a)/2``.  The phase conv therefore needs padding
    ``q = max(0, -min(d))`` so that every read lands inside the phase image.
    """
    ds = []
    for k in (0, 1, 2):
        u = k - pad
        a = u % 2
        ds.append((u - a) // 2)
    return max(0, -min(ds))


def _conv_geom(x, w, b, stride, pad, groups):
    f32(x, "conv2d(input)")
    f32(w, "conv2d(weight)")
    if x.dim() != 4 or w.dim() != 4:
        raise RuntimeError("torch_vc4: conv2d supports 4-D NCHW tensors only, "
                           "got %d-D input and %d-D weight"
                           % (x.dim(), w.dim()))
    N, C, H, W = x.shape
    Cout, Cin, kh, kw = w.shape
    if N != 1:
        raise RuntimeError("torch_vc4: conv2d runs batch size 1 (got %d); the "
                           "engine processes one image per launch" % N)
    depthwise = groups != 1
    if depthwise:
        if groups != C:
            raise RuntimeError(
                "torch_vc4: conv2d supports groups=1 and depthwise "
                "groups=Cin, got groups=%d with Cin=%d" % (groups, C))
        if not (Cout == C and Cin == 1):
            raise RuntimeError(
                "torch_vc4: depthwise conv2d needs Cout == Cin and a "
                "[C][1][k][k] weight (multiplier 1); got %d -> %d"
                % (C, Cout))
    elif Cin != C:
        raise RuntimeError("torch_vc4: conv2d: weight has %d input channels, "
                           "input has %d" % (Cin, C))
    if kh != kw or kh not in (1, 3):
        raise RuntimeError("torch_vc4: conv2d supports 1x1 and 3x3 kernels, "
                           "got %dx%d" % (kh, kw))
    if stride not in (1, 2):
        raise RuntimeError("torch_vc4: conv2d supports stride 1 and 2, got %d"
                           % stride)
    if kh == 1 and (pad != 0 or stride not in (1, 2)):
        raise RuntimeError("torch_vc4: conv2d 1x1 takes padding 0")
    if pad < 0 or pad > kh // 2:
        raise RuntimeError("torch_vc4: conv2d padding %d is outside PyTorch's "
                           "range for a %dx%d kernel" % (pad, kh, kh))
    if b is not None:
        f32(b, "conv2d(bias)")
        if b.dim() != 1 or b.numel() != Cout:
            raise RuntimeError("torch_vc4: conv2d bias must be [%d], got %s"
                               % (Cout, tuple(b.shape)))
    return C, H, W, Cout, kh, stride, pad, depthwise


def _padded(w, b, C, Cout, depthwise):
    """Weight/bias in the engine's padded layout (byte copies only)."""
    cpad, opad = _align16(C), _align16(Cout)
    if depthwise:
        wd = T.dense_dw_w(w, cpad, opad)
    elif cpad != C or opad != Cout:
        wd = T.pad_w(w, cpad, opad)
    else:
        wd = w
    bp = b
    if b is not None and opad != Cout:
        bp = T.pad_ch(b, opad)
    return wd, bp, cpad, opad


def conv2d(x, w, b, stride=1, pad=0, groups=1):
    C, H, W, Cout, k, stride, pad, depthwise = _conv_geom(
        x, w, b, stride, pad, groups)
    wd, bp, cpad, opad = _padded(w, b, C, Cout, depthwise)
    x3 = x.view(C, H, W)
    xp = T.pad_ch(x3, cpad) if cpad != C else x3
    if stride == 1:
        y3 = T.conv_raw(xp, wd, bp, pad)
    elif k == 1:
        # stride-2 1x1 == 1x1 over the (0,0) parity phase: exact, no waste
        y3 = T.conv_raw(T.subsample2(xp), wd, bp, 0)
    else:
        q = phase_q(pad)
        ho = (H + 2 * pad - 3) // 2 + 1
        wo = (W + 2 * pad - 3) // 2 + 1
        hp, wp = ho + 2 - 2 * q, wo + 2 - 2 * q
        ph = T.phase_split(xp, hp, wp)
        y3 = T.conv_raw(ph, T.phase_w(wd, cpad, opad, pad, q), bp, q)
    if y3.shape[0] != Cout:
        y3 = T.take_ch(y3, 0, Cout)
    return y3.view(1, Cout, y3.shape[1], y3.shape[2])


def conv2d_backward(x, w, dy, stride=1, pad=0, groups=1):
    """dX, dW for the forward above, using the engine's dX/dW kernels."""
    C, H, W, Cout, k, stride, pad, depthwise = _conv_geom(
        x, w, None, stride, pad, groups)
    f32(dy, "conv2d_backward(dy)")
    ho = (H + 2 * pad - k) // stride + 1
    wo = (W + 2 * pad - k) // stride + 1
    if tuple(dy.shape) != (1, Cout, ho, wo):
        raise RuntimeError("torch_vc4: conv2d_backward: dy %s does not match "
                           "%s" % (tuple(dy.shape), (1, Cout, ho, wo)))
    wd, _, cpad, opad = _padded(w, None, C, Cout, depthwise)
    x3 = x.view(C, H, W)
    xp = T.pad_ch(x3, cpad) if cpad != C else x3
    dy3 = dy.view(Cout, ho, wo)
    dyp = T.pad_ch(dy3, opad) if opad != Cout else dy3
    if stride == 1:
        dxp, dwp = T.conv_bwd_raw(xp, wd, dyp, pad)
        dx3 = T.take_ch(dxp, 0, C) if cpad != C else dxp
    elif k == 1:
        dxs, dwp = T.conv_bwd_raw(T.subsample2(xp), wd, dyp, 0)
        dx3 = T.scatter2(dxs, H, W)
        if cpad != C:
            dx3 = T.take_ch(dx3, 0, C)
    else:
        q = phase_q(pad)
        hp, wp = ho + 2 - 2 * q, wo + 2 - 2 * q
        ph = T.phase_split(xp, hp, wp)
        dpp, dwp = T.conv_bwd_raw(ph, T.phase_w(wd, cpad, opad, pad, q),
                                  dyp, q)
        dx3 = T.phase_merge(dpp, cpad, H, W)
        if cpad != C:
            dx3 = T.take_ch(dx3, 0, C)
    if stride == 1 or k == 1:
        dw = T.w_block(dwp, Cout, C, depthwise)
    else:
        dw = T.w_phase_block(dwp, Cout, C, pad, phase_q(pad), depthwise)
    return dx3.view(1, C, H, W), dw


def channel_sums(t3):
    """Sum a [C][H][W] tensor over H,W -> [C], as one GEMM with a ones column."""
    C, H, W = t3.shape
    ones = torch.ones((H * W, 1), dtype=torch.float32, device=t3.device)
    return torch.mm(t3.reshape(C, H * W), ones).view(C)


# --------------------------------------------------------------- pooling

def max_pool2d(x, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False):
    f32(x, "max_pool2d")
    if x.dim() == 3:
        C, H, W = x.shape
        lead = ()
    elif x.dim() == 4:
        N, C, H, W = x.shape
        if N != 1:
            raise RuntimeError("torch_vc4: max_pool2d runs batch size 1")
        lead = (1,)
    else:
        raise RuntimeError("torch_vc4: max_pool2d expects 3-D or 4-D input")
    k = _pair(kernel_size)
    s = _pair(stride if stride not in (None, []) else kernel_size)
    p = _pair(padding)
    d = _pair(dilation)
    if k[0] != k[1] or s[0] != s[1] or p[0] != p[1]:
        raise RuntimeError("torch_vc4: max_pool2d needs a square kernel, "
                           "stride and padding")
    if d != (1, 1):
        raise RuntimeError("torch_vc4: max_pool2d dilation != 1 is not "
                           "implemented on the engine")
    if ceil_mode:
        raise RuntimeError("torch_vc4: max_pool2d ceil_mode is not implemented")
    if p[0] > k[0] // 2:
        raise RuntimeError("torch_vc4: max_pool2d padding %d > kernel/2"
                           % p[0])
    x3 = x.view(C, H, W)
    if k[0] == 2 and s[0] == 2 and p[0] == 0 and H % 8 == 0 and W % 16 == 0:
        y3 = T.maxpool2x2(x3)          # 2x2 argmax-mask kernel (has backward)
    else:
        y3 = T.maxpooln(x3, k[0], s[0], p[0])   # generic NxN/stride kernel
    return y3.view(lead + tuple(y3.shape))


def max_pool2d_backward(x, dy, k, s, p):
    f32(x, "max_pool2d(backward)")
    f32(dy, "max_pool2d(backward, dy)")
    C, H, W = x.view(x.shape[-3], x.shape[-2], x.shape[-1]).shape
    if not (k == 2 and s == 2 and p == 0):
        raise RuntimeError(
            "torch_vc4: the backward of max_pool2d exists only for the "
            "engine's 2x2/stride-2 kernel (got %dx%d stride %d pad %d); "
            "the generic NxN pool is forward-only" % (k, k, s, p))
    if H % 4 or W % 16:
        raise RuntimeError(
            "torch_vc4: max_pool2d backward needs H%%4==0 and W%%16==0 "
            "(engine task geometry), got %dx%d" % (H, W))
    dx3 = T.maxpool2x2_bwd(x.view(C, H, W), dy.view(C, H // 2, W // 2))
    return dx3.view(x.shape)


def adaptive_matrix(H, W, oh, ow):
    """The adaptive-average matrix A[o][p] = 1/count(window(o) covering p).

    Pure data preparation on the host (PyTorch's floor/ceil window rule); the
    averaging itself is a GEMM on the QPU in both directions.
    """
    a = torch.zeros((oh * ow, H * W), dtype=torch.float32)
    for i in range(oh):
        h0, h1 = (i * H) // oh, -((-(i + 1) * H) // oh)
        for j in range(ow):
            w0, w1 = (j * W) // ow, -((-(j + 1) * W) // ow)
            a[i * ow + j].view(H, W)[h0:h1, w0:w1] = 1.0 / ((h1 - h0) * (w1 - w0))
    return a


def adaptive_avg_pool2d(x, output_size):
    f32(x, "adaptive_avg_pool2d")
    if x.dim() == 3:
        C, H, W = x.shape
        lead = ()
    elif x.dim() == 4:
        N, C, H, W = x.shape
        if N != 1:
            raise RuntimeError("torch_vc4: adaptive_avg_pool2d runs batch 1")
        lead = (1,)
    else:
        raise RuntimeError("torch_vc4: adaptive_avg_pool2d expects 3-D/4-D")
    oh, ow = _pair(output_size)
    if oh <= 0 or ow <= 0:
        raise RuntimeError("torch_vc4: adaptive_avg_pool2d: empty output size")
    a = adaptive_matrix(H, W, oh, ow)          # [oh*ow][H*W] on the CPU
    avt = a.t().contiguous().to(x.device)      # upload: a byte copy
    flat = x.view(C, H * W)
    y = torch.mm(flat, avt)                    # [C][oh*ow], QPU GEMM
    return y.view(lead + (C, oh, ow))


# --------------------------------------------------------------- softmax

def softmax(x, dim, half_to_float=False):
    f32(x, "softmax")
    if x.dim() == 0:
        raise RuntimeError("torch_vc4: softmax of a scalar")
    if dim != x.dim() - 1 and dim != -1:
        raise RuntimeError("torch_vc4: softmax is implemented on the last dim "
                           "only (got dim=%d of %d)" % (dim, x.dim()))
    cols = x.shape[-1]
    if cols > 16:
        raise RuntimeError("torch_vc4: softmax supports at most 16 classes on "
                           "the last dim (engine lane width), got %d" % cols)
    y = T.softmax(x.view(-1, cols))
    return y.view(x.shape)


# --------------------------------------------------------------- elementwise

def sigmoid(x):
    f32(x, "sigmoid")
    return T.sigmoid(x)


def silu(x):
    f32(x, "silu")
    return T.silu(x)


# ---------------------------------------------------------------- views

def view(x, size):
    f32(x, "view")
    return T.view_alias(x, list(size))


def flatten(x, start_dim=0, end_dim=-1):
    f32(x, "flatten")
    nd = x.dim()
    if nd == 0:
        return x.view(1)
    s = start_dim if start_dim >= 0 else start_dim + nd
    e = end_dim if end_dim >= 0 else end_dim + nd
    if s < 0 or e >= nd or s > e:
        raise RuntimeError("torch_vc4: flatten: bad dims %d..%d of %d"
                           % (start_dim, end_dim, nd))
    shape = list(x.shape)
    merged = 1
    for i in range(s, e + 1):
        merged *= shape[i]
    return x.view(shape[:s] + [merged] + shape[e + 1:])


def as_strided(x, size, stride, storage_offset=None):
    f32(x, "as_strided")
    return T.as_strided_alias(x, list(size), list(stride),
                              None if storage_offset is None
                              else int(storage_offset))


# ---------------------------------------------------------------- concat

def cat(tensors, dim=0):
    if not tensors:
        raise RuntimeError("torch_vc4: cat of an empty list")
    for t in tensors:
        f32(t, "cat")
    t0 = tensors[0]
    if t0.dim() == 4:
        if dim not in (1, -3):
            raise RuntimeError("torch_vc4: cat is implemented on the channel "
                               "axis of NCHW tensors (dim=1), got dim=%d" % dim)
        parts = [t.view(t.shape[1], t.shape[2], t.shape[3]) for t in tensors]
        y = T.cat_ch(parts)
        return y.view(1, y.shape[0], y.shape[1], y.shape[2])
    if t0.dim() == 3 and dim in (0, -3):
        return T.cat_ch(list(tensors))
    raise RuntimeError("torch_vc4: cat: unsupported rank %d / dim %d"
                       % (t0.dim(), dim))


# --------------------------------------------------------------- resize

def upsample_nearest2d(x, output_size, scales_h=None, scales_w=None):
    f32(x, "upsample_nearest2d")
    if x.dim() != 4:
        raise RuntimeError("torch_vc4: upsample_nearest2d expects NCHW")
    N, C, H, W = x.shape
    if N != 1:
        raise RuntimeError("torch_vc4: upsample_nearest2d runs batch 1")
    ho, wo = int(output_size[0]), int(output_size[1])
    y = T.up_nearest(x.view(C, H, W), ho, wo,
                     float(scales_h) if scales_h else 0.0,
                     float(scales_w) if scales_w else 0.0)
    return y.view(1, C, ho, wo)


def upsample_nearest2d_vec(x, output_size, scale_factors):
    if output_size is not None:
        return upsample_nearest2d(x, output_size)
    if scale_factors is None:
        raise RuntimeError("torch_vc4: upsample_nearest2d needs a size or "
                           "scale factors")
    H, W = x.shape[-2], x.shape[-1]
    sh, sw = float(scale_factors[0]), float(scale_factors[1])
    ho = int(math.floor(H * sh))
    wo = int(math.floor(W * sw))
    return upsample_nearest2d(x, (ho, wo), sh, sw)


# --------------------------------------------------------------- batch norm

def _bn(x, weight, bias, mean, var, training, eps, momentum, cudnn=False):
    if training:
        raise RuntimeError(
            "torch_vc4: batch_norm is implemented in inference mode only "
            "(running statistics); training-mode batch norm would need a "
            "per-channel reduction kernel the engine does not have.  Use "
            "module.eval() and fold the BN instead (see fold_bn_into_conv).")
    f32(x, "batch_norm")
    if mean is None or var is None:
        raise RuntimeError("torch_vc4: batch_norm needs running_mean and "
                           "running_var in inference mode")
    if x.dim() != 4:
        raise RuntimeError("torch_vc4: batch_norm expects NCHW")
    N, C, H, W = x.shape
    if N != 1:
        raise RuntimeError("torch_vc4: batch_norm runs batch 1")
    f32(mean, "batch_norm(running_mean)")
    f32(var, "batch_norm(running_var)")
    if mean.numel() != C or var.numel() != C:
        raise RuntimeError("torch_vc4: batch_norm running stats have %d/%d "
                           "elements, expected %d" % (mean.numel(),
                                                      var.numel(), C))
    # scale = gamma / sqrt(var + eps); shift = beta - mean * scale.
    # Both vectors are C long and every elementwise op below is a QPU kernel;
    # the only host work is broadcasting them to [C][H][W] (bytes).
    v = T.add_scalar(var, eps) if eps else var
    inv = T.rsqrt(v)
    scale = T.mul(inv, weight) if weight is not None else inv
    shift = T.mul(scale, mean)
    shift = T.mul_scalar(shift, -1.0)
    if bias is not None:
        f32(bias, "batch_norm(bias)")
        shift = T.add(shift, bias)
    sb = T.bcast_ch(scale, H, W)
    hb = T.bcast_ch(shift, H, W)
    y = T.add(T.mul(x.view(C, H, W), sb), hb)
    empty = torch.empty((0,), dtype=torch.float32, device=x.device)
    return y.view(N, C, H, W), empty, empty


def native_batch_norm(x, weight, bias, mean, var, training, momentum, eps):
    return _bn(x, weight, bias, mean, var, bool(training), float(eps),
               float(momentum))


def batch_norm(x, weight, bias, mean, var, training, momentum, eps,
               cudnn_enabled=False):
    return _bn(x, weight, bias, mean, var, bool(training), float(eps),
               float(momentum), bool(cudnn_enabled))[0]


# ---------------------------------------------------------------- linear

def linear(x, weight, bias=None):
    f32(x, "linear(input)")
    f32(weight, "linear(weight)")
    if weight.dim() != 2:
        raise RuntimeError("torch_vc4: linear expects a 2-D weight")
    out_f, in_f = weight.shape
    if x.dim() == 0:
        raise RuntimeError("torch_vc4: linear of a scalar")
    if x.shape[-1] != in_f:
        raise RuntimeError("torch_vc4: linear: input %s vs weight %s"
                           % (tuple(x.shape), tuple(weight.shape)))
    lead = tuple(x.shape[:-1])
    x2 = x.view(-1, in_f)
    y = T.mm_bt(x2, weight)          # x @ W^T on the QPU (pads to 16)
    if bias is not None:
        f32(bias, "linear(bias)")
        y = T.add(y, T.bcast_rows(bias, y.shape[0]))
    return y.view(lead + (out_f,))


# --------------------------------------------------------------- reductions

_SUM_CHUNK = 4096


def sum_all(t, dtype=None):
    """Full reduction as GEMM(s) with a host-materialised ones vector.

    One GEMM over the whole tensor would need a 64-padded B panel of
    64*n floats (16 MiB at n = 64 Ki), which does not fit the engine's
    scratch next to a 24 MiB tensor region; the reduction is therefore done
    in 4096-element chunks (one launch each) and summed with GPU adds.
    """
    f32(t, "sum")
    if dtype is not None and dtype != torch.float32:
        raise RuntimeError("torch_vc4: sum(dtype=...) is not supported")
    flat = t.view(-1)
    n = flat.numel()
    if n <= _SUM_CHUNK:
        return T.sum_all(flat).view([])
    acc = None
    for i in range(0, n, _SUM_CHUNK):
        part = T.sum_all(flat[i:i + _SUM_CHUNK])
        acc = part if acc is None else T.add(acc, part)
    return acc.view([])


def contiguous(t):
    """A contiguous tensor with the same values (byte copy when needed)."""
    f32(t, "contiguous")
    return t if t.is_contiguous() else T.contiguous(t)


def mean_all(t, dtype=None):
    f32(t, "mean")
    if t.numel() == 0:
        raise RuntimeError("torch_vc4: mean of an empty tensor")
    return T.mul_scalar(sum_all(t), 1.0 / t.numel())


# ------------------------------------------------------------- fold helper

def fold_bn_into_conv(weight, bias, mean, var, gamma, beta, eps=1e-5):
    """Inference-time constant folding: conv(x, w', b') == bn(conv(x, w, b)).

    Parameters are CPU tensors here on purpose: folding is a one-off on the
    *parameters*, not on activations, so it is host arithmetic by design.
    Returns (w', b') as contiguous CPU tensors; upload them to vc4 with .to().
    """
    if weight.dim() != 4:
        raise RuntimeError("torch_vc4: fold_bn_into_conv expects [Cout][Cin]"
                           "[kh][kw]")
    scale = (gamma / torch.sqrt(var + eps)).view(-1, 1, 1, 1)
    w2 = weight * scale
    b2 = (bias - mean) * scale.view(-1) + beta
    return w2.contiguous(), b2.contiguous()


# ------------------------------------------------------------- registration

def register():
    # torch deregisters a Library when it is collected; keep it alive
    lib = _keep(torch.library.Library("aten", "IMPL", "PrivateUse1"))

    def impl(name, fn):
        lib.impl(name, fn)

    impl("convolution_overrideable",
         lambda x, w, b, stride, pad, dil, transposed, out_pad, groups:
         _conv_op(x, w, b, stride, pad, dil, transposed, out_pad, groups))
    impl("convolution_backward_overrideable", convolution_backward)
    impl("max_pool2d", max_pool2d)
    impl("_adaptive_avg_pool2d", adaptive_avg_pool2d)
    impl("adaptive_avg_pool2d", adaptive_avg_pool2d)
    impl("_softmax", softmax)
    impl("sigmoid", sigmoid)
    impl("silu", silu)
    # aten::view/aten::as_strided are registered natively (csrc) -- the
    # Python layer only supplies the autograd Functions for them.
    impl("flatten.using_ints", flatten)
    impl("cat", cat)
    impl("upsample_nearest2d", upsample_nearest2d)
    impl("upsample_nearest2d.vec", upsample_nearest2d_vec)
    impl("native_batch_norm", native_batch_norm)
    impl("batch_norm", batch_norm)
    impl("linear", linear)
    impl("sum", sum_all)
    impl("mean", mean_all)
    # (the Tensor overloads above are re-registered here so that a scalar
    # operand in the *second* position never reaches the two-tensor kernels)
    impl("mul.Scalar", lambda x, s: T.mul_scalar(x, float(s)))
    impl("div.Scalar", lambda x, s: T.mul_scalar(x, 1.0 / float(s)))
    impl("div.Tensor", _div_tensor)
    impl("add.Tensor", _add_tensor)
    impl("mul.Tensor", _mul_tensor)
    impl("sub.Tensor", _sub_tensor)
    impl("add_.Tensor", _add_inplace)
    impl("add.Scalar", lambda x, s, alpha=1: T.add_scalar(x, float(s) * alpha))
    impl("sub.Scalar", lambda x, s, alpha=1:
         T.add_scalar(x, -float(s) * alpha))
    impl("neg", lambda x: T.mul_scalar(x, -1.0))
    impl("pow.Tensor_Scalar", _pow_scalar)
    return lib


def _is_cpu_scalar(t):
    return torch.is_tensor(t) and t.device.type == "cpu" and t.numel() == 1


def _scalar_of(x):
    """The Python value of a scalar operand (Python number or CPU 0-dim)."""
    if isinstance(x, (int, float)):
        return float(x)
    if _is_cpu_scalar(x):
        return float(x)
    return None


def _div_tensor(a, b):
    """x / s is one GPU multiply by 1/s; x / y on device has no kernel."""
    f32(a, "div")
    s = _scalar_of(b)
    if s is not None:
        return T.mul_scalar(a, 1.0 / s)
    raise RuntimeError(
        "torch_vc4: division of two device tensors is not implemented (the "
        "engine has no reciprocal-divide kernel); divide by a host scalar or "
        "multiply by a precomputed reciprocal tensor on the CPU")


def _add_tensor(a, b, alpha=1):
    f32(a, "add")
    s = _scalar_of(b)
    if s is not None:
        return T.add_scalar(a, s * float(alpha))
    return T.add(a, b, alpha)


def _mul_tensor(a, b):
    f32(a, "mul")
    s = _scalar_of(b)
    if s is not None:
        return T.mul_scalar(a, s)
    return T.mul(a, b)


def _sub_tensor(a, b, alpha=1):
    f32(a, "sub")
    s = _scalar_of(b)
    if s is not None:
        return T.add_scalar(a, -s * float(alpha))
    return T.add(a, T.mul_scalar(b, -float(alpha)))


def _add_inplace(self, other, alpha=1):
    """p.add_(g, alpha) for the autograd engine's gradient accumulation."""
    out = _add_scalar_alpha(self, other, alpha)
    self.copy_(out)
    return self


def _add_scalar_alpha(a, b, alpha):
    f32(a, "add_")
    if alpha == 1:
        return T.add(a, b)
    return T.add(a, T.mul_scalar(b, float(alpha)))


def convolution_backward(g, x, w, stride, pad, dil, transposed, out_pad,
                        groups, output_mask):
    """The backward hook torch's own conv autograd calls for custom backends
    (``aten::convolution_backward_overrideable``).

    The whole convolution family in torch funnels through
    ``aten::_convolution``, whose generated autograd node asks the backend for
    ``convolution_backward_overrideable``.  Serving it means ``F.conv2d`` and
    ``nn.Conv2d`` train through the standard torch autograd engine, with the
    engine's dX/dW kernels underneath.
    """
    if transposed:
        raise RuntimeError("torch_vc4: transposed conv backward is not "
                           "implemented")
    if _pair(dil) != (1, 1):
        raise RuntimeError("torch_vc4: conv dilation != 1 is not implemented")
    s, p = _pair(stride), _pair(pad)
    if s[0] != s[1] or p[0] != p[1]:
        raise RuntimeError("torch_vc4: conv needs square stride/padding")
    f32(g, "conv2d_backward(grad)")
    want_x, want_w, want_b = (bool(v) for v in output_mask)
    dx = dw = db = None
    if want_x or want_w:
        dx_, dw_ = conv2d_backward(x, w, g, s[0], p[0], int(groups))
        dx = dx_ if want_x else None
        dw = dw_ if want_w else None
    if want_b:
        c, ho, wo = g.shape[1], g.shape[2], g.shape[3]
        db = channel_sums(g.view(c, ho, wo))
    return dx, dw, db


def _conv_op(x, w, b, stride, pad, dil, transposed, out_pad, groups):
    if transposed:
        raise RuntimeError("torch_vc4: transposed convolution is not "
                           "implemented")
    if any(int(v) for v in out_pad):
        raise RuntimeError("torch_vc4: conv output_padding is not implemented")
    if _pair(dil) != (1, 1):
        raise RuntimeError("torch_vc4: conv dilation != 1 is not implemented "
                           "on the engine")
    s = _pair(stride)
    p = _pair(pad)
    if s[0] != s[1]:
        raise RuntimeError("torch_vc4: conv needs a square stride, got %s"
                           % (tuple(s),))
    if p[0] != p[1]:
        raise RuntimeError("torch_vc4: conv needs symmetric padding, got %s"
                           % (tuple(p),))
    return conv2d(x, w, b, s[0], p[0], int(groups))


def _pow_scalar(x, exponent):
    if float(exponent) != 2.0:
        raise RuntimeError("torch_vc4: pow is implemented for exponent 2 only "
                           "(x*x); got %r" % (exponent,))
    f32(x, "pow")
    return T.mul(x, x)

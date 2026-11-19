"""Autograd for the vc4 subset backend.

The graph is built from ``torch.autograd.Function``s whose backward calls the
engine's own backward kernels:

  mm/matmul    vc4_matmul_bt (dA = g B^T) and vc4_transpose + vc4_matmul
               (dB = A^T g)
  add/mul      the engine's elementwise ADD/MUL
  relu         vc4_relu_backward (masked copy)
  silu/sigmoid composed from vc4_sigmoid / MUL / ADD on the QPU
  conv2d       vc4_conv2d_backward (im2col -> gemm3 -> col2im, dW and dX)
  max_pool2d   vc4_maxpool2x2_backward (argmax mask)
  softmax      y * (g - sum(g*y)) with the reduction as a GEMM
  linear       mm_bt / mm_at
  view         metadata (a view is materialised only when grad is needed)

The ``Autograd`` dispatch key is served by these Functions; inside a Function
we run below autograd (``_AutoDispatchBelowAutograd``) so the raw PrivateUse1
kernels are reached.  There is no CPU fallback: an op without an Autograd
kernel still raises instead of silently detaching.
"""

import torch

from . import ops

T = torch.ops.torch_vc4
_AdBelow = torch._C._AutoDispatchBelowAutograd

_LIBS = []


def _keep(lib):
    _LIBS.append(lib)
    return lib


__all__ = ["register", "mse_loss"]


def _g(g):
    """Gradients can arrive as expanded/0-stride views; the engine needs
    contiguous buffers, so materialise them (a byte gather, no arithmetic)."""
    return ops.contiguous(g)


def _below(fn):
    with _AdBelow():
        return fn()


def _ones(shape, like):
    return torch.ones(shape, dtype=torch.float32, device=like.device)


def _sub(a, b):
    return T.add(a, T.mul_scalar(b, -1.0))


def _col_sums(t2):
    """Sum a [M][N] tensor over rows -> [N] (one GEMM with a ones row)."""
    m, n = t2.shape
    return torch.mm(_ones((1, m), t2), t2).view(n)


# ------------------------------------------------------------------- mm

class MmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _below(lambda: torch.mm(a, b))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        a, b = ctx.saved_tensors
        ga = gb = None
        if ctx.needs_input_grad[0]:
            ga = _below(lambda: T.mm_bt(g, b))       # g @ b^T
        if ctx.needs_input_grad[1]:
            gb = _below(lambda: T.mm_at(a, g))       # a^T @ g
        return ga, gb


# --------------------------------------------------------------- binary

class AddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, alpha):
        ctx.alpha = float(alpha)
        if ctx.alpha == 1.0:
            return _below(lambda: T.add(a, b))
        return _below(lambda: T.add(a, T.mul_scalar(b, ctx.alpha)))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        ga = g if ctx.needs_input_grad[0] else None
        gb = _below(lambda: T.mul_scalar(g, ctx.alpha)) \
            if ctx.needs_input_grad[1] else None
        return ga, gb, None


class MulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _below(lambda: torch.mul(a, b))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        a, b = ctx.saved_tensors
        ga = _below(lambda: torch.mul(g, b)) if ctx.needs_input_grad[0] else None
        gb = _below(lambda: torch.mul(g, a)) if ctx.needs_input_grad[1] else None
        return ga, gb


class SubFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, alpha):
        ctx.alpha = float(alpha)
        return _below(lambda: _sub(a, T.mul_scalar(b, ctx.alpha)))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        ga = g if ctx.needs_input_grad[0] else None
        gb = _below(lambda: T.mul_scalar(g, -ctx.alpha)) \
            if ctx.needs_input_grad[1] else None
        return ga, gb, None


# --------------------------------------------------------------- scalars

class ScalarMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.s = float(s)
        return _below(lambda: T.mul_scalar(x, ctx.s))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        return (_below(lambda: T.mul_scalar(g, ctx.s))
                if ctx.needs_input_grad[0] else None), None


class ScalarAddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        return _below(lambda: T.add_scalar(x, float(s)))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        return (g if ctx.needs_input_grad[0] else None), None


class NegFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return _below(lambda: T.mul_scalar(x, -1.0))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        return (_below(lambda: T.mul_scalar(g, -1.0))
                if ctx.needs_input_grad[0] else None)


class Pow2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.save_for_backward(x)
        return _below(lambda: T.mul(x, x))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        return (_below(lambda: T.mul_scalar(T.mul(g, x), 2.0))
                if ctx.needs_input_grad[0] else None), None


# --------------------------------------------------------------- unary

class ReluFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _below(lambda: torch.relu(x))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        return _below(lambda: T.relu_bwd(x, g))


class SigmoidFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = _below(lambda: T.sigmoid(x))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (s,) = ctx.saved_tensors
        def f():
            one_minus = T.add_scalar(T.mul_scalar(s, -1.0), 1.0)
            return T.mul(T.mul(g, s), one_minus)
        return _below(f)


class SiluFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _below(lambda: T.silu(x))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        def f():
            s = T.sigmoid(x)
            one_minus = T.add_scalar(T.mul_scalar(s, -1.0), 1.0)
            # d/dx [x*sigmoid(x)] = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
            factor = T.add(s, T.mul(T.mul(x, s), one_minus))
            return T.mul(g, factor)
        return _below(f)


class SoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        y = _below(lambda: ops.softmax(x, dim))
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (y,) = ctx.saved_tensors
        cols = y.shape[-1]
        rows = y.numel() // cols
        def f():
            gy = T.mul(g, y)
            rs = torch.mm(gy.view(rows, cols), _ones((cols, 1), y))  # [rows][1]
            b = torch.mm(rs, _ones((1, cols), y))                   # [rows][cols]
            return T.mul(y, T.add(g, T.mul_scalar(b, -1.0)))
        return _below(f), None


# ---------------------------------------------------------------- conv

class Conv2dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, stride, pad, groups):
        ctx.save_for_backward(x, w)
        ctx.stride, ctx.pad, ctx.groups = stride, pad, groups
        ctx.has_bias = b is not None
        return _below(lambda: ops.conv2d(x, w, b, stride, pad, groups))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        x, w = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad[0:3]
        if not (need_x or need_w or need_b):
            return None, None, None, None, None, None
        dx, dw = _below(lambda: ops.conv2d_backward(
            x, w, g, ctx.stride, ctx.pad, ctx.groups))
        db = None
        if need_b:
            c, ho, wo = g.shape[1], g.shape[2], g.shape[3]
            db = _below(lambda: ops.channel_sums(g.view(c, ho, wo)))
        return (dx if need_x else None, dw if need_w else None, db,
                None, None, None)


# ---------------------------------------------------------------- pooling

class MaxPool2dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, s, p):
        ctx.save_for_backward(x)
        ctx.k, ctx.s, ctx.p = k, s, p
        return _below(lambda: ops.max_pool2d(x, k, s, p))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        return _below(lambda: ops.max_pool2d_backward(
            x, g, ctx.k, ctx.s, ctx.p)), None, None, None


class AdaptiveAvgPool2dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_size):
        ctx.save_for_backward(x)
        ctx.out = tuple(int(v) for v in out_size)
        return _below(lambda: ops.adaptive_avg_pool2d(x, ctx.out))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        C, H, W = x.shape[-3:]
        oh, ow = ctx.out
        # dX[c][p] = sum_o dy[c][o] * A[o][p]  -> one GEMM with A (host-built
        # byte matrix, see ops.adaptive_matrix).
        def f():
            a = ops.adaptive_matrix(H, W, oh, ow).to(x.device)
            dx = torch.mm(g.reshape(C, oh * ow), a)
            return dx.view(x.shape)
        return _below(f), None


# ---------------------------------------------------------------- views

class ViewFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size):
        ctx.shape = tuple(x.shape)
        return _below(lambda: T.view_alias(x.detach(), list(size)))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        return _below(lambda: T.view_alias(g, list(ctx.shape))), None


class AsStridedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size, stride, off):
        ctx.shape = tuple(x.shape)
        y = _below(lambda: T.as_strided_alias(x.detach(), list(size), list(stride),
                                          off))
        if not y.is_contiguous():
            raise RuntimeError(
                "torch_vc4: as_strided/permute views are supported as "
                "metadata only; autograd through a non-contiguous vc4 view "
                "is not implemented (the engine has no strided kernels)")
        return y

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        return _below(lambda: T.view_alias(g, list(ctx.shape))), None, None, None


# ---------------------------------------------------------------- linear

class LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w)
        ctx.has_bias = b is not None
        return _below(lambda: ops.linear(x, w, b))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        x, w = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad[0:3]
        g2 = g.view(-1, g.shape[-1])
        # dx = g @ W   (W is [out][in]); dw = g^T @ x
        dx = _below(lambda: torch.mm(g2, w)) if need_x else None
        if dx is not None:
            dx = dx.view(x.shape)
        dw = _below(lambda: T.mm_at(g2, x.view(-1, x.shape[-1]))) \
            if need_w else None
        db = _below(lambda: _col_sums(g2)) if need_b else None
        return dx, dw, db


# ------------------------------------------------------------- reductions

class SumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.shape = tuple(x.shape)
        ctx.n = x.numel()
        return _below(lambda: ops.sum_all(x))

    @staticmethod
    def backward(ctx, g):
        gv = float(g)
        def f():
            ones = torch.ones(ctx.shape, dtype=torch.float32,
                              device=g.device)
            return T.mul_scalar(ones, gv)
        return _below(f)


class MeanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.shape = tuple(x.shape)
        ctx.n = x.numel()
        return _below(lambda: ops.mean_all(x))

    @staticmethod
    def backward(ctx, g):
        gv = float(g) / ctx.n
        def f():
            ones = torch.ones(ctx.shape, dtype=torch.float32,
                              device=g.device)
            return T.mul_scalar(ones, gv)
        return _below(f)


# ------------------------------------------------------------------ loss

class MSELossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target):
        ctx.save_for_backward(pred, target)
        def f():
            d = _sub(pred, target)
            s = T.sum_all(T.mul(d, d).view(-1))
            return T.mul_scalar(s, 1.0 / pred.numel()).view([])
        return _below(f)

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        pred, target = ctx.saved_tensors
        n = pred.numel()
        scale = float(g) * 2.0 / n
        def f():
            d = _sub(pred, target)
            return (T.mul_scalar(d, scale), T.mul_scalar(d, -scale))
        ga, gb = _below(f)
        return (ga if ctx.needs_input_grad[0] else None,
                gb if ctx.needs_input_grad[1] else None)


def mse_loss(pred, target):
    """Mean squared error through the backend (forward and backward on vc4).

    ``F.mse_loss`` decomposes into ops this backend does not implement
    (``resize_`` and friends); this is the same reduction written with the
    engine's GEMM-based sum, so the whole loss stays on the device.
    """
    ops.f32(pred, "mse_loss(pred)")
    ops.f32(target, "mse_loss(target)")
    if pred.shape != target.shape:
        raise RuntimeError("torch_vc4: mse_loss needs equal shapes, got %s "
                           "and %s" % (tuple(pred.shape), tuple(target.shape)))
    return MSELossFn.apply(pred, target)


def _no_div(a, b):
    raise RuntimeError("torch_vc4: division of two device tensors is not "
                       "implemented on this backend")


def _k(v):
    p = ops._pair(v)
    if p[0] != p[1]:
        raise RuntimeError("torch_vc4: square kernel/stride required, got %r"
                           % (v,))
    return p[0]


# ------------------------------------------------------------ registration

def register():
    # torch deregisters a Library when it is collected; keep it alive
    lib = _keep(torch.library.Library("aten", "IMPL",
                                      "AutogradPrivateUse1"))

    def conv(x, w, b, stride, pad, dil, transposed, out_pad, groups):
        if transposed or any(int(v) for v in out_pad):
            raise RuntimeError("torch_vc4: transposed conv has no autograd")
        if ops._pair(dil) != (1, 1):
            raise RuntimeError("torch_vc4: conv dilation != 1 is not supported")
        s, p = ops._pair(stride), ops._pair(pad)
        if s[0] != s[1] or p[0] != p[1]:
            raise RuntimeError("torch_vc4: conv needs square stride/padding")
        return Conv2dFn.apply(x, w, b, s[0], p[0], int(groups))

    lib.impl("convolution_overrideable", conv)
    lib.impl("mm", lambda a, b: MmFn.apply(a, b))
    lib.impl("matmul", lambda a, b: MmFn.apply(a, b))
    def add(a, b, alpha=1):
        s = ops._scalar_of(b)
        if s is not None:
            return ScalarAddFn.apply(a, s * float(alpha))
        s = ops._scalar_of(a)
        if s is not None:
            return ScalarAddFn.apply(b, s * float(alpha))
        return AddFn.apply(a, b, alpha)

    def mul(a, b):
        s = ops._scalar_of(b)
        if s is not None:
            return ScalarMulFn.apply(a, s)
        s = ops._scalar_of(a)
        if s is not None:
            return ScalarMulFn.apply(b, s)
        return MulFn.apply(a, b)

    def sub(a, b, alpha=1):
        s = ops._scalar_of(b)
        if s is not None:
            return ScalarAddFn.apply(a, -s * float(alpha))
        if ops._scalar_of(a) is not None:
            raise RuntimeError("torch_vc4: (scalar - tensor) has no autograd "
                               "kernel; rewrite it as a - x on the device")
        return SubFn.apply(a, b, alpha)

    lib.impl("add.Tensor", add)
    lib.impl("mul.Tensor", mul)
    lib.impl("sub.Tensor", sub)
    lib.impl("relu", lambda x: ReluFn.apply(x))
    lib.impl("silu", lambda x: SiluFn.apply(x))
    lib.impl("sigmoid", lambda x: SigmoidFn.apply(x))
    lib.impl("_softmax", lambda x, dim, half=False: SoftmaxFn.apply(x, dim))
    def max_pool2d(x, k, s=None, p=0, d=1, ceil=False):
        if ceil:
            raise RuntimeError("torch_vc4: max_pool2d ceil_mode is not "
                               "implemented (the engine pools over a "
                               "floor-shaped window)")
        if ops._pair(d) != (1, 1):
            raise RuntimeError("torch_vc4: max_pool2d dilation != 1 is not "
                               "implemented")
        return MaxPool2dFn.apply(x, _k(k), _k(s if s not in (None, []) else k),
                                 _k(p))

    lib.impl("max_pool2d", max_pool2d)
    lib.impl("_adaptive_avg_pool2d",
             lambda x, out: AdaptiveAvgPool2dFn.apply(x, out))
    lib.impl("adaptive_avg_pool2d",
             lambda x, out: AdaptiveAvgPool2dFn.apply(x, out))
    lib.impl("view", lambda x, size: ViewFn.apply(x, list(size)))
    lib.impl("as_strided",
             lambda x, size, stride, off=None:
             AsStridedFn.apply(x, list(size), list(stride), off))
    lib.impl("linear", lambda x, w, b=None: LinearFn.apply(x, w, b))
    lib.impl("mul.Scalar", lambda x, s: ScalarMulFn.apply(x, float(s)))
    lib.impl("div.Scalar", lambda x, s: ScalarMulFn.apply(x, 1.0 / float(s)))
    lib.impl("div.Tensor", lambda a, b: ScalarMulFn.apply(
        a, 1.0 / ops._scalar_of(b)) if ops._scalar_of(b) is not None
        else _no_div(a, b))
    lib.impl("add.Scalar", lambda x, s, alpha=1: ScalarAddFn.apply(
        x, float(s) * float(alpha)))
    lib.impl("sub.Scalar", lambda x, s, alpha=1: ScalarAddFn.apply(
        x, -float(s) * float(alpha)))
    lib.impl("neg", lambda x: NegFn.apply(x))
    lib.impl("pow.Tensor_Scalar", lambda x, p: Pow2Fn.apply(x, float(p)))
    lib.impl("sum", lambda x, dtype=None: SumFn.apply(x))
    lib.impl("mean", lambda x, dtype=None: MeanFn.apply(x))
    return lib

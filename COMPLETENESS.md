# torch_vc4 0.3.0 — completeness, accuracy and measured cost

This is the authoritative op table for the backend.  Every aten op that
`torch_vc4` intercepts is listed with

* its **backend**: `device` (all arithmetic is a QPU kernel), `composite`
  (a documented composition of QPU kernels plus byte-level layout work on the
  ARM), or `raise` (unsupported, with the exact reason), and
* its **measured accuracy** against PyTorch's CPU fp32 implementation on at
  least three shapes (max abs and max rel with the engine's `|ref| >= 1`
  denominator), and
* its **measured wall time at three sizes**: vc4 end-to-end (dispatch + vc4d
  round trip) vs the same op on the CPU.

## The honest headline

**The eager path is launch-bound, and it is slower than the CPU for small
tensors.**  Every device op is one or more *synchronous* `LAUNCH_WAIT` round
trips to `vc4d` (about 0.25-0.5 ms each at stock clocks), so a 64-float
`relu` costs 0.24 ms on vc4 vs 0.018 ms on the CPU (13x), a 64-float `add`
costs 0.85 ms vs 0.018 ms (47x), and `mm 32x32x32` costs 0.93 ms vs 0.041 ms
(23x).  The QPU only wins once there is enough arithmetic to amortise the
launch: the same `mm` at 256x256x256 measured **5.09 ms on vc4 vs 6.17 ms on
the CPU** (the earlier "1.7x slower" figure was the cold-cache/first-launch
path, which includes JIT).  This backend is useful because the ops are real
GPU kernels with a verified numerical contract, not because it beats a
1.2 GHz Cortex-A53 on small tensors.

Timings: Pi 3B, stock clocks (ARM 1200 MHz, V3D 300 MHz), 905 MB, torch
2.15.0+cpu, best of 3 after a warm-up, `vc4d` backend.  **The board was
shared with a concurrent job's long QPU training run while this table was
measured**; launches serialise on the daemon's GPU lock, so the vc4 numbers
are upper bounds (the `softmax >16`, `batch_norm` and `index_select` rows are
the most visibly inflated).  Accuracy: the same board, fp32 CPU reference
(engine accumulation order differs, hence non-zero errors).

## Op table

Sizes are the three listed per row.  Accuracy is the worst over the three
shapes in `tests/test_complete.py` / `tests/completeness.py`.

| aten op | backend | max abs / max rel | vc4 ms (3 sizes) | CPU ms (3 sizes) |
|---|---|---|---|---|
| `aten::mm` (32³ / 128³ / 256³) | device | 4.31e-04 / 2.00e-04 | 0.927 / 1.880 / 5.089 | 0.041 / 2.802 / 6.173 |
| `aten::linear` (1x256x64 / 32x256x64 / 128x512x128) | device | 1.16e-03 / 5.42e-04 | 3.215 / 2.517 / 6.415 | 1.393 / 4.905 / 10.201 |
| `conv2d` dense stride 1 | device | 5.57e-04 / 2.10e-04 | 2.119 / 15.743 / 9.420 | 1.935 / 15.080 / 3.472 |
| `conv2d` dense stride 2 | device (exact phase decomposition) | 5.49e-04 / 1.80e-04 | 4.049 / 17.184 / 19.285 | 0.087 / 1.580 / 1.937 |
| `conv2d` depthwise 3x3 | device (dense block-diagonal) | 4.77e-06 / 2.01e-06 | 2.503 / 4.798 / 5.686 | 0.920 / 1.189 / 1.380 |
| `max_pool2d 2x2` | device | 0 / 0 (exact) | 0.933 / 3.605 / 5.106 | 0.053 / 0.306 / 0.665 |
| `adaptive_avg_pool2d 1x1` | composite (GEMM) | 4.32e-06 / 4.32e-06 | 1.397 / 2.602 / 4.135 | 0.075 / 0.093 / 0.139 |
| `upsample_nearest2d 2x` | composite | 0 / 0 (exact) | 0.565 / 5.298 / 14.624 | 0.099 / 1.415 / 4.655 |
| `aten::cat` (channel) | composite (host byte copy) | 0 / 0 (exact) | 0.687 / 5.276 / 11.413 | 0.048 / 0.237 / 1.237 |
| `softmax` <= 16 classes | device | 5.66e-07 / 5.66e-07 | 0.654 / 0.783 / 2.246 | 0.031 / 0.035 / 0.100 |
| `softmax` > 16 classes | composite (sigmoid-exp + max/sum) | 8.49e-07 / 8.49e-07 | 10.408 / 45.567 / 491.933 | 0.030 / 2.179 / 1.189 |
| `aten::relu` (64 / 4096 / 65536) | device | 0 / 0 (exact) | 0.241 / 0.394 / 3.473 | 0.018 / 0.027 / 0.169 |
| `aten::add.Tensor` same shape | device | 4.77e-07 / 1.19e-07 | 0.848 / 1.300 / 4.354 | 0.018 / 0.031 / 1.816 |
| `aten::add.Tensor` broadcast | composite (expand copy + device) | 4.77e-07 / 1.19e-07 | 1.692 / 4.630 / 48.848 | 0.022 / 0.031 / 1.286 |
| `aten::abs` | device | 0 / 0 (exact) | 0.849 / 0.524 / 3.579 | 0.025 / 0.031 / 0.186 |
| `aten::tanh` | composite (2*sigmoid(2x)-1) | 1.61e-06 / 1.61e-06 | 0.986 / 1.942 / 17.716 | 0.020 / 0.127 / 0.834 |
| `aten::sqrt` | composite (x*rsqrt(x)) | 1.19e-07 / 1.19e-07 | 0.561 / 0.849 / 7.078 | 0.017 / 0.033 / 0.153 |
| `aten::sum(dim)` | composite (GEMM) | 3.24e-04 / 1.32e-04 | 1.019 / 1.067 / 2.324 | 0.034 / 0.044 / 0.077 |
| `aten::amax(dim)` | composite (relu max tree) | 7.15e-07 / 3.05e-07 | 5.247 / 10.603 / 21.336 | 0.032 / 0.036 / 0.046 |
| `aten::argmax(dim)` | composite (host index scan) | 0 / 0 (exact) | 0.162 / 0.382 / 1.040 | 0.031 / 0.049 / 0.103 |
| `aten::index_select` | composite (host word gather) | 0 / 0 (exact) | 0.821 / 6.803 / 48.426 | 0.110 / 0.129 / 2.540 |
| `batch_norm` eval | composite (device elementwise) | 4.77e-07 / 1.18e-07 | 16.285 / 270.144 / 521.756 | 2.060 / 2.104 / 0.269 |
| `batch_norm` train | composite (device reductions) | y 5e-5, running stats 7e-7 | 30.986 / 381.285 / 738.938 | 6.158 / 9.162 / 7.394 |

## The rest of the intercepted surface (verified; same backend pattern)

These share the rows above (one launch for a native unary body, a handful for
a composite, or a host word copy for indexing); they are listed separately so
the table stays readable.

| aten op | backend | measured accuracy (worst of 3 shapes) |
|---|---|---|
| `aten::neg`, `aten::sign`, `aten::abs` | device (bit ops) | exact (sign: +0 for -0, as ONNX) |
| `aten::clip` (min/max, any finite bounds) | composite (relu + constants) | 1.2e-07 abs / 1.1e-07 rel |
| `aten::leaky_relu` | composite (relu + constants) | 0 / 0 (exact in fp32) |
| `aten::hardsigmoid` | device | 5.96e-08 / 5.96e-08 |
| `aten::hardswish` | device | 2.38e-07 / 1.80e-07 |
| `aten::reciprocal` | composite (rsqrt) | 1.91e-06 / 1.70e-07 |
| `aten::gelu(approximate='tanh')` | composite (sigmoid) | 4.77e-07 / 3.13e-07 |
| `aten::sum` / `mean` full and dim, keepdim | composite (GEMM) | 1.6e-05 abs / 1.9e-06 rel |
| `aten::amin`, `aten::min`, `aten::max` | composite (relu tree) | 7.15e-07 / 3.05e-07 |
| `aten::argmin`, `aten::argmax` (full/dim) | composite (host scan) | exact |
| `aten::max.dim` / `min.dim` | composite (values + host indices) | values 2e-5, indices exact |
| `aten::gather`, `aten::index_select` | composite (host word gather) | exact |
| `aten::sigmoid`, `aten::silu`, `aten::relu` | device | 2e-6 / 2e-6 (forward), exact (relu) |
| `aten::convolution` any batch, stride 1/2, depthwise | device | 5.6e-04 / 2.1e-04 |
| `aten::max_pool2d` generic NxN/stride/pad, any batch | device | exact |
| `aten::_adaptive_avg_pool2d`, `upsample_nearest2d`, any batch | composite | 4.3e-06 / 4.3e-06; resize exact |
| `aten::native_batch_norm` / `batch_norm` train+eval | composite | y 5e-5, stats 7e-7 |
| `aten::view` / `flatten` / `reshape` / `as_strided` | metadata only | exact |
| autograd | see README | gradients <= 2e-3 relative vs CPU (measured) |

## Unsupported ops (raise, never a CPU fallback)

| op | reason |
|---|---|
| `aten::exp` | The engine's `vc4_unary` exp body is **wrong on this board**: `genops_probe.py exp` measured max abs 2.5e5 / max rel 85 over 4096 inputs (4096/4096 outside tolerance).  Softmax does not need it (see below). |
| `aten::log` | The engine's log body returns NaN for 4081/4096 inputs (measured). |
| `aten::erf` | The engine's erf body does not generate / is numerically wrong in this kops build (`VC4_ERR_KERNEL`); `gelu(approximate='tanh')` is the available form. |
| `aten::floor`, `aten::ceil` | The kernel does not generate in this kgen build (`kops.py b_ceil/b_floor` raises a Python `TypeError`, then `VC4_ERR_KERNEL`). |
| `aten::round` | The engine body is wrong: max abs 12 / max rel 8.01, 3997/4096 inputs outside tolerance (measured). |
| `aten::scatter*` | No device scatter kernel.  The table-driven `vc4_gather` only reads, and it hangs on this board (below); a host scatter would be a CPU fallback. |
| `aten::sort`, `aten::topk`, `aten::kthvalue`, `aten::median`, `aten::unique*` | No QPU sort/selection kernel; a host sort would be a full CPU fallback. |
| `amax`/`amin` **backward**, `gather`/`index_select` **backward** | Their gradients need an additive scatter (equality mask for ties); no such kernel exists.  Forward is supported. |
| float64 (`.double()`, `dtype=torch.float64`) | Unsupported **by design**: the VC4 QPU and every engine kernel are float32-only.  Emulating fp64 would double storage and add conversion launches while still computing in fp32 — rejected instead of silently downgraded. |
| conv `dilation != 1` | The engine's NHWC path has an explicit-dilation entry point, but it is not wired into this backend; the NCHW path is stride 1/2, dilation 1. |
| transposed conv, `output_padding`, `ceil_mode` pooling, non-square stride/padding, kernels other than 1x1/3x3, `groups` other than 1 or depthwise | Not implemented by the kernels this backend wires; the message names the constraint. |
| `torch.device("vc4")` string | torch 2.15's device-string parser was initialised before the PrivateUse1 rename; use `torch_vc4.device()` (documented in the README). |
| host RNG (`normal_`, `uniform_`) | The GPU has no RNG kernel; parameters/data are generated on the host and uploaded.  Stated, not hidden: data generation, never compute. |

## Engine findings that shaped the implementation (measured)

* **`vc4_reduce` hangs the GPU in this engine build.**  A single call on a
  1x37 tensor returned `VC4_ERR_TIMEOUT` (-4) and killed the context.  Reductions
  therefore use a GEMM (sum/mean) or a relu max tree (amax/amin), all on the QPU.
* **The engine's batched NHWC conv path (`vc4_conv_nhwc` /
  `vc4_conv_nhwc_pack_w`) hangs this board** in the 1.2.1+training build: its
  weight-pack launch returned `VC4_ERR_TIMEOUT` and killed the context.  Item 6
  therefore keeps the **exact decomposition** (stride-2 phase images, dense
  block-diagonal depthwise) that has been verified since 0.1.0; it is not a
  fallback, it is the supported path, with no host loop over channels.
* **The transcendental unary bodies are wrong** even though the verified subset
  (`abs`, `neg`, `sign`, `clip` params, `leakyrelu`, `hardsigmoid`,
  `hardswish`) is exact.  The backend routes only the verified set and
  composes `tanh`/`sqrt`/`reciprocal` from `vc4_sigmoid`/`vc4_rsqrt`, which are
  verified (batch norm has used them since 0.1.0).
* **vc4d's launch policy rejects uniform words that alias the GPU bus
  window.**  A float with |v| >= 2 and a negative sign (including -inf) has
  that bit pattern, so `clip` is composed from `relu` + host-materialised
  constants instead of passing the bounds as unary uniforms.

## Install

```sh
pip install --extra-index-url https://rughinnit.github.io/vc4-apt/pip/simple/ torch_vc4
```

The wheel is a cp313/aarch64 binary (the backend is a compiled extension
statically linked against `libvc4infer.a`), published next to `vc4infer` in
the signed `vc4-apt` repository:

* index: <https://rughinnit.github.io/vc4-apt/pip/simple/>
* wheel: `pip/torch_vc4-0.3.0-cp313-cp313-linux_aarch64.whl`
* sha256: `7397b46ff1e8b3709e48ca4d59e2a23d63b12ed762062111b7439a9a462ec561`

## Reproduce

```sh
.venv/bin/python tests/test_complete.py   # this table's accuracy checks
.venv/bin/python tests/completeness.py    # this table's timings
sh tests/run_all.sh                       # all four suites
# against an installed wheel (a clean venv):
PYTHON=/path/to/cleanvenv/bin/python sh tests/run_all.sh
```

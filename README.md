# torch-vc4 — a minimal, honest PyTorch backend for VideoCore IV

Status: **subset backend, working**. `torch_vc4.device()` is a real PyTorch
device (`PrivateUse1`, renamed `vc4`) whose matmul runs on the VC4 QPU and is
verified against the CPU. Everything outside the op table below **raises**;
there is no CPU fallback and no pretence of one.

Tensors live in the compute engine's dma-buf arena (`libvc4infer`), and every
GPU launch goes through **vc4d**, so the Python process is unprivileged: one
AF_UNIX socket, one 64 MiB dma-buf mapping, no `/dev/mem`, `CapEff=0`.

Measured numbers and the blockers that shaped this design: see
[RESULTS.md](RESULTS.md).

## Install (on the Pi)

```sh
cd /home/raspi/work/torch-vc4
python3 -m venv .venv
.venv/bin/pip install --no-cache-dir --no-deps \
    .tmp/torch-2.15.0+cpu-cp313-cp313-manylinux_2_28_aarch64.whl
.venv/bin/pip install --no-cache-dir typing_extensions filelock sympy \
    networkx jinja2 fsspec setuptools
MAX_JOBS=1 .venv/bin/pip install --no-cache-dir --no-build-isolation -e .
```

Prerequisites:
- `vc4d.service` active and the user in group `vc4` (`systemctl status vc4d`);
- `libvc4infer.so` built in `/home/raspi/work/engine-daemon/build`
  (override the path with `VC4_ENGINE_DIR=...`);
- a 2 GB swapfile if you rebuild: the compile fits in ~1 GB of address space
  under `ulimit -v 1200000`, but not comfortably in 1 GB of RAM (see RESULTS).

The wheel must be the **`+cpu`** one. The PyPI aarch64 wheel is the CUDA build
and `import torch` dies with `libcudart.so.13: cannot open shared object file`
unless ~3 GB of `nvidia-*` wheels are installed as well.

## Run

```sh
cd /home/raspi/work/torch-vc4
.venv/bin/python tests/test_backend.py     # acceptance test, exit 0 = pass
.venv/bin/python tests/bench.py 5          # timing
```

```python
import torch, torch_vc4

d = torch_vc4.device()                      # torch.device("vc4", 0) in effect
x = torch.randn(256, 256, device=d)         # host RNG (std::mt19937) + upload
y = x @ x                                   # QPU GEMM (pack + asm10 gemm)
print((y.cpu() - x.cpu() @ x.cpu()).abs().max())
```

### Device string

`torch.utils.rename_privateuse1_backend("vc4")` is applied at import, and
tensors report `device(type='vc4')`. However torch's device-string parser is
populated during `import torch`, before the rename, so `torch.device("vc4")`
raises `RuntimeError: Invalid device string: 'vc4'` (measured; the built-in
spelling `"privateuseone"` still parses, consistent with a static name table).
Use `torch_vc4.device()`, or `torch.device("privateuseone")`; both give a
device whose `.type` is `"vc4"`.

## Op table (this is the whole surface)

| aten op | how implemented | constraints |
|---|---|---|
| `empty.memory_format`, `empty_strided` | `torch_vc4` arena sub-allocator | f32, strided, contiguous strides only |
| `copy_` (and `.to`/`.cpu`/`.clone`) | `vc4_upload` / `vc4_download` | same shape, same dtype (f32), contiguous |
| `mm` | `vc4_matmul` → QPU packer + asm10 QPU GEMM | f32, 2-D, contiguous, any M/N/K |
| `matmul` | 2-D only, same path | f32, 2-D × 2-D |
| `add.Tensor` | `vc4_elementwise(ADD)` | same shape (no broadcast), alpha == 1 |
| `mul.Tensor` | `vc4_elementwise(MUL)` | same shape (no broadcast) |
| `relu` | `vc4_elementwise(RELU)` | f32, contiguous |
| `normal_` | host `std::mt19937`, uploaded | f32; **the RNG is on the host** |
| everything else | catch-all `PrivateUse1` fallback | raises `RuntimeError: torch_vc4: 'aten::sin' is not implemented for device vc4 …` |

Deliberately absent, therefore raising: `sin`, `mean`, `sum`, `softmax`,
`conv2d`, `bmm`, broadcasting, `as_strided` (so slicing/permute/reshape of a
vc4 tensor fails), device-side dtype conversions, autograd, `torch.compile`.
The *engine* has conv2d/sigmoid/silu/maxpool/softmax kernels; exposing them
through aten is mechanical but out of scope here.

One device only (index 0). There is no stream API: every launch is a
synchronous `LAUNCH_WAIT` round trip (`torch_vc4.synchronize()` is a no-op).

## Honesty notes

- `torch.randn(..., device=…)` draws from `std::mt19937` on the host and
  uploads the result; the QPU has no RNG. `torch.manual_seed(seed)` reaches
  it (`torch_vc4.manual_seed_all`), and `TORCH_VC4_SEED=…` seeds the first
  draw. This is data generation, not an aten compute fallback.
- Shape padding for matmul (rows→8, cols→64, k→16) is done by the engine with
  byte copies and zero fills on the ARM; no multiply-accumulate runs on the CPU.
- Tensors cannot be `print()`ed or `repr()`ed: torch's formatter needs ops
  that are not implemented.
- `normal_` materialises a host `std::vector<float>` of the tensor size, so a
  very large `randn` needs the RAM for it.
- The backend reserves 24 MiB of the 64 MiB client arena for torch tensors;
  the rest is the engine's scratch (matmul panels, padding, elementwise tails).
  Running out raises `torch_vc4: out of arena …`.
- `torch_vc4` is a plain shared object loaded with `ctypes`, not a pybind11
  extension module: pybind11 + `torch/extension.h` do not fit this board's
  compile memory budget (measured, see RESULTS).

## GPU serialisation

`vc4d` takes `flock(LOCK_EX|LOCK_NB)` on `/tmp/vc4-gpu.lock` around every
launch (`vc4d/src/vc4d.c:133`, `VC4D_GPU_LOCK=1` in the unit) and bounds the
wait, so **all** GPU work from this backend is serialised by that one lock.
The client must *not* hold the lock itself (the daemon would report `E_BUSY`
and the engine would retry for ~60 s) — the same rule as `docs/DAEMON.md`.
Measured: with another process holding the lock, an engine matmul on the same
boot takes 5.70 s instead of ~0.25 ms, i.e. the daemon serialises.

## Files

```
csrc/torch_vc4.cpp      the backend: allocator, guard, hooks, 9 aten kernels,
                        catch-all fallback, diagnostics ops
torch_vc4/__init__.py   rename + device module + helpers (device(), stats(), …)
setup.py                one-TU CppExtension against libvc4infer.so
tests/test_backend.py   acceptance test (CPU verification, strict failures)
tests/bench.py          timing
RESULTS.md              what was measured, including the failed paths
```

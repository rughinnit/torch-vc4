# torch-vc4 — what was measured

One boot of the Raspberry Pi 3B, stock clocks (V3D 300 MHz, ARM 1200 MHz,
`get_throttled=0x0` after the runs), user `raspi`, GPU work through `vc4d`
(the daemon holds `/tmp/vc4-gpu.lock` per launch; the client never does).

Date of the run: 2026-11-19, Debian 13 (trixie) arm64, Python 3.13.5,
glibc 2.41, torch 2.15.0+cpu.

## Step 0 — is there a wheel?

| item | result |
|---|---|
| PyPI aarch64 wheel for cp313 | **yes**: `torch-2.15.0-cp313-cp313-manylinux_2_28_aarch64.whl`, 483 665 057 B, sha256 matches PyPI |
| does it import? | **no.** It is the CUDA build: `OSError: libcudart.so.13: cannot open shared object file`, then `ValueError: libcublasLt.so.*[0-9] not found`. Its dependency closure is `nvidia-cudnn-cu13` (595 MB) plus cuBLAS/cuSPARSELt/NCCL/cuDNN etc. — ~3 GB of downloads at the Pi's ~0.45 MB/s |
| CPU-only wheel | `torch-2.15.0+cpu-cp313-cp313-manylinux_2_28_aarch64.whl`, 170 747 118 B, sha256 `ab35875b…f0a5e78` (verified), deps are pure Python only |
| installed | yes: `torch.__version__ == "2.15.0+cpu"` |
| download path | PyPI direct to the Pi was ~0.45 MB/s (first attempt died at 30 MB in `/tmp` — `/tmp` is a 453 MB tmpfs). The wheel was fetched at 94 MB/s on the VPS and copied; the sha256 was re-verified on the Pi |
| venv size | **816 MB** (`.venv`), of which `site-packages/torch` is 688 MB. Wheel kept at `.tmp/` (163 MiB), `pip --no-cache-dir` used |
| ABI | manylinux_2_28 on glibc 2.41 — fine. Nothing built from source except this extension |

Import cost: `import torch` 11.9 s with a warm page cache, 42 s cold from the
SD card; `import torch_vc4` +0.15–0.55 s; peak RSS 135–220 MB.

## Step 1 — the extension

- `PrivateUse1` renamed to `vc4`; allocator, device guard, `PrivateUse1Hooks`
  interface (torch refuses to create a privateuse1 tensor without the hooks),
  nine aten kernels, catch-all `PrivateUse1` fallback that throws.
- All tensors come from one 24 MiB `vc4_alloc()` region inside the engine's
  64 MiB client arena; 64-byte alignment + 64-byte slack per block, free list
  with coalescing. The engine's scratch is the rest of the arena.
- The native part is loaded with `ctypes`, not pybind11 (see "build" below).

### Build on a 1 GB board (and the two reboots)

| attempt | result |
|---|---|
| `torch/extension.h` + pybind11, `-O2`, no swap beyond the 926 MB zram | the **board reset** (hard reboot) twice, once in `cc1plus`; no OOM line in the previous boot's journal, last messages were normal |
| same, after adding a 2 GB swapfile | compiled for 8+ min, 849 MB RAM + 1.35 GB swap in use, then the board reset again |
| final: ATen + `torch/library.h` only (no C++ frontend, no pybind11), `-O0 -g0 -std=c++20`, one TU, `MAX_JOBS=1`, `ulimit -v 1200000` | **builds**, ~4 min, no reset |

So: `torch 2.15 requires -std=c++20`, and the compile needs roughly 1 GB of
address space. A 2 GB swapfile was added and persisted (`/swapfile` in
`/etc/fstab`; total swap 2.9 GB with the existing zram). It is only needed for
rebuilds, not for running.

## Step 2 — acceptance run

`tests/test_backend.py`, all checks pass:

```
PASS  backend renamed to vc4  vc4
PASS  torch_vc4.device()  vc4:0
      torch.device('vc4') string: rejected by torch's parser
      (Invalid device string: 'vc4'); torch_vc4.device() is the supported spelling
      arena: {'region_off': 1048576, 'region_bytes': 25165824,
              'arena_bytes': 67108864, 'arena_used': 26214400}
PASS  engine arena up (vc4d)
PASS  randn on vc4 (host RNG + upload)  torch.Size([256, 256])
PASS  vc4->cpu copy is bit exact
PASS  x @ x returns a vc4 tensor
      256^3 wall 5.384 ms, engine gpu 3.974 ms (9.46 GFLOP/s kernel), 2 launches
PASS  matmul matches CPU (allclose rtol 1e-4, atol 1e-3)
      max rel 1.78e-04, max abs 5.04e-04 (fp64 ref: 4.54e-04)
PASS  matmul max relative error < 1e-3 (engine-style denominator)  1.78e-04
      256^3 best wall 4.201 ms -> 7.99 GFLOP/s wall; engine GPU 3.980 ms
PASS  add on vc4 == CPU
PASS  mul on vc4 == CPU
PASS  relu on vc4 == CPU
PASS  odd-shaped mm 20x36x44 == CPU  max rel 7.05e-06, max abs 1.91e-05
PASS  torch.sin(vc4) raises    ('aten::sin' is not implemented for device vc4)
PASS  torch.mean(vc4) raises   ('aten::mean' is not implemented for device vc4)
PASS  broadcasting add raises  (add requires identical shapes, [256,256] and [256,1])
PASS  slicing a vc4 tensor raises (as_strided missing)  (as_strided not implemented)
PASS  float64 on vc4 raises    (only float32 tensors are supported, got Double)
PASS  mm cpu x vc4 raises      (mm(A): expected a vc4 tensor, got cpu)
PASS  client process has no /dev/mem fd
PASS  client holds a vc4d socket
      CapEff: 0000000000000000  uid=1000
```

### Numerics, stated precisely

256³, `x = x @ x` with `x ~ N(0,1)`:

- vs CPU **fp64**: max |err| 4.54e-04 on entries with |y| up to ~60 (typical
  |y| ≈ 16). This is the expected fp32 accumulation error for 256 terms
  (`n·eps·Σ|terms|` ≈ 5e-4).
- vs CPU **fp32**: max |err| 5.04e-04, `torch.allclose(rtol=1e-4, atol=1e-3)`
  passes.
- engine-style relative metric `|err| / max(|ref|, 1)`: 1.78e-04. The 1.0
  denominator floor makes near-zero output entries dominate, so it is slightly
  above the engine's own `TOL 1e-4`; the absolute error is what the fp32 maths
  predicts. Nothing is rounded in the report: both numbers are above.
- 20×36×44 (non-multiple-of-64 N, exercises the engine's padded scratch):
  max rel 7.05e-06, max abs 1.91e-05.
- `add`, `mul`, `relu` match the CPU to 1e-6 (`allclose`).

### Timing (best of 5, same boot, stock clocks)

| shape | wall ms | wall GFLOP/s | engine GPU ms | GPU GFLOP/s |
|---|---:|---:|---:|---:|
| 64³ | 0.390 | 1.34 | 0.211 | 2.48 |
| 128³ | 0.962 | 4.36 | 0.772 | 5.44 |
| 256³ | 4.182 | 8.02 | 3.960 | 8.47 |

A second run of the same binary on the same boot, ~25 minutes later with the
board at load average 2.4 (other users; core still 400 MHz, V3D 300 MHz,
`get_throttled=0x0`), gave 64³ 0.424 ms, 128³ 1.254 ms, 256³ 4.990 ms / 7.61
GFLOP/s GPU — i.e. the spread between runs on this shared board is ~15–25 %,
which is why both are quoted. See `docs/CLOCK-VARIANCE.md` for why wall time
moves with unrelated load while the kernel cycle counts do not.

"engine GPU" is `vc4_get_stats()`: pack + kernel time as reported by the
engine; "wall" is `time.perf_counter()` around the Python call (two launch
round trips plus the copy of the inputs, no host copies for the result).

Elementwise is launch-bound: 256 floats per QPU group, 8 groups per launch, so
65 536 floats = 32 `LAUNCH_WAIT` round trips ≈ 3.2 ms (relu 3.22 ms, add
3.20 ms, mul 4.53 ms). The 64×128 elementwise checks in the acceptance test
are unaffected by this.

### Unprivileged operation

`/proc/self/fd` during the run: 1 socket to `vc4d`, 2 pipes, nothing else;
`CapEff=0000000000000000`, uid 1000.

`strace -f -e trace=openat,connect,socket,mmap`:

```
/dev/mem opens:                 0
unix socket connects:           1  sun_path="/run/vc4d.sock"
mmap(NULL, 67108864, ...):      1  (the dma-buf arena)
```

### GPU lock

`flock` is taken by `vc4d` per launch. Evidence: with another process holding
`/tmp/vc4-gpu.lock` for 6 s, the engine's `bench_matmul 64` took **5.70 s**
wall for 0.25 ms of GPU time (`vc4d/src/vc4d.c:133`, `VC4D_GPU_LOCK=1`).
The torch client process itself never calls `flock`; holding it there would
make the daemon answer `E_BUSY` and the engine retry for ~60 s.

## What is not there (and is reported as such)

- `torch.device("vc4")`: torch's device-string parser is populated during
  `import torch`, before the rename (it still accepts the built-in spelling
  `"privateuseone"` and rejects `"vc4"` afterwards; consistent with a static
  name table, though the C++ source is not shipped in the wheel).
  `torch_vc4.device()` is the supported entry point.
- No autograd, no `torch.compile`, no conv/pool/softmax aten ops, no
  broadcasting, no `as_strided` (hence no slicing/reshape/view), no device-side
  dtype conversion, no multi-device, no streams.
- Printing a vc4 tensor fails (the formatter needs unimplemented ops).
- `torch.randn` is host-generated.

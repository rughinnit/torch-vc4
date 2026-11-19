"""torch_vc4 - a minimal, honest PyTorch backend for the Raspberry Pi VC4 GPU.

Usage:

    import torch, torch_vc4
    x = torch.randn(256, 256, device=torch_vc4.device())
    y = x @ x

Only a handful of ops exist on the device; anything else raises instead of
falling back to the CPU (see README.md for the op table).

Note on the device string: this backend renames PrivateUse1 to "vc4", and
tensors report `device('vc4')`, but torch 2.15's device-string *parser* was
initialised before the rename and still only accepts the built-in spelling
"privateuseone" (measured; see RESULTS.md).  Use `torch_vc4.device()`, or
`torch.device("privateuseone")`, in the few places a string/device object is
needed.
"""

import ctypes as _ctypes
import glob as _glob
import os as _os
import sys as _sys
import types as _types

import torch as _torch

# The native part is a plain shared object (no pybind11: it would not fit in
# this board's compile budget).  Loading it runs the static registrations:
# the PrivateUse1 allocator, the device guard, the aten kernels and the
# torch_vc4.* diagnostic ops.  It also forces VC4INFER_DAEMON=1 (vc4d).
_here = _os.path.dirname(_os.path.abspath(__file__))
_native = sorted(_glob.glob(_os.path.join(_here, "_C*.so")))
if not _native:
    raise ImportError(
        "torch_vc4: the native backend is not built. Run:\n"
        "  MAX_JOBS=1 .venv/bin/pip install --no-build-isolation -e .")
_ctypes.CDLL(_native[0], mode=_ctypes.RTLD_GLOBAL)

if _torch._C._get_privateuse1_backend_name() != "vc4":
    _torch.utils.rename_privateuse1_backend("vc4")

# torch looks up torch.<backend> for RNG/seed bookkeeping; without it
# torch.device("privateuseone") raises ModuleNotFoundError: No module named
# 'torch.vc4'.
_vc4_mod = _types.ModuleType("torch.vc4")


def _manual_seed_all(seed: int) -> None:
    _torch.ops.torch_vc4.seed(int(seed))


_rng_state = _torch.zeros(4, dtype=_torch.uint8)


def _get_rng_state(device=None):
    return _rng_state.clone()


def _set_rng_state(new_state, device=None):
    global _rng_state
    _rng_state = new_state.clone()


_vc4_mod.manual_seed_all = _manual_seed_all
_vc4_mod.is_available = lambda: True
_vc4_mod.device_count = lambda: 1
_vc4_mod.current_device = lambda: 0
_vc4_mod._is_in_bad_fork = lambda: False
_vc4_mod.get_rng_state = _get_rng_state
_vc4_mod.set_rng_state = _set_rng_state
_sys.modules["torch.vc4"] = _vc4_mod
setattr(_torch, "vc4", _vc4_mod)
try:
    # torch insists on parsing the name; "privateuseone" is the only spelling
    # its device-string parser accepts in this build.
    _torch._register_device_module("privateuseone", _vc4_mod)
except Exception:  # pragma: no cover - informational only
    pass

try:  # adds Tensor.vc4() / Module.vc4() convenience methods
    _torch.utils.generate_methods_for_privateuse1_backend()
except Exception as _e:  # pragma: no cover - informational only
    import warnings as _w

    _w.warn(f"torch_vc4: generate_methods_for_privateuse1_backend failed: {_e}")


def device(index: int = 0) -> "_torch.device":
    """The vc4 device (always index 0).

    Uses the built-in "privateuseone" spelling because torch 2.15's
    device-string parser does not accept the renamed name; the resulting
    object still reports `type == "vc4"`.
    """
    return _torch.device("privateuseone", index)


def is_available() -> bool:
    """True if the engine arena came up (vc4d reachable)."""
    try:
        arena()
        return True
    except Exception:
        return False


def stats() -> dict:
    """Engine counters: launches, GPU kernel ms, FLOPs issued on the QPU."""
    keys = ("launches", "pack_ms", "kernel_ms", "gpu_cycles", "flops",
            "gflops_kernel")
    return dict(zip(keys, _torch.ops.torch_vc4.stats()))


def reset_stats() -> None:
    _torch.ops.torch_vc4.reset_stats()


def arena() -> dict:
    """Arena geometry: torch region bytes, engine arena bytes/used."""
    keys = ("region_off", "region_bytes", "arena_bytes", "arena_used")
    return dict(zip(keys, _torch.ops.torch_vc4.arena()))


def synchronize() -> None:
    """No-op: every vc4 launch is a synchronous LAUNCH_WAIT round trip."""
    return None


SUPPORTED_OPS = (
    "aten::empty.memory_format",
    "aten::empty_strided",
    "aten::copy_ (and Tensor.to/.cpu for f32, same shape)",
    "aten::mm",
    "aten::matmul (2-D x 2-D only)",
    "aten::add.Tensor (same shape, alpha=1)",
    "aten::mul.Tensor (same shape)",
    "aten::relu",
    "aten::normal_ (host RNG, uploaded)",
)

__all__ = [
    "device",
    "is_available",
    "stats",
    "reset_stats",
    "arena",
    "synchronize",
    "SUPPORTED_OPS",
]

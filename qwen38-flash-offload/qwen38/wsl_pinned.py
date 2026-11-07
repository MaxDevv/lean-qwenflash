"""Pinned + GPU-mapped host memory whose device address equals its host address, also under WSL2.

The engine writes raw host addresses into tables that GPU kernels dereference (zero-copy over PCIe), so it needs
UVA alias == host address. On bare-metal Linux, anon mmap + cudaHostRegister(Mapped) gives that. Under WSL2 the
driver reports cudaDevAttrCanUseHostPointerForRegisteredMem == 0 and the registered alias differs, but memory from
cudaHostAlloc(Portable|Mapped) is still identity-mapped. needed() decides; alloc() returns a buffer-protocol object
(usable with torch.frombuffer exactly like the mmap it replaces) plus its address.

QWEN_PINNED_HOSTALLOC=1/0 forces the choice.
"""
import ctypes
import os

import torch

_RT = None
_KEEP = []          # allocations live for the whole process, like the mmaps they replace
_NEEDED = {}
_ATTR_CAN_USE_HOST_PTR_FOR_REGISTERED = 91
_HOST_ALLOC_PORTABLE, _HOST_ALLOC_MAPPED = 1, 2


def _rt():
    """The CUDA runtime torch loaded (libcudart.so.12 or .13), so no second runtime is loaded next to it."""
    global _RT
    if _RT is None:
        torch.cuda.init()
        path = None
        try:
            with open("/proc/self/maps") as f:
                path = next((line.split()[-1] for line in f if "/libcudart.so" in line), None)
        except OSError:
            pass
        _RT = ctypes.CDLL(path or f"libcudart.so.{torch.version.cuda.split('.')[0]}")
    return _RT


def needed(dev_idx=0):
    forced = os.environ.get("QWEN_PINNED_HOSTALLOC")
    if forced in ("0", "1"):
        return forced == "1"
    if dev_idx not in _NEEDED:
        v = ctypes.c_int(0)
        r = _rt().cudaDeviceGetAttribute(ctypes.byref(v), _ATTR_CAN_USE_HOST_PTR_FOR_REGISTERED, int(dev_idx))
        _NEEDED[dev_idx] = (r == 0 and v.value == 0)
    return _NEEDED[dev_idx]


def alloc(nbytes, dev_idx, tag):
    """Zero-filled, page-aligned, pinned+mapped host region. Returns (buffer, address)."""
    nbytes = (int(nbytes) + 4095) // 4096 * 4096
    p = ctypes.c_void_p()
    with torch.cuda.device(dev_idx):
        r = _rt().cudaHostAlloc(ctypes.byref(p), ctypes.c_size_t(nbytes), _HOST_ALLOC_PORTABLE | _HOST_ALLOC_MAPPED)
    assert r == 0 and p.value, f"cudaHostAlloc failed ({tag}, {nbytes} B): {r}"
    assert p.value % 4096 == 0
    ctypes.memset(p.value, 0, nbytes)       # anon mmap is zero-filled; keep that guarantee
    _KEEP.append(p.value)
    return (ctypes.c_uint8 * nbytes).from_address(p.value), p.value

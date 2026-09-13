"""Smart device selection shared across the whole pipeline.

Auto-detects the best compute device available on the host:
    CUDA GPU  -> used automatically when present
    Apple MPS -> used automatically when present (no GPU/no CUDA)
    CPU       -> fallback everywhere else

Behaviour:
    * The device is resolved once and cached (process-wide).
    * Set ``ALGRIGHT_DEVICE`` (cpu | cuda | mps) to force a device. A forced
      device that is not actually available degrades to the next-best choice
      instead of crashing.
    * ``torch.cuda.is_available()`` is only True when PyTorch was installed
      with CUDA support AND a working NVIDIA driver/GPU exists, so this works
      on every machine without any extra code.
"""
import os
import threading

import torch

_DEVICE = None
_LOCK = threading.Lock()


def _mps_available():
    return (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _pick(override):
    if override == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if override == "mps" and _mps_available():
        return torch.device("mps")
    if override == "cpu":
        return torch.device("cpu")
    if override in ("cuda", "mps"):
        print(f"[Device] {override} requested but not available -> using CPU")
        return torch.device("cpu")
    return None


def _device_name(device):
    if device.type == "cuda":
        try:
            return torch.cuda.get_device_name(device)
        except Exception:
            return "NVIDIA GPU"
    return "no GPU detected"


def get_device():
    """Return the cached best available torch device (CUDA -> MPS -> CPU)."""
    global _DEVICE
    with _LOCK:
        if _DEVICE is not None:
            return _DEVICE

        override = os.environ.get("ALGRIGHT_DEVICE", "").strip().lower()
        _DEVICE = _pick(override)
        if _DEVICE is None:
            if torch.cuda.is_available():
                _DEVICE = torch.device("cuda")
            elif _mps_available():
                _DEVICE = torch.device("mps")
            else:
                _DEVICE = torch.device("cpu")

        if str(_DEVICE) == "cuda":
            _DEVICE = torch.device(
                f"cuda:{torch.cuda.current_device()}")
        print(f"[Device] Using: {_DEVICE} ({_device_name(_DEVICE)})")
        return _DEVICE


def clear_device_cache():
    """Reset the cached device (mainly for tests)."""
    global _DEVICE
    with _LOCK:
        _DEVICE = None
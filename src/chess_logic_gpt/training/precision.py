from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _torch():
    import torch

    return torch


def cuda_supports_bf16() -> bool:
    torch = _torch()
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _is_auto(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.lower() == "auto")


def _as_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def auto_cuda_dtype() -> torch.dtype:
    torch = _torch()
    return torch.bfloat16 if cuda_supports_bf16() else torch.float16


def dtype_from_config(value: object = "auto") -> torch.dtype:
    torch = _torch()
    if _is_auto(value):
        return auto_cuda_dtype()
    if isinstance(value, torch.dtype):
        return value
    if not isinstance(value, str):
        raise ValueError(f"unsupported dtype value: {value!r}")
    name = value.strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype value: {value!r}")


def training_precision_flags(config: dict) -> tuple[bool, bool]:
    """Return ``(bf16, fp16)`` flags compatible with the current CUDA device."""
    torch = _torch()
    raw_bf16 = config.get("bf16", "auto")
    raw_fp16 = config.get("fp16", "auto")

    if _is_auto(raw_bf16) and _is_auto(raw_fp16):
        if not torch.cuda.is_available():
            return False, False
        return cuda_supports_bf16(), not cuda_supports_bf16()

    bf16 = _as_bool(raw_bf16) if not _is_auto(raw_bf16) else False
    fp16 = _as_bool(raw_fp16) if not _is_auto(raw_fp16) else False

    if bf16 and not cuda_supports_bf16():
        print("bf16 requested but this GPU does not support it; falling back to fp16.", flush=True)
        bf16 = False
        fp16 = torch.cuda.is_available()

    if bf16 and fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled")
    return bf16, fp16

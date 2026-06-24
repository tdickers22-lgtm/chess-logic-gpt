from __future__ import annotations

from chess_logic_gpt.training import precision


class _FakeDType:
    pass


class _FakeCuda:
    def __init__(self, *, available: bool, bf16: bool) -> None:
        self._available = available
        self._bf16 = bf16

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16


class _FakeTorch:
    dtype = _FakeDType

    def __init__(self, *, available: bool, bf16: bool) -> None:
        self.cuda = _FakeCuda(available=available, bf16=bf16)
        self.bfloat16 = _FakeDType()
        self.float16 = _FakeDType()
        self.float32 = _FakeDType()


def test_auto_precision_falls_back_to_fp16_on_t4_class_gpu(monkeypatch):
    fake = _FakeTorch(available=True, bf16=False)
    monkeypatch.setattr(precision, "_torch", lambda: fake)

    assert precision.training_precision_flags({"bf16": "auto", "fp16": "auto"}) == (False, True)
    assert precision.dtype_from_config("auto") is fake.float16


def test_auto_precision_uses_bf16_when_cuda_supports_it(monkeypatch):
    fake = _FakeTorch(available=True, bf16=True)
    monkeypatch.setattr(precision, "_torch", lambda: fake)

    assert precision.training_precision_flags({"bf16": "auto", "fp16": "auto"}) == (True, False)
    assert precision.dtype_from_config("auto") is fake.bfloat16


def test_auto_precision_disables_mixed_precision_without_cuda(monkeypatch):
    fake = _FakeTorch(available=False, bf16=False)
    monkeypatch.setattr(precision, "_torch", lambda: fake)

    assert precision.training_precision_flags({"bf16": "auto", "fp16": "auto"}) == (False, False)

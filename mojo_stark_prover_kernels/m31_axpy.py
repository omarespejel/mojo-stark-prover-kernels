"""M31 batch linear-combination kernel contracts and reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .debug import DebugTracer

M31_PRIME = 2_147_483_647
MAX_DEBUG_LEVEL = 3
MAX_VECTOR_LENGTH = 1_000_000



def _ensure_m31_value(value: int, field_name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")
    if value < 0 or value >= M31_PRIME:
        raise ValueError(f"{field_name} must be in [0, {M31_PRIME - 1}]")


@dataclass(frozen=True)
class M31AxpyRequest:
    """
    Request contract for batch linear combination over M31.

    Computes, for each i:
      out[i] = (alpha * a[i] + beta * b[i] + c[i]) mod M31_PRIME
    """

    a: tuple[int, ...]
    b: tuple[int, ...]
    c: tuple[int, ...]
    alpha: int
    beta: int
    debug_level: int = 0

    @property
    def length(self) -> int:
        return len(self.a)

    def validate(self) -> None:
        if not isinstance(self.debug_level, int):
            raise TypeError("debug_level must be an int")
        if self.debug_level < 0 or self.debug_level > MAX_DEBUG_LEVEL:
            raise ValueError(f"debug_level must be in [0, {MAX_DEBUG_LEVEL}]")

        if len(self.a) == 0:
            raise ValueError("input vectors cannot be empty")
        if len(self.a) > MAX_VECTOR_LENGTH:
            raise ValueError(
                f"input vector length cannot exceed {MAX_VECTOR_LENGTH}; got {len(self.a)}"
            )

        if len(self.a) != len(self.b) or len(self.a) != len(self.c):
            raise ValueError(
                "input vector length mismatch: "
                f"len(a)={len(self.a)}, len(b)={len(self.b)}, len(c)={len(self.c)}"
            )

        _ensure_m31_value(self.alpha, "alpha")
        _ensure_m31_value(self.beta, "beta")

        for idx, value in enumerate(self.a):
            _ensure_m31_value(value, f"a[{idx}]")
        for idx, value in enumerate(self.b):
            _ensure_m31_value(value, f"b[{idx}]")
        for idx, value in enumerate(self.c):
            _ensure_m31_value(value, f"c[{idx}]")

    @classmethod
    def from_sequences(
        cls,
        *,
        a: Sequence[int],
        b: Sequence[int],
        c: Sequence[int],
        alpha: int,
        beta: int,
        debug_level: int = 0,
    ) -> "M31AxpyRequest":
        req = cls(
            a=tuple(int(v) for v in a),
            b=tuple(int(v) for v in b),
            c=tuple(int(v) for v in c),
            alpha=int(alpha),
            beta=int(beta),
            debug_level=debug_level,
        )
        req.validate()
        return req


def m31_axpy_reference(
    request: M31AxpyRequest,
    tracer: DebugTracer | None = None,
) -> tuple[int, ...]:
    request.validate()
    if tracer is not None:
        tracer.emit(1, f"m31 reference start len={request.length}")

    out = tuple(
        (request.alpha * av + request.beta * bv + cv) % M31_PRIME
        for av, bv, cv in zip(request.a, request.b, request.c)
    )

    if tracer is not None:
        tracer.emit(1, f"m31 reference done len={request.length}")
    return out

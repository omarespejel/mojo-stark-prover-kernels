"""Data contracts for kernel execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

Hash32 = bytes

MAX_LOG_SIZE = 20
MAX_COLUMNS = 4096
MAX_DEBUG_LEVEL = 3
MAX_TOTAL_CELLS = 200_000


def _ensure_u32(value: int, field_name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"{field_name} must be in [0, 2^32 - 1]")


def _ensure_hash32(value: bytes, field_name: str) -> None:
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{field_name} entries must be bytes")
    if len(value) != 32:
        raise ValueError(f"{field_name} entries must be exactly 32 bytes")


@dataclass(frozen=True)
class CommitLayerRequest:
    """
    Canonical request contract for layer commitment.

    `columns` are interpreted as field columns with size `2^log_size`.
    """

    log_size: int
    columns: tuple[tuple[int, ...], ...]
    prev_layer_hashes: tuple[Hash32, ...] | None = None
    debug_level: int = 0

    @property
    def n_rows(self) -> int:
        return 1 << self.log_size

    def validate(self) -> None:
        if not isinstance(self.log_size, int):
            raise TypeError("log_size must be an int")
        if self.log_size < 0 or self.log_size > MAX_LOG_SIZE:
            raise ValueError(f"log_size must be in [0, {MAX_LOG_SIZE}]")

        if not isinstance(self.debug_level, int):
            raise TypeError("debug_level must be an int")
        if self.debug_level < 0 or self.debug_level > MAX_DEBUG_LEVEL:
            raise ValueError(f"debug_level must be in [0, {MAX_DEBUG_LEVEL}]")

        if len(self.columns) == 0:
            raise ValueError("columns cannot be empty")
        if len(self.columns) > MAX_COLUMNS:
            raise ValueError(f"columns count cannot exceed {MAX_COLUMNS}")

        expected_rows = self.n_rows
        total_cells = expected_rows * len(self.columns)
        if total_cells > MAX_TOTAL_CELLS:
            raise ValueError(
                f"total input cells cannot exceed {MAX_TOTAL_CELLS}; got {total_cells}"
            )

        for col_idx, column in enumerate(self.columns):
            if len(column) != expected_rows:
                raise ValueError(
                    f"column[{col_idx}] length mismatch: expected {expected_rows}, got {len(column)}"
                )
            for row_idx, value in enumerate(column):
                _ensure_u32(value, f"columns[{col_idx}][{row_idx}]")

        if self.prev_layer_hashes is None:
            return

        expected_prev_hashes = 1 << (self.log_size + 1)
        if len(self.prev_layer_hashes) != expected_prev_hashes:
            raise ValueError(
                "prev_layer_hashes length mismatch: "
                f"expected {expected_prev_hashes}, got {len(self.prev_layer_hashes)}"
            )
        for idx, digest in enumerate(self.prev_layer_hashes):
            _ensure_hash32(digest, f"prev_layer_hashes[{idx}]")

    @classmethod
    def from_sequences(
        cls,
        *,
        log_size: int,
        columns: Sequence[Sequence[int]],
        prev_layer_hashes: Sequence[bytes] | None = None,
        debug_level: int = 0,
    ) -> "CommitLayerRequest":
        req = cls(
            log_size=log_size,
            columns=tuple(tuple(int(v) for v in col) for col in columns),
            prev_layer_hashes=(
                tuple(bytes(h) for h in prev_layer_hashes)
                if prev_layer_hashes is not None
                else None
            ),
            debug_level=debug_level,
        )
        req.validate()
        return req


@dataclass(frozen=True)
class CommitLayerResult:
    layer_hashes: tuple[Hash32, ...]
    backend_name: str
    duration_ms: float
    debug_events: tuple[str, ...]

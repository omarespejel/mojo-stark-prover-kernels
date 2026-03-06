"""Deterministic reference implementation for layer commitment."""

from __future__ import annotations

import hashlib
import struct

from .contracts import CommitLayerRequest, Hash32
from .debug import DebugTracer

PERSONALIZATION = b"MSPKv1"  # <= 8 bytes for blake2s personalization.
DOMAIN_TAG = b"B2S_LAYER_V1"


def commit_layer(request: CommitLayerRequest, tracer: DebugTracer | None = None) -> tuple[Hash32, ...]:
    request.validate()

    if tracer is not None:
        tracer.emit(
            1,
            f"reference commit start rows={request.n_rows} columns={len(request.columns)} "
            f"has_prev={request.prev_layer_hashes is not None}",
        )

    out: list[Hash32] = []
    for row_idx in range(request.n_rows):
        h = hashlib.blake2s(digest_size=32, person=PERSONALIZATION)
        h.update(DOMAIN_TAG)
        h.update(struct.pack("<I", request.log_size))
        h.update(struct.pack("<I", row_idx))

        if request.prev_layer_hashes is not None:
            h.update(request.prev_layer_hashes[2 * row_idx])
            h.update(request.prev_layer_hashes[2 * row_idx + 1])

        for col_idx, column in enumerate(request.columns):
            h.update(struct.pack("<I", col_idx))
            h.update(struct.pack("<I", column[row_idx]))

        digest = h.digest()
        out.append(digest)

        if tracer is not None and tracer.level >= 3:
            tracer.emit(3, f"row={row_idx} digest={digest.hex()}")

    if tracer is not None:
        tracer.emit(1, f"reference commit done hashes={len(out)}")

    return tuple(out)


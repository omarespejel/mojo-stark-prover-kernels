from __future__ import annotations

import json
import pathlib
import unittest

from mojo_stark_prover_kernels.contracts import CommitLayerRequest
from mojo_stark_prover_kernels.debug import DebugTracer
from mojo_stark_prover_kernels.reference_blake2s_merkle import commit_layer


VECTORS_DIR = pathlib.Path(__file__).parent / "vectors"


class ReferenceBlake2sMerkleTests(unittest.TestCase):
    def test_fixed_vector_matches_snapshot(self) -> None:
        vector_path = VECTORS_DIR / "blake2s_merkle_case_01.json"
        payload = json.loads(vector_path.read_text())

        req = CommitLayerRequest.from_sequences(
            log_size=payload["log_size"],
            columns=payload["columns"],
            prev_layer_hashes=(
                [bytes.fromhex(x) for x in payload["prev_layer_hashes"]]
                if payload["prev_layer_hashes"] is not None
                else None
            ),
            debug_level=payload["debug_level"],
        )

        out = commit_layer(req)
        actual = [x.hex() for x in out]
        self.assertEqual(actual, payload["expected_hashes"])

    def test_prev_layer_changes_output(self) -> None:
        cols = [[1, 2, 3, 4], [10, 20, 30, 40]]
        req_no_prev = CommitLayerRequest.from_sequences(log_size=2, columns=cols)
        no_prev = commit_layer(req_no_prev)

        prev = [bytes([idx]) * 32 for idx in range(8)]
        req_prev = CommitLayerRequest.from_sequences(
            log_size=2, columns=cols, prev_layer_hashes=prev
        )
        with_prev = commit_layer(req_prev)
        self.assertNotEqual(no_prev, with_prev)

    def test_debug_tracing_is_emitted(self) -> None:
        req = CommitLayerRequest.from_sequences(
            log_size=1,
            columns=[[1, 2], [3, 4]],
            debug_level=3,
        )
        tracer = DebugTracer(level=3)
        _ = commit_layer(req, tracer=tracer)
        events = tracer.snapshot()
        self.assertGreaterEqual(len(events), 3)


if __name__ == "__main__":
    unittest.main()


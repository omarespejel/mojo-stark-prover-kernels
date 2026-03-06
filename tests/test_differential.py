from __future__ import annotations

import unittest

from mojo_stark_prover_kernels.backends import ReferenceKernelBackend
from mojo_stark_prover_kernels.contracts import CommitLayerRequest
from mojo_stark_prover_kernels.differential import compare_case, run_randomized_suite


class _MismatchBackend:
    name = "mismatch-backend"

    def __init__(self) -> None:
        self._reference = ReferenceKernelBackend()

    def commit_layer(self, request: CommitLayerRequest, tracer=None):
        out = list(self._reference.commit_layer(request, tracer=tracer))
        if out:
            first = bytearray(out[0])
            first[0] ^= 0xFF
            out[0] = bytes(first)
        return tuple(out)


class DifferentialTests(unittest.TestCase):
    def test_compare_case_match(self) -> None:
        req = CommitLayerRequest.from_sequences(log_size=1, columns=[[1, 2], [3, 4]])
        result = compare_case(
            case_id=0,
            request=req,
            candidate_backend=ReferenceKernelBackend(),
        )
        self.assertTrue(result.match)
        self.assertIsNone(result.mismatch_index)

    def test_compare_case_mismatch(self) -> None:
        req = CommitLayerRequest.from_sequences(log_size=1, columns=[[1, 2], [3, 4]])
        result = compare_case(
            case_id=0,
            request=req,
            candidate_backend=_MismatchBackend(),
        )
        self.assertFalse(result.match)
        self.assertEqual(result.mismatch_index, 0)
        self.assertIsNotNone(result.oracle_hex)
        self.assertIsNotNone(result.candidate_hex)

    def test_randomized_suite_passes_for_reference(self) -> None:
        suite = run_randomized_suite(
            candidate_backend=ReferenceKernelBackend(),
            seed=13,
            n_cases=20,
            max_log_size=5,
            max_columns=4,
        )
        self.assertEqual(suite.failed_cases, 0)
        self.assertEqual(suite.passed_cases, suite.total_cases)

    def test_randomized_suite_fail_fast(self) -> None:
        suite = run_randomized_suite(
            candidate_backend=_MismatchBackend(),
            seed=17,
            n_cases=20,
            max_log_size=5,
            max_columns=4,
            fail_fast=True,
        )
        self.assertGreaterEqual(suite.failed_cases, 1)
        self.assertLess(suite.passed_cases, suite.total_cases)
        self.assertEqual(len(suite.failures), 1)


if __name__ == "__main__":
    unittest.main()


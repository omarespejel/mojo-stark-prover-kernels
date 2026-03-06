from __future__ import annotations

import unittest

from mojo_stark_prover_kernels.backends import BackendExecutionError, ReferenceKernelBackend
from mojo_stark_prover_kernels.contracts import CommitLayerRequest
from mojo_stark_prover_kernels.runner import ProverKernelRunner


class _AlwaysFailBackend:
    name = "always-fail"

    def commit_layer(self, request: CommitLayerRequest, tracer=None):
        raise BackendExecutionError("simulated failure")


class RunnerTests(unittest.TestCase):
    def test_primary_success(self) -> None:
        runner = ProverKernelRunner(primary_backend=ReferenceKernelBackend())
        req = CommitLayerRequest.from_sequences(log_size=1, columns=[[1, 2], [3, 4]])
        result = runner.commit_layer(req)
        self.assertEqual(result.backend_name, "reference-python")
        self.assertEqual(len(result.layer_hashes), 2)
        self.assertGreater(result.duration_ms, 0.0)

    def test_fallback_executes_on_primary_failure(self) -> None:
        runner = ProverKernelRunner(
            primary_backend=_AlwaysFailBackend(),
            fallback_backend=ReferenceKernelBackend(),
        )
        req = CommitLayerRequest.from_sequences(log_size=1, columns=[[1, 2], [3, 4]])
        result = runner.commit_layer(req)
        self.assertEqual(result.backend_name, "reference-python")
        self.assertTrue(any("primary failure" in e for e in result.debug_events))
        self.assertTrue(any("fallback done" in e for e in result.debug_events))

    def test_failure_without_fallback_raises(self) -> None:
        runner = ProverKernelRunner(primary_backend=_AlwaysFailBackend())
        req = CommitLayerRequest.from_sequences(log_size=1, columns=[[1, 2], [3, 4]])
        with self.assertRaises(BackendExecutionError):
            runner.commit_layer(req)


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import gc
import unittest

from mojo_stark_prover_kernels.m31_axpy import M31AxpyRequest, m31_axpy_reference
from scripts.benchmark_m31_axpy import (
    parse_affinity_spec,
    run_bench_interleaved,
    run_bench_native,
    run_bench_reference,
)


class _FakeNativeBackend:
    def __init__(self) -> None:
        self.calls = 0

    def m31_axpy(self, request: M31AxpyRequest):
        self.calls += 1
        return m31_axpy_reference(request)


def _req() -> M31AxpyRequest:
    return M31AxpyRequest.from_sequences(
        a=[1, 2, 3, 4],
        b=[5, 6, 7, 8],
        c=[9, 10, 11, 12],
        alpha=7,
        beta=11,
    )


class BenchmarkM31AxpyTests(unittest.TestCase):
    def test_parse_affinity_spec_accepts_ranges(self) -> None:
        cpus = parse_affinity_spec("0,2,4-6")
        self.assertEqual(cpus, {0, 2, 4, 5, 6})

    def test_parse_affinity_spec_rejects_invalid_ranges(self) -> None:
        with self.assertRaises(ValueError):
            _ = parse_affinity_spec("7-3")
        with self.assertRaises(ValueError):
            _ = parse_affinity_spec("")

    def test_interleaved_produces_expected_sample_counts(self) -> None:
        backend = _FakeNativeBackend()
        req = _req()
        ref_samples, nat_samples = run_bench_interleaved(
            backend,
            req,
            iters=7,
            warmup_iters=2,
            seed=42,
        )
        self.assertEqual(len(ref_samples), 7)
        self.assertEqual(len(nat_samples), 7)
        # warmups + measured iterations
        self.assertEqual(backend.calls, 9)

    def test_disable_gc_restores_state_reference(self) -> None:
        req = _req()
        gc.enable()
        self.assertTrue(gc.isenabled())
        _ = run_bench_reference(req, iters=3, warmup_iters=1, disable_gc=True)
        self.assertTrue(gc.isenabled())

    def test_disable_gc_restores_state_native(self) -> None:
        backend = _FakeNativeBackend()
        req = _req()
        gc.enable()
        self.assertTrue(gc.isenabled())
        _ = run_bench_native(backend, req, iters=3, warmup_iters=1, disable_gc=True)
        self.assertTrue(gc.isenabled())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from scripts.check_m31_perf_gate import (
    PerfMetrics,
    bootstrap_trimmed_speedup_ci,
    evaluate_gate,
    summarize_metrics,
)


class M31PerfGateTests(unittest.TestCase):
    def test_summarize_metrics_computes_speedups(self) -> None:
        ref = [100.0, 102.0, 98.0, 101.0]
        nat = [80.0, 82.0, 79.0, 81.0]
        metrics = summarize_metrics(ref, nat, trim_ratio=0.1)
        self.assertGreater(metrics.speedup_mean, 1.0)
        self.assertGreater(metrics.speedup_median, 1.0)
        self.assertGreater(metrics.speedup_trimmed, 1.0)

    def test_evaluate_gate_passes_for_good_metrics(self) -> None:
        metrics = PerfMetrics(
            ref_mean=100.0,
            ref_median=100.0,
            ref_p95=110.0,
            ref_trimmed=100.0,
            nat_mean=80.0,
            nat_median=80.0,
            nat_p95=120.0,
            nat_trimmed=80.0,
            speedup_mean=1.25,
            speedup_median=1.25,
            speedup_trimmed=1.25,
        )
        failures = evaluate_gate(
            metrics,
            min_trimmed_speedup=1.05,
            min_median_speedup=1.01,
            max_native_p95_over_median=2.0,
            min_trimmed_speedup_ci_low=1.0,
        )
        self.assertEqual(failures, [])

    def test_evaluate_gate_fails_trimmed_speedup(self) -> None:
        metrics = PerfMetrics(
            ref_mean=100.0,
            ref_median=100.0,
            ref_p95=110.0,
            ref_trimmed=100.0,
            nat_mean=98.0,
            nat_median=90.0,
            nat_p95=120.0,
            nat_trimmed=98.0,
            speedup_mean=1.02,
            speedup_median=1.11,
            speedup_trimmed=1.02,
        )
        failures = evaluate_gate(
            metrics,
            min_trimmed_speedup=1.05,
            min_median_speedup=1.01,
            max_native_p95_over_median=2.0,
            min_trimmed_speedup_ci_low=1.0,
        )
        self.assertTrue(any("trimmed speedup" in failure for failure in failures))

    def test_evaluate_gate_fails_tail_ratio(self) -> None:
        metrics = PerfMetrics(
            ref_mean=100.0,
            ref_median=100.0,
            ref_p95=110.0,
            ref_trimmed=100.0,
            nat_mean=80.0,
            nat_median=80.0,
            nat_p95=220.0,
            nat_trimmed=80.0,
            speedup_mean=1.25,
            speedup_median=1.25,
            speedup_trimmed=1.25,
        )
        failures = evaluate_gate(
            metrics,
            min_trimmed_speedup=1.05,
            min_median_speedup=1.01,
            max_native_p95_over_median=2.5,
            min_trimmed_speedup_ci_low=1.0,
        )
        self.assertTrue(any("instability" in failure for failure in failures))

    def test_bootstrap_trimmed_speedup_ci_orders_bounds(self) -> None:
        ref = [100.0, 102.0, 98.0, 101.0, 99.0]
        nat = [80.0, 82.0, 79.0, 81.0, 80.0]
        low, high = bootstrap_trimmed_speedup_ci(
            ref,
            nat,
            trim_ratio=0.1,
            resamples=200,
            confidence=0.95,
            seed=7,
        )
        self.assertLessEqual(low, high)
        self.assertGreater(low, 1.0)

    def test_evaluate_gate_fails_ci_lower_bound(self) -> None:
        metrics = PerfMetrics(
            ref_mean=100.0,
            ref_median=100.0,
            ref_p95=110.0,
            ref_trimmed=100.0,
            nat_mean=80.0,
            nat_median=80.0,
            nat_p95=90.0,
            nat_trimmed=80.0,
            speedup_mean=1.25,
            speedup_median=1.25,
            speedup_trimmed=1.25,
            speedup_trimmed_ci_low=0.99,
            speedup_trimmed_ci_high=1.30,
        )
        failures = evaluate_gate(
            metrics,
            min_trimmed_speedup=1.05,
            min_median_speedup=1.01,
            max_native_p95_over_median=2.5,
            min_trimmed_speedup_ci_low=1.0,
        )
        self.assertTrue(any("bootstrap CI lower-bound" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from scripts.ci_perf_gate import (
    AggregateMetrics,
    RunResult,
    _build_export_args,
    _extract_artifact_path,
    evaluate_aggregate_gate,
    summarize,
)


class CIBenchmarkGateTests(unittest.TestCase):
    def test_extract_artifact_path(self) -> None:
        out = "\n".join(
            [
                "artifact_json=/tmp/a.json",
                "artifact_markdown=/tmp/a.md",
            ]
        )
        self.assertEqual(str(_extract_artifact_path(out, "artifact_json")), "/tmp/a.json")
        self.assertEqual(str(_extract_artifact_path(out, "artifact_markdown")), "/tmp/a.md")
        self.assertIsNone(_extract_artifact_path(out, "missing"))

    def test_summarize_uses_only_passing_runs(self) -> None:
        runs = [
            RunResult(
                index=1,
                seed=1,
                returncode=0,
                artifact_json=None,
                artifact_markdown=None,
                trimmed_speedup=1.10,
                median_speedup=1.05,
                ci_low=0.97,
                tail_ratio=1.2,
                stdout="",
                stderr="",
            ),
            RunResult(
                index=2,
                seed=2,
                returncode=1,
                artifact_json=None,
                artifact_markdown=None,
                trimmed_speedup=0.50,
                median_speedup=0.50,
                ci_low=0.10,
                tail_ratio=9.9,
                stdout="",
                stderr="",
            ),
            RunResult(
                index=3,
                seed=3,
                returncode=0,
                artifact_json=None,
                artifact_markdown=None,
                trimmed_speedup=1.20,
                median_speedup=1.08,
                ci_low=1.01,
                tail_ratio=1.4,
                stdout="",
                stderr="",
            ),
        ]
        metrics = summarize(runs)
        self.assertEqual(metrics.runs_total, 3)
        self.assertEqual(metrics.runs_passed, 2)
        self.assertAlmostEqual(metrics.trimmed_median, 1.15, places=6)
        self.assertAlmostEqual(metrics.median_speedup_median, 1.065, places=6)
        self.assertAlmostEqual(metrics.ci_low_median, 0.99, places=6)
        self.assertAlmostEqual(metrics.tail_ratio_median, 1.3, places=6)

    def test_evaluate_aggregate_gate_failures(self) -> None:
        metrics = AggregateMetrics(
            runs_total=3,
            runs_passed=1,
            trimmed_median=0.99,
            median_speedup_median=0.98,
            ci_low_median=0.80,
            tail_ratio_median=6.0,
        )
        failures = evaluate_aggregate_gate(
            metrics,
            min_pass_runs=2,
            min_trimmed_speedup=1.05,
            min_median_speedup=1.01,
            min_trimmed_speedup_ci_low=0.90,
            max_native_p95_over_median=5.0,
        )
        self.assertEqual(len(failures), 5)

    def test_build_export_args_contains_required_fields(self) -> None:
        class Args:
            length = 65536
            seed = 20260305
            iters = 80
            warmup_iters = 40
            trim_ratio = 0.1
            rayon_threads = 2
            affinity = None
            target_cpu_native = "on"
            interleaved = "on"
            disable_gc = True
            min_trimmed_speedup = 1.05
            min_median_speedup = 1.01
            min_trimmed_speedup_ci_low = 0.90
            max_native_p95_over_median = 5.0
            bootstrap_resamples = 2000
            bootstrap_confidence = 0.95
            bootstrap_seed = 20260305

        args = _build_export_args(Args(), seed=20260322)
        self.assertIn("--disable-gc", args)
        self.assertIn("--rayon-threads", args)
        self.assertIn("2", args)
        seed_idx = args.index("--seed")
        self.assertEqual(args[seed_idx + 1], "20260322")


if __name__ == "__main__":
    unittest.main()

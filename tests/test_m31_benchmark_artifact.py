from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_m31_perf_gate import PerfMetrics
from scripts.export_m31_benchmark_artifact import (
    build_report,
    collect_environment_fingerprint,
    metrics_to_dict,
    render_markdown_report,
    resolve_output_paths,
    write_report_files,
)


def _args_namespace() -> argparse.Namespace:
    return argparse.Namespace(
        length=65536,
        seed=20260305,
        iters=20,
        warmup_iters=10,
        trim_ratio=0.1,
        rayon_threads=8,
        affinity=None,
        target_cpu_native="on",
        interleaved="on",
        disable_gc=False,
        bootstrap_resamples=2000,
        bootstrap_confidence=0.95,
        bootstrap_seed=20260305,
        min_trimmed_speedup=1.05,
        min_median_speedup=1.01,
        min_trimmed_speedup_ci_low=1.00,
        max_native_p95_over_median=2.5,
    )


def _metrics() -> PerfMetrics:
    return PerfMetrics(
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
    )


class M31BenchmarkArtifactTests(unittest.TestCase):
    def test_resolve_output_paths_rejects_bad_basename(self) -> None:
        with self.assertRaises(ValueError):
            resolve_output_paths(output_dir=Path("/tmp"), basename="bad/name")

    def test_collect_environment_fingerprint_with_tooling(self) -> None:
        with patch("scripts.export_m31_benchmark_artifact._run_command") as run_mock:
            run_mock.side_effect = [
                (0, "cargo 1.86.0", ""),
                (0, "rustc 1.86.0", ""),
                (0, "deadbeef", ""),
                (0, "main", ""),
                (0, " M file.txt", ""),
            ]
            fp = collect_environment_fingerprint(
                repo_root=Path("/tmp"),
                env={
                    "RAYON_NUM_THREADS": "8",
                    "MSPK_ENABLE_TARGET_CPU_NATIVE": "1",
                    "MSPK_KERNEL_STRICT": "true",
                },
            )
        self.assertEqual(fp["toolchain"]["cargo"], "cargo 1.86.0")
        self.assertEqual(fp["toolchain"]["rustc"], "rustc 1.86.0")
        self.assertTrue(fp["git"]["available"])
        self.assertTrue(fp["git"]["dirty"])
        self.assertEqual(fp["runtime_flags"]["kernel_strict_env"], True)
        self.assertIn("process_affinity", fp["cpu"])

    def test_collect_environment_fingerprint_without_tooling(self) -> None:
        with patch("scripts.export_m31_benchmark_artifact._run_command", return_value=None):
            fp = collect_environment_fingerprint(repo_root=Path("/tmp"), env={})
        self.assertIsNone(fp["toolchain"]["cargo"])
        self.assertIsNone(fp["toolchain"]["rustc"])
        self.assertFalse(fp["git"]["available"])
        self.assertIsNone(fp["git"]["head"])

    def test_build_report_and_markdown_render(self) -> None:
        report = build_report(
            generated_at_utc="2026-03-06T00:00:00Z",
            args=_args_namespace(),
            metrics=_metrics(),
            gate_failures=[],
            reference_samples_ms=[100.0, 101.0],
            native_samples_ms=[80.0, 81.0],
            native_artifact_path=Path("/tmp/libkernel.so"),
            native_artifact_sha256="ab" * 32,
            fingerprint={
                "platform": {
                    "system": "Linux",
                    "release": "6.0",
                    "machine": "x86_64",
                    "python_version": "3.11.0",
                },
                "cpu": {"logical_cores": 8, "process_affinity": [2, 3]},
                "runtime_flags": {
                    "rayon_num_threads": "8",
                    "target_cpu_native_env": "1",
                },
                "toolchain": {"cargo": "cargo 1.86.0", "rustc": "rustc 1.86.0"},
                "git": {"available": True, "head": "deadbeef", "branch": "main", "dirty": True},
            },
        )
        self.assertTrue(report["gate"]["pass"])
        self.assertEqual(report["native_artifact"]["sha256"], "ab" * 32)

        markdown = render_markdown_report(report)
        self.assertIn("Gate: **PASS**", markdown)
        self.assertIn("speedup_trimmed_mean_x", markdown)
        self.assertIn("deadbeef", markdown)

    def test_write_report_files_emits_json_and_markdown(self) -> None:
        report = {
            "schema_version": 1,
            "generated_at_utc": "2026-03-06T00:00:00Z",
            "gate": {"pass": True, "failures": []},
            "parameters": {
                "length": 1,
                "seed": 1,
                "iters": 1,
                "warmup_iters": 0,
                "trim_ratio": 0.1,
                "rayon_threads": None,
                "affinity": None,
                "target_cpu_native": "on",
                "interleaved": "on",
                "disable_gc": False,
                "bootstrap_resamples": 10,
                "bootstrap_confidence": 0.95,
                "bootstrap_seed": 1,
            },
            "metrics": {
                "speedup_trimmed_mean_x": 1.1,
                "speedup_median_x": 1.1,
                "native_p95_over_median": 1.0,
            },
            "native_artifact": {"path": "/tmp/lib.so", "sha256": "ab" * 32},
            "fingerprint": {
                "platform": {"system": "Linux", "release": "6", "machine": "x86_64", "python_version": "3.11"},
                "cpu": {"logical_cores": 8, "process_affinity": None},
                "runtime_flags": {"rayon_num_threads": None, "target_cpu_native_env": "1"},
                "toolchain": {"cargo": None, "rustc": None},
                "git": {"available": False, "head": None, "branch": None, "dirty": None},
            },
        }
        with tempfile.TemporaryDirectory() as d:
            json_out = Path(d) / "artifact.json"
            md_out = Path(d) / "artifact.md"
            write_report_files(report=report, json_out=json_out, md_out=md_out)
            self.assertTrue(json_out.exists())
            self.assertTrue(md_out.exists())
            loaded = json.loads(json_out.read_text())
            self.assertEqual(loaded["schema_version"], 1)

    def test_metrics_to_dict_rejects_non_finite_metrics(self) -> None:
        bad = PerfMetrics(
            ref_mean=100.0,
            ref_median=100.0,
            ref_p95=110.0,
            ref_trimmed=100.0,
            nat_mean=80.0,
            nat_median=0.0,
            nat_p95=90.0,
            nat_trimmed=80.0,
            speedup_mean=1.25,
            speedup_median=1.25,
            speedup_trimmed=1.25,
        )
        with self.assertRaises(ValueError):
            metrics_to_dict(bad)

    def test_write_report_files_rejects_non_finite_json_numbers(self) -> None:
        report = {
            "schema_version": 1,
            "generated_at_utc": "2026-03-06T00:00:00Z",
            "gate": {"pass": True, "failures": []},
            "parameters": {"length": 1},
            "metrics": {"speedup_trimmed_mean_x": math.nan},
            "native_artifact": {"path": "/tmp/lib.so", "sha256": "ab" * 32},
            "fingerprint": {"platform": {}, "cpu": {}, "runtime_flags": {}, "toolchain": {}, "git": {}},
        }
        with tempfile.TemporaryDirectory() as d:
            json_out = Path(d) / "artifact.json"
            with self.assertRaises(ValueError):
                write_report_files(report=report, json_out=json_out, md_out=None)


if __name__ == "__main__":
    unittest.main()

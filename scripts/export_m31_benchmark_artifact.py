#!/usr/bin/env python3
"""Run M31 benchmark/gate and emit reproducible JSON + Markdown artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
import platform
from pathlib import Path
import subprocess  # nosec B404
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mojo_stark_prover_kernels.native_backend import (  # noqa: E402
    NativeRustM31Backend,
    build_native_kernel_with_sha256,
    repository_root,
)
from scripts.benchmark_m31_axpy import (  # noqa: E402
    build_request,
    maybe_set_cpu_affinity,
    run_bench_interleaved,
    run_bench_native,
    run_bench_reference,
)
from scripts.check_m31_perf_gate import PerfMetrics, evaluate_gate, summarize_metrics  # noqa: E402


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(ts: datetime) -> str:
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _artifact_stamp(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ")


def _run_command(cmd: list[str], *, cwd: Path, timeout_sec: float = 10.0) -> tuple[int, str, str] | None:
    try:
        proc = subprocess.run(  # nosec B603
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _parse_bool_env(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def collect_environment_fingerprint(*, repo_root: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    runtime_env = os.environ if env is None else env

    cargo_version = _run_command(["cargo", "--version"], cwd=repo_root)
    rustc_version = _run_command(["rustc", "--version"], cwd=repo_root)
    git_head = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_branch = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    git_status = _run_command(["git", "status", "--porcelain"], cwd=repo_root)

    git_available = git_head is not None and git_head[0] == 0
    git_dirty: bool | None = None
    if git_status is not None and git_status[0] == 0:
        git_dirty = bool(git_status[1])

    current_affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            current_affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            current_affinity = None

    return {
        "captured_at_utc": _format_utc(_utc_now()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "cpu": {
            "logical_cores": os.cpu_count(),
            "process_affinity": current_affinity,
        },
        "runtime_flags": {
            "rayon_num_threads": runtime_env.get("RAYON_NUM_THREADS"),
            "target_cpu_native_env": runtime_env.get("MSPK_ENABLE_TARGET_CPU_NATIVE"),
            "kernel_strict_env": _parse_bool_env(runtime_env.get("MSPK_KERNEL_STRICT")),
        },
        "toolchain": {
            "cargo": cargo_version[1] if cargo_version is not None and cargo_version[0] == 0 else None,
            "rustc": rustc_version[1] if rustc_version is not None and rustc_version[0] == 0 else None,
        },
        "git": {
            "available": git_available,
            "head": git_head[1] if git_available else None,
            "branch": git_branch[1] if git_branch is not None and git_branch[0] == 0 else None,
            "dirty": git_dirty,
        },
    }


def resolve_output_paths(*, output_dir: Path, basename: str) -> tuple[Path, Path]:
    stem = basename.strip()
    if not stem:
        raise ValueError("basename cannot be empty")
    if "/" in stem or "\\" in stem:
        raise ValueError("basename must not contain path separators")
    return output_dir / f"{stem}.json", output_dir / f"{stem}.md"


def metrics_to_dict(metrics: PerfMetrics) -> dict[str, float]:
    metrics_out = {
        "reference_mean_ms": metrics.ref_mean,
        "reference_median_ms": metrics.ref_median,
        "reference_p95_ms": metrics.ref_p95,
        "reference_trimmed_mean_ms": metrics.ref_trimmed,
        "native_mean_ms": metrics.nat_mean,
        "native_median_ms": metrics.nat_median,
        "native_p95_ms": metrics.nat_p95,
        "native_trimmed_mean_ms": metrics.nat_trimmed,
        "speedup_mean_x": metrics.speedup_mean,
        "speedup_median_x": metrics.speedup_median,
        "speedup_trimmed_mean_x": metrics.speedup_trimmed,
        "native_p95_over_median": metrics.nat_p95 / metrics.nat_median if metrics.nat_median > 0 else float("inf"),
    }
    if metrics.speedup_trimmed_ci_low is not None:
        metrics_out["speedup_trimmed_ci_low_x"] = metrics.speedup_trimmed_ci_low
    if metrics.speedup_trimmed_ci_high is not None:
        metrics_out["speedup_trimmed_ci_high_x"] = metrics.speedup_trimmed_ci_high
    for key, value in metrics_out.items():
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric {key}: {value!r}")
    return metrics_out


def build_report(
    *,
    generated_at_utc: str,
    args: argparse.Namespace,
    metrics: PerfMetrics,
    gate_failures: list[str],
    reference_samples_ms: list[float],
    native_samples_ms: list[float],
    native_artifact_path: Path,
    native_artifact_sha256: str,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc,
        "workload": "m31_axpy",
        "parameters": {
            "length": args.length,
            "seed": args.seed,
            "iters": args.iters,
            "warmup_iters": args.warmup_iters,
            "trim_ratio": args.trim_ratio,
            "rayon_threads": args.rayon_threads,
            "affinity": args.affinity,
            "target_cpu_native": args.target_cpu_native,
            "interleaved": args.interleaved,
            "disable_gc": args.disable_gc,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_confidence": args.bootstrap_confidence,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "gate": {
            "pass": not gate_failures,
            "thresholds": {
                "min_trimmed_speedup": args.min_trimmed_speedup,
                "min_median_speedup": args.min_median_speedup,
                "min_trimmed_speedup_ci_low": args.min_trimmed_speedup_ci_low,
                "max_native_p95_over_median": args.max_native_p95_over_median,
            },
            "failures": gate_failures,
        },
        "metrics": metrics_to_dict(metrics),
        "samples_ms": {
            "reference": reference_samples_ms,
            "native": native_samples_ms,
        },
        "native_artifact": {
            "path": str(native_artifact_path),
            "sha256": native_artifact_sha256,
        },
        "fingerprint": fingerprint,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    gate_status = "PASS" if report["gate"]["pass"] else "FAIL"
    params = report["parameters"]
    metrics = report["metrics"]
    artifact = report["native_artifact"]
    fp = report["fingerprint"]
    git_fp = fp["git"]
    runtime_flags = fp["runtime_flags"]
    toolchain = fp["toolchain"]

    failures = report["gate"]["failures"]
    failure_lines = "\n".join(f"- {f}" for f in failures) if failures else "- none"

    return "\n".join(
        [
            "# M31 Benchmark Artifact",
            "",
            f"- Generated: {report['generated_at_utc']}",
            f"- Gate: **{gate_status}**",
            "",
            "## Parameters",
            "",
            f"- length: `{params['length']}`",
            f"- seed: `{params['seed']}`",
            f"- iters: `{params['iters']}`",
            f"- warmup_iters: `{params['warmup_iters']}`",
            f"- trim_ratio: `{params['trim_ratio']}`",
            f"- rayon_threads: `{params['rayon_threads']}`",
            f"- affinity: `{params['affinity']}`",
            f"- target_cpu_native: `{params['target_cpu_native']}`",
            f"- interleaved: `{params['interleaved']}`",
            f"- disable_gc: `{params['disable_gc']}`",
            "",
            "## Metrics",
            "",
            f"- speedup_trimmed_mean_x: `{metrics['speedup_trimmed_mean_x']:.3f}`",
            f"- speedup_median_x: `{metrics['speedup_median_x']:.3f}`",
            f"- native_p95_over_median: `{metrics['native_p95_over_median']:.3f}`",
            (
                f"- speedup_trimmed_ci_low_x: `{metrics['speedup_trimmed_ci_low_x']:.3f}`"
                if "speedup_trimmed_ci_low_x" in metrics
                else "- speedup_trimmed_ci_low_x: `n/a`"
            ),
            (
                f"- speedup_trimmed_ci_high_x: `{metrics['speedup_trimmed_ci_high_x']:.3f}`"
                if "speedup_trimmed_ci_high_x" in metrics
                else "- speedup_trimmed_ci_high_x: `n/a`"
            ),
            "",
            "## Gate Failures",
            "",
            failure_lines,
            "",
            "## Native Artifact",
            "",
            f"- path: `{artifact['path']}`",
            f"- sha256: `{artifact['sha256']}`",
            "",
            "## Toolchain Fingerprint",
            "",
            f"- cargo: `{toolchain['cargo']}`",
            f"- rustc: `{toolchain['rustc']}`",
            "",
            "## Runtime Fingerprint",
            "",
            f"- os: `{fp['platform']['system']} {fp['platform']['release']}`",
            f"- machine: `{fp['platform']['machine']}`",
            f"- python: `{fp['platform']['python_version']}`",
            f"- logical_cores: `{fp['cpu']['logical_cores']}`",
            f"- process_affinity: `{fp['cpu']['process_affinity']}`",
            f"- RAYON_NUM_THREADS: `{runtime_flags['rayon_num_threads']}`",
            f"- MSPK_ENABLE_TARGET_CPU_NATIVE: `{runtime_flags['target_cpu_native_env']}`",
            "",
            "## Git Fingerprint",
            "",
            f"- available: `{git_fp['available']}`",
            f"- head: `{git_fp['head']}`",
            f"- branch: `{git_fp['branch']}`",
            f"- dirty: `{git_fp['dirty']}`",
            "",
        ]
    )


def write_report_files(*, report: dict[str, Any], json_out: Path, md_out: Path | None) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(render_markdown_report(report))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run M31 benchmark and emit reproducible JSON/Markdown artifacts."
    )
    parser.add_argument("--length", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20260305)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup-iters", type=int, default=40)
    parser.add_argument("--trim-ratio", type=float, default=0.1)
    parser.add_argument("--rayon-threads", type=int, default=None)
    parser.add_argument(
        "--affinity",
        type=str,
        default=None,
        help="optional CPU affinity list, e.g. '2,3,6-7' (Linux only)",
    )
    parser.add_argument("--target-cpu-native", choices=["on", "off"], default="on")
    parser.add_argument(
        "--interleaved",
        choices=["on", "off"],
        default="on",
        help="randomly interleave reference/native samples to reduce drift",
    )
    parser.add_argument(
        "--disable-gc",
        action="store_true",
        help="disable Python cyclic GC during benchmark timing loops",
    )
    parser.add_argument("--min-trimmed-speedup", type=float, default=1.05)
    parser.add_argument("--min-median-speedup", type=float, default=1.01)
    parser.add_argument("--min-trimmed-speedup-ci-low", type=float, default=0.90)
    parser.add_argument("--max-native-p95-over-median", type=float, default=5.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260305)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "benchmarks",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help="artifact basename; default is m31_axpy_<UTC timestamp>",
    )
    parser.add_argument("--no-markdown", action="store_true")
    args = parser.parse_args()

    if args.length <= 0:
        raise SystemExit("--length must be positive")
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.warmup_iters < 0:
        raise SystemExit("--warmup-iters cannot be negative")
    if args.trim_ratio < 0 or args.trim_ratio >= 0.5:
        raise SystemExit("--trim-ratio must be in [0.0, 0.5)")
    if args.rayon_threads is not None and args.rayon_threads <= 0:
        raise SystemExit("--rayon-threads must be positive")
    if args.min_trimmed_speedup <= 0 or args.min_median_speedup <= 0:
        raise SystemExit("speedup thresholds must be positive")
    if args.min_trimmed_speedup_ci_low <= 0:
        raise SystemExit("--min-trimmed-speedup-ci-low must be positive")
    if args.max_native_p95_over_median <= 1.0:
        raise SystemExit("--max-native-p95-over-median must be > 1")
    if args.bootstrap_resamples <= 0:
        raise SystemExit("--bootstrap-resamples must be positive")
    if args.bootstrap_confidence <= 0.0 or args.bootstrap_confidence >= 1.0:
        raise SystemExit("--bootstrap-confidence must be in (0, 1)")

    if args.affinity is not None:
        try:
            _ = maybe_set_cpu_affinity(args.affinity)
        except (ValueError, RuntimeError, OSError) as exc:
            raise SystemExit(f"failed to apply --affinity: {exc}") from exc

    if args.rayon_threads is not None:
        os.environ["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    os.environ["MSPK_ENABLE_TARGET_CPU_NATIVE"] = "1" if args.target_cpu_native == "on" else "0"

    started = _utc_now()
    basename = args.basename or f"m31_axpy_{_artifact_stamp(started)}"
    json_out, md_out = resolve_output_paths(output_dir=args.output_dir, basename=basename)

    request = build_request(args.length, args.seed)
    artifact_path, artifact_sha256 = build_native_kernel_with_sha256(release=True)
    native = NativeRustM31Backend(
        artifact_path,
        allow_relative_path=True,
        expected_sha256=artifact_sha256,
    )

    if args.interleaved == "on":
        reference_samples, native_samples = run_bench_interleaved(
            native,
            request,
            args.iters,
            args.warmup_iters,
            seed=args.seed,
            disable_gc=args.disable_gc,
        )
    else:
        reference_samples = run_bench_reference(
            request, args.iters, args.warmup_iters, disable_gc=args.disable_gc
        )
        native_samples = run_bench_native(
            native, request, args.iters, args.warmup_iters, disable_gc=args.disable_gc
        )
    metrics = summarize_metrics(
        reference_samples,
        native_samples,
        args.trim_ratio,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    gate_failures = evaluate_gate(
        metrics,
        min_trimmed_speedup=args.min_trimmed_speedup,
        min_median_speedup=args.min_median_speedup,
        max_native_p95_over_median=args.max_native_p95_over_median,
        min_trimmed_speedup_ci_low=args.min_trimmed_speedup_ci_low,
    )
    fingerprint = collect_environment_fingerprint(repo_root=repository_root())

    try:
        report = build_report(
            generated_at_utc=_format_utc(_utc_now()),
            args=args,
            metrics=metrics,
            gate_failures=gate_failures,
            reference_samples_ms=reference_samples,
            native_samples_ms=native_samples,
            native_artifact_path=artifact_path,
            native_artifact_sha256=artifact_sha256,
            fingerprint=fingerprint,
        )
        write_report_files(
            report=report,
            json_out=json_out,
            md_out=None if args.no_markdown else md_out,
        )
    except ValueError as exc:
        raise SystemExit(f"failed to serialize benchmark artifact: {exc}") from exc

    print(f"artifact_json={json_out}")
    if not args.no_markdown:
        print(f"artifact_markdown={md_out}")
    metric_line = (
        "metrics: "
        f"speedup(trimmed)={metrics.speedup_trimmed:.3f}x "
        f"speedup(median)={metrics.speedup_median:.3f}x "
        f"nat_p95/median={(metrics.nat_p95 / metrics.nat_median):.3f}"
    )
    if metrics.speedup_trimmed_ci_low is not None and metrics.speedup_trimmed_ci_high is not None:
        metric_line += (
            f" speedup(trimmed)_ci{int(args.bootstrap_confidence * 100)}="
            f"[{metrics.speedup_trimmed_ci_low:.3f}x,{metrics.speedup_trimmed_ci_high:.3f}x]"
        )
    print(metric_line)
    print(f"gate: {'PASS' if not gate_failures else 'FAIL'}")
    return 0 if not gate_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

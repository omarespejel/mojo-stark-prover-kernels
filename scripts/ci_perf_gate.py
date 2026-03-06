#!/usr/bin/env python3
"""Run benchmark export multiple times and enforce an aggregate CI gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import statistics
import subprocess  # nosec B404
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export_m31_benchmark_artifact.py"


@dataclass(frozen=True)
class RunResult:
    index: int
    seed: int
    returncode: int
    artifact_json: Path | None
    artifact_markdown: Path | None
    trimmed_speedup: float | None
    median_speedup: float | None
    ci_low: float | None
    tail_ratio: float | None
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class AggregateMetrics:
    runs_total: int
    runs_passed: int
    trimmed_median: float
    median_speedup_median: float
    ci_low_median: float
    tail_ratio_median: float


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ")


def _extract_artifact_path(output: str, key: str) -> Path | None:
    prefix = f"{key}="
    for line in output.splitlines():
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            if value:
                return Path(value)
    return None


def _float_from_metrics(metrics: dict[str, object], key: str) -> float | None:
    raw = metrics.get(key)
    if raw is None:
        return None
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return float(raw)
    return None


def _read_metrics(path: Path) -> tuple[float | None, float | None, float | None, float | None]:
    payload = json.loads(path.read_text())
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return None, None, None, None
    trimmed = _float_from_metrics(metrics, "speedup_trimmed_mean_x")
    median = _float_from_metrics(metrics, "speedup_median_x")
    ci_low = _float_from_metrics(metrics, "speedup_trimmed_ci_low_x")
    tail_ratio = _float_from_metrics(metrics, "native_p95_over_median")
    return trimmed, median, ci_low, tail_ratio


def run_once(run_index: int, seed: int, export_args: list[str]) -> RunResult:
    cmd = [sys.executable, str(EXPORT_SCRIPT), *export_args]
    proc = subprocess.run(  # nosec B603
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    artifact_json = _extract_artifact_path(proc.stdout, "artifact_json")
    artifact_markdown = _extract_artifact_path(proc.stdout, "artifact_markdown")
    trimmed = median = ci_low = tail_ratio = None
    if artifact_json is not None and artifact_json.exists():
        trimmed, median, ci_low, tail_ratio = _read_metrics(artifact_json)

    return RunResult(
        index=run_index,
        seed=seed,
        returncode=proc.returncode,
        artifact_json=artifact_json,
        artifact_markdown=artifact_markdown,
        trimmed_speedup=trimmed,
        median_speedup=median,
        ci_low=ci_low,
        tail_ratio=tail_ratio,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _median_or_fail(values: list[float], label: str) -> float:
    if not values:
        raise ValueError(f"no values for {label}")
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"non-finite value for {label}: {value!r}")
        if value <= 0:
            raise ValueError(f"non-positive value for {label}: {value!r}")
    median = float(statistics.median(values))
    if not math.isfinite(median):
        raise ValueError(f"non-finite median for {label}: {median!r}")
    if median <= 0:
        raise ValueError(f"non-positive median for {label}: {median!r}")
    return median


def summarize(results: list[RunResult]) -> AggregateMetrics:
    passed = [r for r in results if r.passed]
    trimmed_values = [r.trimmed_speedup for r in passed if r.trimmed_speedup is not None]
    median_values = [r.median_speedup for r in passed if r.median_speedup is not None]
    ci_low_values = [r.ci_low for r in passed if r.ci_low is not None]
    tail_values = [r.tail_ratio for r in passed if r.tail_ratio is not None]
    return AggregateMetrics(
        runs_total=len(results),
        runs_passed=len(passed),
        trimmed_median=_median_or_fail(trimmed_values, "trimmed speedup"),
        median_speedup_median=_median_or_fail(median_values, "median speedup"),
        ci_low_median=_median_or_fail(ci_low_values, "CI lower bound"),
        tail_ratio_median=_median_or_fail(tail_values, "tail ratio"),
    )


def evaluate_aggregate_gate(
    metrics: AggregateMetrics,
    *,
    min_pass_runs: int,
    min_trimmed_speedup: float,
    min_median_speedup: float,
    min_trimmed_speedup_ci_low: float,
    max_native_p95_over_median: float,
) -> list[str]:
    failures: list[str] = []
    finite_checks = [
        ("aggregate trimmed speedup", metrics.trimmed_median),
        ("aggregate median speedup", metrics.median_speedup_median),
        ("aggregate CI lower-bound", metrics.ci_low_median),
        ("aggregate native tail ratio", metrics.tail_ratio_median),
    ]
    for label, value in finite_checks:
        if not math.isfinite(value):
            failures.append(f"{label} is non-finite: {value!r}")
            continue
        if value <= 0:
            failures.append(f"{label} must be positive: {value!r}")
    if metrics.runs_passed < min_pass_runs:
        failures.append(
            f"insufficient passing runs: {metrics.runs_passed}/{metrics.runs_total} < {min_pass_runs}"
        )
    if metrics.trimmed_median < min_trimmed_speedup:
        failures.append(
            "aggregate trimmed speedup below threshold: "
            f"{metrics.trimmed_median:.3f}x < {min_trimmed_speedup:.3f}x"
        )
    if metrics.median_speedup_median < min_median_speedup:
        failures.append(
            "aggregate median speedup below threshold: "
            f"{metrics.median_speedup_median:.3f}x < {min_median_speedup:.3f}x"
        )
    if metrics.ci_low_median < min_trimmed_speedup_ci_low:
        failures.append(
            "aggregate CI lower-bound below threshold: "
            f"{metrics.ci_low_median:.3f}x < {min_trimmed_speedup_ci_low:.3f}x"
        )
    if metrics.tail_ratio_median > max_native_p95_over_median:
        failures.append(
            "aggregate native tail ratio too high: "
            f"{metrics.tail_ratio_median:.3f} > {max_native_p95_over_median:.3f}"
        )
    return failures


def _build_export_args(args: argparse.Namespace, *, seed: int) -> list[str]:
    export_args: list[str] = [
        "--length",
        str(args.length),
        "--seed",
        str(seed),
        "--iters",
        str(args.iters),
        "--warmup-iters",
        str(args.warmup_iters),
        "--trim-ratio",
        str(args.trim_ratio),
        "--target-cpu-native",
        args.target_cpu_native,
        "--interleaved",
        args.interleaved,
        "--min-trimmed-speedup",
        str(args.min_trimmed_speedup),
        "--min-median-speedup",
        str(args.min_median_speedup),
        "--min-trimmed-speedup-ci-low",
        str(args.min_trimmed_speedup_ci_low),
        "--max-native-p95-over-median",
        str(args.max_native_p95_over_median),
        "--bootstrap-resamples",
        str(args.bootstrap_resamples),
        "--bootstrap-confidence",
        str(args.bootstrap_confidence),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
    ]
    if args.rayon_threads is not None:
        export_args.extend(["--rayon-threads", str(args.rayon_threads)])
    if args.affinity is not None:
        export_args.extend(["--affinity", args.affinity])
    if args.disable_gc:
        export_args.append("--disable-gc")
    return export_args


def _write_aggregate_artifacts(
    *,
    results: list[RunResult],
    metrics: AggregateMetrics,
    failures: list[str],
    out_json: Path,
    out_md: Path,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "generated_at_utc": _now_utc().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "runs": [
            {
                "index": r.index,
                "seed": r.seed,
                "returncode": r.returncode,
                "artifact_json": str(r.artifact_json) if r.artifact_json is not None else None,
                "artifact_markdown": (
                    str(r.artifact_markdown) if r.artifact_markdown is not None else None
                ),
                "trimmed_speedup": r.trimmed_speedup,
                "median_speedup": r.median_speedup,
                "ci_low": r.ci_low,
                "tail_ratio": r.tail_ratio,
            }
            for r in results
        ],
        "aggregate": {
            "runs_total": metrics.runs_total,
            "runs_passed": metrics.runs_passed,
            "trimmed_speedup_median": metrics.trimmed_median,
            "median_speedup_median": metrics.median_speedup_median,
            "ci_low_median": metrics.ci_low_median,
            "tail_ratio_median": metrics.tail_ratio_median,
        },
        "gate": {
            "pass": not failures,
            "failures": failures,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")

    lines = [
        "# CI Benchmark Aggregate",
        "",
        f"- Gate: **{'PASS' if not failures else 'FAIL'}**",
        f"- Runs passed: `{metrics.runs_passed}/{metrics.runs_total}`",
        f"- aggregate trimmed speedup median: `{metrics.trimmed_median:.3f}x`",
        f"- aggregate median speedup median: `{metrics.median_speedup_median:.3f}x`",
        f"- aggregate ci_low median: `{metrics.ci_low_median:.3f}x`",
        f"- aggregate tail ratio median: `{metrics.tail_ratio_median:.3f}`",
        "",
        "## Failures",
    ]
    if failures:
        lines.extend([f"- {failure}" for failure in failures])
    else:
        lines.append("- none")
    lines.extend(["", "## Runs"])
    for r in results:
        lines.append(
            f"- run {r.index}: seed={r.seed} rc={r.returncode} trimmed={r.trimmed_speedup} "
            f"median={r.median_speedup} ci_low={r.ci_low} tail={r.tail_ratio}"
        )
        if r.artifact_json is not None:
            lines.append(f"  - artifact_json: `{r.artifact_json}`")
        if r.artifact_markdown is not None:
            lines.append(f"  - artifact_markdown: `{r.artifact_markdown}`")
    out_md.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-run CI perf gate wrapper.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--min-pass-runs", type=int, default=2)
    parser.add_argument("--length", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20260305)
    parser.add_argument(
        "--seed-step",
        type=int,
        default=7919,
        help="per-run seed increment for multi-run CI sampling",
    )
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup-iters", type=int, default=40)
    parser.add_argument("--trim-ratio", type=float, default=0.1)
    parser.add_argument("--rayon-threads", type=int, default=2)
    parser.add_argument("--affinity", type=str, default=None)
    parser.add_argument("--target-cpu-native", choices=["on", "off"], default="on")
    parser.add_argument("--interleaved", choices=["on", "off"], default="on")
    parser.add_argument("--disable-gc", action="store_true")
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
        help="basename for aggregate artifacts; default ci_m31_aggregate_<UTC stamp>",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    if args.min_pass_runs <= 0 or args.min_pass_runs > args.runs:
        raise SystemExit("--min-pass-runs must be in [1, --runs]")
    if args.seed_step < 0:
        raise SystemExit("--seed-step must be >= 0")
    if args.runs > 1 and args.seed_step == 0:
        raise SystemExit("--seed-step must be > 0 when --runs > 1")

    results: list[RunResult] = []
    for i in range(args.runs):
        run_index = i + 1
        run_seed = args.seed + (i * args.seed_step)
        export_args = _build_export_args(args, seed=run_seed)
        results.append(run_once(run_index, run_seed, export_args))

    try:
        metrics = summarize(results)
        failures = evaluate_aggregate_gate(
            metrics,
            min_pass_runs=args.min_pass_runs,
            min_trimmed_speedup=args.min_trimmed_speedup,
            min_median_speedup=args.min_median_speedup,
            min_trimmed_speedup_ci_low=args.min_trimmed_speedup_ci_low,
            max_native_p95_over_median=args.max_native_p95_over_median,
        )
    except ValueError as exc:
        failures = [str(exc)]
        metrics = AggregateMetrics(
            runs_total=len(results),
            runs_passed=sum(1 for r in results if r.passed),
            trimmed_median=0.0,
            median_speedup_median=0.0,
            ci_low_median=0.0,
            tail_ratio_median=0.0,
        )

    stamp = _stamp(_now_utc())
    basename = args.basename or f"ci_m31_aggregate_{stamp}"
    out_json = args.output_dir / f"{basename}.json"
    out_md = args.output_dir / f"{basename}.md"
    _write_aggregate_artifacts(
        results=results,
        metrics=metrics,
        failures=failures,
        out_json=out_json,
        out_md=out_md,
    )

    print(f"artifact_json={out_json}")
    print(f"artifact_markdown={out_md}")
    if failures:
        print("gate: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        "metrics: "
        f"runs_passed={metrics.runs_passed}/{metrics.runs_total} "
        f"trimmed_median={metrics.trimmed_median:.3f}x "
        f"median_speedup_median={metrics.median_speedup_median:.3f}x "
        f"ci_low_median={metrics.ci_low_median:.3f}x "
        f"tail_ratio_median={metrics.tail_ratio_median:.3f}"
    )
    print("gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run M31 benchmark and fail if performance gate thresholds are not met."""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_m31_axpy import (  # noqa: E402
    _trimmed_mean,
    build_request,
    maybe_set_cpu_affinity,
    run_bench_interleaved,
    run_bench_native,
    run_bench_reference,
)
from mojo_stark_prover_kernels.native_backend import NativeRustM31Backend  # noqa: E402


@dataclass(frozen=True)
class PerfMetrics:
    ref_mean: float
    ref_median: float
    ref_p95: float
    ref_trimmed: float
    nat_mean: float
    nat_median: float
    nat_p95: float
    nat_trimmed: float
    speedup_mean: float
    speedup_median: float
    speedup_trimmed: float
    speedup_trimmed_ci_low: float | None = None
    speedup_trimmed_ci_high: float | None = None


def bootstrap_trimmed_speedup_ci(
    ref_samples: list[float],
    nat_samples: list[float],
    *,
    trim_ratio: float,
    resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    if not ref_samples or not nat_samples:
        raise ValueError("sample lists cannot be empty")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence must be in (0, 1)")

    rnd = random.Random(seed)  # nosec B311
    n_ref = len(ref_samples)
    n_nat = len(nat_samples)
    boot: list[float] = []
    for _ in range(resamples):
        ref_draw = [ref_samples[rnd.randrange(0, n_ref)] for _ in range(n_ref)]
        nat_draw = [nat_samples[rnd.randrange(0, n_nat)] for _ in range(n_nat)]
        ref_trimmed = _trimmed_mean(ref_draw, trim_ratio)
        nat_trimmed = _trimmed_mean(nat_draw, trim_ratio)
        speedup = ref_trimmed / nat_trimmed if nat_trimmed > 0 else float("inf")
        boot.append(speedup)

    alpha = (1.0 - confidence) / 2.0
    lower = _percentile(boot, alpha * 100.0)
    upper = _percentile(boot, (1.0 - alpha) * 100.0)
    return lower, upper


def summarize_metrics(
    ref_samples: list[float],
    nat_samples: list[float],
    trim_ratio: float,
    *,
    bootstrap_resamples: int = 0,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> PerfMetrics:
    if not ref_samples or not nat_samples:
        raise ValueError("sample lists cannot be empty")

    ref_mean = statistics.mean(ref_samples)
    ref_median = statistics.median(ref_samples)
    ref_p95 = _percentile(ref_samples, 95.0)
    ref_trimmed = _trimmed_mean(ref_samples, trim_ratio)

    nat_mean = statistics.mean(nat_samples)
    nat_median = statistics.median(nat_samples)
    nat_p95 = _percentile(nat_samples, 95.0)
    nat_trimmed = _trimmed_mean(nat_samples, trim_ratio)

    speedup_mean = ref_mean / nat_mean if nat_mean > 0 else float("inf")
    speedup_median = ref_median / nat_median if nat_median > 0 else float("inf")
    speedup_trimmed = ref_trimmed / nat_trimmed if nat_trimmed > 0 else float("inf")

    ci_low: float | None = None
    ci_high: float | None = None
    if bootstrap_resamples > 0:
        ci_low, ci_high = bootstrap_trimmed_speedup_ci(
            ref_samples,
            nat_samples,
            trim_ratio=trim_ratio,
            resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )

    return PerfMetrics(
        ref_mean=ref_mean,
        ref_median=ref_median,
        ref_p95=ref_p95,
        ref_trimmed=ref_trimmed,
        nat_mean=nat_mean,
        nat_median=nat_median,
        nat_p95=nat_p95,
        nat_trimmed=nat_trimmed,
        speedup_mean=speedup_mean,
        speedup_median=speedup_median,
        speedup_trimmed=speedup_trimmed,
        speedup_trimmed_ci_low=ci_low,
        speedup_trimmed_ci_high=ci_high,
    )


def evaluate_gate(
    metrics: PerfMetrics,
    *,
    min_trimmed_speedup: float,
    min_median_speedup: float,
    max_native_p95_over_median: float,
    min_trimmed_speedup_ci_low: float,
) -> list[str]:
    failures: list[str] = []
    if metrics.speedup_trimmed < min_trimmed_speedup:
        failures.append(
            "trimmed speedup below threshold: "
            f"{metrics.speedup_trimmed:.3f}x < {min_trimmed_speedup:.3f}x"
        )
    if metrics.speedup_median < min_median_speedup:
        failures.append(
            "median speedup below threshold: "
            f"{metrics.speedup_median:.3f}x < {min_median_speedup:.3f}x"
        )
    if (
        metrics.speedup_trimmed_ci_low is not None
        and metrics.speedup_trimmed_ci_low < min_trimmed_speedup_ci_low
    ):
        failures.append(
            "bootstrap CI lower-bound speedup below threshold: "
            f"{metrics.speedup_trimmed_ci_low:.3f}x < {min_trimmed_speedup_ci_low:.3f}x"
        )
    tail_ratio = metrics.nat_p95 / metrics.nat_median if metrics.nat_median > 0 else float("inf")
    if tail_ratio > max_native_p95_over_median:
        failures.append(
            "native latency instability too high: "
            f"p95/median={tail_ratio:.3f} > {max_native_p95_over_median:.3f}"
        )
    return failures


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def main() -> int:
    parser = argparse.ArgumentParser(description="Performance gate for native M31 backend.")
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
    parser.add_argument(
        "--validate-m31-output",
        choices=["on", "off"],
        default="off",
        help="enable strict Python-side per-element output validation after native calls",
    )
    parser.add_argument("--min-trimmed-speedup", type=float, default=1.05)
    parser.add_argument("--min-median-speedup", type=float, default=1.01)
    parser.add_argument("--min-trimmed-speedup-ci-low", type=float, default=0.90)
    parser.add_argument("--max-native-p95-over-median", type=float, default=5.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260305)
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

    req = build_request(args.length, args.seed)
    native = NativeRustM31Backend.build_and_create(
        release=True,
        validate_m31_output=(args.validate_m31_output == "on"),
    )
    if args.interleaved == "on":
        ref_samples, nat_samples = run_bench_interleaved(
            native,
            req,
            args.iters,
            args.warmup_iters,
            seed=args.seed,
            disable_gc=args.disable_gc,
        )
    else:
        ref_samples = run_bench_reference(
            req, args.iters, args.warmup_iters, disable_gc=args.disable_gc
        )
        nat_samples = run_bench_native(
            native, req, args.iters, args.warmup_iters, disable_gc=args.disable_gc
        )
    metrics = summarize_metrics(
        ref_samples,
        nat_samples,
        args.trim_ratio,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    failures = evaluate_gate(
        metrics,
        min_trimmed_speedup=args.min_trimmed_speedup,
        min_median_speedup=args.min_median_speedup,
        max_native_p95_over_median=args.max_native_p95_over_median,
        min_trimmed_speedup_ci_low=args.min_trimmed_speedup_ci_low,
    )

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
    if failures:
        print("gate: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

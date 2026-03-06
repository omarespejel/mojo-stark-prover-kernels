#!/usr/bin/env python3
"""Micro-benchmark for reference vs native kernel backend."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mojo_stark_prover_kernels.backends import ReferenceKernelBackend
from mojo_stark_prover_kernels.contracts import CommitLayerRequest
from mojo_stark_prover_kernels.native_backend import NativeRustKernelBackend


def build_request(log_size: int, n_columns: int, with_prev: bool) -> CommitLayerRequest:
    n_rows = 1 << log_size
    columns = []
    for c in range(n_columns):
        columns.append([((c + 1) * 1_000_003 + r * 17) & 0xFFFFFFFF for r in range(n_rows)])
    prev = None
    if with_prev:
        prev = [bytes(((i + j) & 0xFF) for j in range(32)) for i in range(1 << (log_size + 1))]
    return CommitLayerRequest.from_sequences(
        log_size=log_size,
        columns=columns,
        prev_layer_hashes=prev,
        debug_level=0,
    )


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


def _trimmed_mean(values: list[float], trim_ratio: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    if trim_ratio <= 0:
        return statistics.mean(values)
    ordered = sorted(values)
    trim = int(len(ordered) * trim_ratio)
    if trim * 2 >= len(ordered):
        return statistics.mean(ordered)
    core = ordered[trim : len(ordered) - trim]
    return statistics.mean(core)


def run_bench(backend, req: CommitLayerRequest, iters: int, *, warmup_iters: int) -> list[float]:
    for _ in range(max(0, warmup_iters)):
        _ = backend.commit_layer(req)

    durations_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = backend.commit_layer(req)
        dt = (time.perf_counter() - t0) * 1000.0
        durations_ms.append(dt)
    return durations_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark reference vs native backend.")
    parser.add_argument("--log-size", type=int, default=8)
    parser.add_argument("--columns", type=int, default=16)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--with-prev", action="store_true")
    parser.add_argument("--rayon-threads", type=int, default=None)
    parser.add_argument(
        "--target-cpu-native",
        choices=["on", "off"],
        default="on",
        help="toggle Rust -C target-cpu=native for native backend build",
    )
    parser.add_argument("--trim-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.warmup_iters < 0:
        raise SystemExit("--warmup-iters cannot be negative")
    if args.rayon_threads is not None and args.rayon_threads <= 0:
        raise SystemExit("--rayon-threads must be positive")
    if args.trim_ratio < 0 or args.trim_ratio >= 0.5:
        raise SystemExit("--trim-ratio must be in [0.0, 0.5)")

    if args.rayon_threads is not None:
        os.environ["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    os.environ["MSPK_ENABLE_TARGET_CPU_NATIVE"] = "1" if args.target_cpu_native == "on" else "0"

    req = build_request(args.log_size, args.columns, args.with_prev)
    ref = ReferenceKernelBackend()
    native = NativeRustKernelBackend.build_and_create(release=True)

    ref_samples = run_bench(ref, req, args.iters, warmup_iters=args.warmup_iters)
    nat_samples = run_bench(native, req, args.iters, warmup_iters=args.warmup_iters)

    ref_mean = statistics.mean(ref_samples)
    ref_median = statistics.median(ref_samples)
    ref_p95 = _percentile(ref_samples, 95.0)
    ref_max = max(ref_samples)
    ref_trim = _trimmed_mean(ref_samples, args.trim_ratio)

    nat_mean = statistics.mean(nat_samples)
    nat_median = statistics.median(nat_samples)
    nat_p95 = _percentile(nat_samples, 95.0)
    nat_max = max(nat_samples)
    nat_trim = _trimmed_mean(nat_samples, args.trim_ratio)

    speedup = ref_mean / nat_mean if nat_mean > 0 else float("inf")
    speedup_median = ref_median / nat_median if nat_median > 0 else float("inf")
    speedup_trim = ref_trim / nat_trim if nat_trim > 0 else float("inf")

    print(f"rows={1 << args.log_size} columns={args.columns} with_prev={args.with_prev}")
    print(f"iters={args.iters} warmup_iters={args.warmup_iters}")
    print(f"target_cpu_native={args.target_cpu_native}")
    if args.rayon_threads is not None:
        print(f"rayon_threads={args.rayon_threads}")
    print("reference(ms):")
    print(
        "  "
        f"mean={ref_mean:.4f} median={ref_median:.4f} p95={ref_p95:.4f} "
        f"trimmed_mean={ref_trim:.4f} max={ref_max:.4f}"
    )
    print("native(ms):")
    print(
        "  "
        f"mean={nat_mean:.4f} median={nat_median:.4f} p95={nat_p95:.4f} "
        f"trimmed_mean={nat_trim:.4f} max={nat_max:.4f}"
    )
    print(f"speedup(mean)={speedup:.3f}x")
    print(f"speedup(median)={speedup_median:.3f}x")
    print(f"speedup(trimmed_mean)={speedup_trim:.3f}x trim_ratio={args.trim_ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

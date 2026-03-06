#!/usr/bin/env python3
"""Benchmark M31 batch linear-combination: reference vs native backend."""

from __future__ import annotations

import argparse
import gc
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mojo_stark_prover_kernels.m31_axpy import M31AxpyRequest, M31_PRIME, m31_axpy_reference
from mojo_stark_prover_kernels.native_backend import NativeRustM31Backend


def parse_affinity_spec(spec: str) -> set[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            if start < 0 or end < 0 or end < start:
                raise ValueError(f"invalid affinity range: {token}")
            cpus.update(range(start, end + 1))
            continue
        cpu = int(token)
        if cpu < 0:
            raise ValueError(f"invalid cpu id: {token}")
        cpus.add(cpu)
    if not cpus:
        raise ValueError("affinity spec cannot be empty")
    return cpus


def maybe_set_cpu_affinity(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    cpus = parse_affinity_spec(spec)
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU affinity is not supported on this platform")
    os.sched_setaffinity(0, cpus)
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return sorted(cpus)


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


def _measure_call_ms(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter_ns()
    _ = fn()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1_000_000.0


def build_request(length: int, seed: int) -> M31AxpyRequest:
    # Deterministic benchmark fixture generation; not used for cryptography.
    rnd = random.Random(seed)  # nosec B311
    a = [rnd.randrange(0, M31_PRIME) for _ in range(length)]
    b = [rnd.randrange(0, M31_PRIME) for _ in range(length)]
    c = [rnd.randrange(0, M31_PRIME) for _ in range(length)]
    alpha = rnd.randrange(0, M31_PRIME)
    beta = rnd.randrange(0, M31_PRIME)
    return M31AxpyRequest.from_sequences(a=a, b=b, c=c, alpha=alpha, beta=beta)


def run_bench_reference(
    req: M31AxpyRequest, iters: int, warmup_iters: int, *, disable_gc: bool = False
) -> list[float]:
    gc_was_enabled = gc.isenabled()
    if disable_gc and gc_was_enabled:
        gc.disable()
    try:
        for _ in range(max(0, warmup_iters)):
            _ = m31_axpy_reference(req)

        durations_ms: list[float] = []
        for _ in range(iters):
            durations_ms.append(_measure_call_ms(lambda: m31_axpy_reference(req)))
        return durations_ms
    finally:
        if disable_gc and gc_was_enabled and not gc.isenabled():
            gc.enable()


def run_bench_native(
    backend: NativeRustM31Backend,
    req: M31AxpyRequest,
    iters: int,
    warmup_iters: int,
    *,
    disable_gc: bool = False,
) -> list[float]:
    gc_was_enabled = gc.isenabled()
    if disable_gc and gc_was_enabled:
        gc.disable()
    try:
        for _ in range(max(0, warmup_iters)):
            _ = backend.m31_axpy(req)

        durations_ms: list[float] = []
        for _ in range(iters):
            durations_ms.append(_measure_call_ms(lambda: backend.m31_axpy(req)))
        return durations_ms
    finally:
        if disable_gc and gc_was_enabled and not gc.isenabled():
            gc.enable()


def run_bench_interleaved(
    backend: NativeRustM31Backend,
    req: M31AxpyRequest,
    iters: int,
    warmup_iters: int,
    *,
    seed: int,
    disable_gc: bool = False,
) -> tuple[list[float], list[float]]:
    gc_was_enabled = gc.isenabled()
    if disable_gc and gc_was_enabled:
        gc.disable()
    try:
        for _ in range(max(0, warmup_iters)):
            _ = m31_axpy_reference(req)
            _ = backend.m31_axpy(req)

        rng = random.Random(seed)  # nosec B311
        ref_samples: list[float] = []
        nat_samples: list[float] = []
        for _ in range(iters):
            if rng.random() < 0.5:
                ref_samples.append(_measure_call_ms(lambda: m31_axpy_reference(req)))
                nat_samples.append(_measure_call_ms(lambda: backend.m31_axpy(req)))
            else:
                nat_samples.append(_measure_call_ms(lambda: backend.m31_axpy(req)))
                ref_samples.append(_measure_call_ms(lambda: m31_axpy_reference(req)))
        return ref_samples, nat_samples
    finally:
        if disable_gc and gc_was_enabled and not gc.isenabled():
            gc.enable()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark M31 AXPY reference vs native backend.")
    parser.add_argument("--length", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20260305)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--rayon-threads", type=int, default=None)
    parser.add_argument(
        "--affinity",
        type=str,
        default=None,
        help="optional CPU affinity list, e.g. '2,3,6-7' (Linux only)",
    )
    parser.add_argument(
        "--target-cpu-native",
        choices=["on", "off"],
        default="on",
        help="toggle Rust -C target-cpu=native for native backend build",
    )
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
    parser.add_argument("--trim-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if args.length <= 0:
        raise SystemExit("--length must be positive")
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.warmup_iters < 0:
        raise SystemExit("--warmup-iters cannot be negative")
    if args.rayon_threads is not None and args.rayon_threads <= 0:
        raise SystemExit("--rayon-threads must be positive")
    if args.trim_ratio < 0 or args.trim_ratio >= 0.5:
        raise SystemExit("--trim-ratio must be in [0.0, 0.5)")

    affinity_effective: list[int] | None = None
    if args.affinity is not None:
        try:
            affinity_effective = maybe_set_cpu_affinity(args.affinity)
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

    ref_mean = statistics.mean(ref_samples)
    ref_median = statistics.median(ref_samples)
    ref_p95 = _percentile(ref_samples, 95.0)
    ref_trim = _trimmed_mean(ref_samples, args.trim_ratio)

    nat_mean = statistics.mean(nat_samples)
    nat_median = statistics.median(nat_samples)
    nat_p95 = _percentile(nat_samples, 95.0)
    nat_trim = _trimmed_mean(nat_samples, args.trim_ratio)

    speedup_mean = ref_mean / nat_mean if nat_mean > 0 else float("inf")
    speedup_median = ref_median / nat_median if nat_median > 0 else float("inf")
    speedup_trim = ref_trim / nat_trim if nat_trim > 0 else float("inf")

    print(f"length={args.length} seed={args.seed}")
    print(f"iters={args.iters} warmup_iters={args.warmup_iters}")
    print(f"target_cpu_native={args.target_cpu_native}")
    print(f"interleaved={args.interleaved}")
    print(f"disable_gc={args.disable_gc}")
    print(f"validate_m31_output={args.validate_m31_output}")
    if args.rayon_threads is not None:
        print(f"rayon_threads={args.rayon_threads}")
    if affinity_effective is not None:
        print(f"cpu_affinity={affinity_effective}")
    print("reference(ms):")
    print(
        "  "
        f"mean={ref_mean:.4f} median={ref_median:.4f} p95={ref_p95:.4f} "
        f"trimmed_mean={ref_trim:.4f}"
    )
    print("native(ms):")
    print(
        "  "
        f"mean={nat_mean:.4f} median={nat_median:.4f} p95={nat_p95:.4f} "
        f"trimmed_mean={nat_trim:.4f}"
    )
    print(f"speedup(mean)={speedup_mean:.3f}x")
    print(f"speedup(median)={speedup_median:.3f}x")
    print(f"speedup(trimmed_mean)={speedup_trim:.3f}x trim_ratio={args.trim_ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

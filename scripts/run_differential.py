#!/usr/bin/env python3
"""Run randomized differential checks for a candidate backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mojo_stark_prover_kernels.backends import (
    BackendExecutionError,
    MojoSharedLibraryBackend,
    ReferenceKernelBackend,
)
from mojo_stark_prover_kernels.differential import run_randomized_suite
from mojo_stark_prover_kernels.native_backend import NativeBuildError, NativeRustKernelBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomized differential correctness suite.")
    parser.add_argument(
        "--backend",
        choices=["reference", "mojo-shared-lib", "native-rust"],
        default="reference",
        help="Candidate backend to test.",
    )
    parser.add_argument(
        "--shared-lib",
        type=Path,
        default=None,
        help="Absolute path to shared library when --backend=mojo-shared-lib",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cases", type=int, default=25)
    parser.add_argument("--max-log-size", type=int, default=6)
    parser.add_argument("--max-columns", type=int, default=6)
    parser.add_argument("--debug-level", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.backend == "reference":
        candidate = ReferenceKernelBackend()
    elif args.backend == "native-rust":
        try:
            candidate = NativeRustKernelBackend.build_and_create(release=True)
        except NativeBuildError as exc:
            print(f"ERROR failed to build native backend: {exc}", file=sys.stderr)
            return 2
    else:
        if args.shared_lib is None:
            print("ERROR --shared-lib is required for mojo-shared-lib backend", file=sys.stderr)
            return 2
        try:
            candidate = MojoSharedLibraryBackend(args.shared_lib)
        except Exception as exc:
            print(f"ERROR failed to initialize backend: {exc}", file=sys.stderr)
            return 2

    try:
        suite = run_randomized_suite(
            candidate_backend=candidate,
            seed=args.seed,
            n_cases=args.cases,
            max_log_size=args.max_log_size,
            max_columns=args.max_columns,
            debug_level=args.debug_level,
            fail_fast=args.fail_fast,
        )
    except BackendExecutionError as exc:
        print(f"ERROR backend execution failed: {exc}", file=sys.stderr)
        return 3

    print(f"candidate={candidate.name}")
    print(f"total_cases={suite.total_cases}")
    print(f"passed_cases={suite.passed_cases}")
    print(f"failed_cases={suite.failed_cases}")
    if suite.failures:
        print("failures:")
        for failure in suite.failures[:10]:
            print(
                "  case="
                f"{failure.case_id} mismatch_index={failure.mismatch_index} "
                f"oracle={failure.oracle_hex} candidate={failure.candidate_hex}"
            )
    print("debug_events_sample:")
    for event in suite.debug_events[: min(20, len(suite.debug_events))]:
        print(f"  {event}")

    return 0 if suite.failed_cases == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

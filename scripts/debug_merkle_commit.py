#!/usr/bin/env python3
"""Debug driver for the reference layer commitment path."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mojo_stark_prover_kernels.backends import BackendExecutionError, ReferenceKernelBackend
from mojo_stark_prover_kernels.contracts import CommitLayerRequest
from mojo_stark_prover_kernels.runner import ProverKernelRunner


@dataclass
class FailingBackend:
    name: str = "failing-backend-demo"

    def commit_layer(self, request: CommitLayerRequest, tracer=None):  # type: ignore[override]
        raise BackendExecutionError("simulated backend failure for fallback demo")


def build_columns(log_size: int, n_columns: int) -> list[list[int]]:
    n_rows = 1 << log_size
    columns = []
    for c in range(n_columns):
        columns.append([(c + 1) * 1000 + r for r in range(n_rows)])
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug runner for layer commitment.")
    parser.add_argument("--log-size", type=int, default=3)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--debug-level", type=int, default=2)
    parser.add_argument("--simulate-primary-failure", action="store_true")
    args = parser.parse_args()

    request = CommitLayerRequest.from_sequences(
        log_size=args.log_size,
        columns=build_columns(args.log_size, args.columns),
        debug_level=args.debug_level,
    )

    if args.simulate_primary_failure:
        runner = ProverKernelRunner(
            primary_backend=FailingBackend(),
            fallback_backend=ReferenceKernelBackend(),
            default_debug_level=args.debug_level,
        )
    else:
        runner = ProverKernelRunner(
            primary_backend=ReferenceKernelBackend(),
            default_debug_level=args.debug_level,
        )

    result = runner.commit_layer(request)
    print(f"backend={result.backend_name}")
    print(f"duration_ms={result.duration_ms:.3f}")
    print("first_hashes:")
    for digest in result.layer_hashes[: min(4, len(result.layer_hashes))]:
        print(f"  {digest.hex()}")
    print("debug_events:")
    for event in result.debug_events:
        print(f"  {event}")


if __name__ == "__main__":
    main()

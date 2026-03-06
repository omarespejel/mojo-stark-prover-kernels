"""Differential correctness harness for kernel backends."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .backends import KernelBackend, ReferenceKernelBackend
from .contracts import CommitLayerRequest
from .debug import DebugTracer


@dataclass(frozen=True)
class DifferentialCaseResult:
    case_id: int
    match: bool
    request: CommitLayerRequest
    oracle_backend_name: str
    candidate_backend_name: str
    oracle_hashes: tuple[bytes, ...]
    candidate_hashes: tuple[bytes, ...]
    mismatch_index: int | None
    oracle_hex: str | None
    candidate_hex: str | None


@dataclass(frozen=True)
class DifferentialSuiteResult:
    total_cases: int
    passed_cases: int
    failed_cases: int
    failures: tuple[DifferentialCaseResult, ...]
    debug_events: tuple[str, ...]


def compare_case(
    *,
    case_id: int,
    request: CommitLayerRequest,
    candidate_backend: KernelBackend,
    oracle_backend: KernelBackend | None = None,
    tracer: DebugTracer | None = None,
) -> DifferentialCaseResult:
    request.validate()
    oracle = oracle_backend if oracle_backend is not None else ReferenceKernelBackend()
    if tracer is not None:
        tracer.emit(1, f"differential case={case_id} start")

    oracle_hashes = oracle.commit_layer(request, tracer=tracer)
    candidate_hashes = candidate_backend.commit_layer(request, tracer=tracer)

    mismatch_index = None
    oracle_hex = None
    candidate_hex = None

    if len(oracle_hashes) != len(candidate_hashes):
        mismatch_index = -1
        oracle_hex = f"len={len(oracle_hashes)}"
        candidate_hex = f"len={len(candidate_hashes)}"
    else:
        for idx, (lhs, rhs) in enumerate(zip(oracle_hashes, candidate_hashes)):
            if lhs != rhs:
                mismatch_index = idx
                oracle_hex = lhs.hex()
                candidate_hex = rhs.hex()
                break

    matched = mismatch_index is None
    if tracer is not None:
        tracer.emit(
            1,
            f"differential case={case_id} done match={matched} mismatch_index={mismatch_index}",
        )

    return DifferentialCaseResult(
        case_id=case_id,
        match=matched,
        request=request,
        oracle_backend_name=oracle.name,
        candidate_backend_name=candidate_backend.name,
        oracle_hashes=oracle_hashes,
        candidate_hashes=candidate_hashes,
        mismatch_index=mismatch_index,
        oracle_hex=oracle_hex,
        candidate_hex=candidate_hex,
    )


def run_randomized_suite(
    *,
    candidate_backend: KernelBackend,
    oracle_backend: KernelBackend | None = None,
    seed: int = 7,
    n_cases: int = 25,
    max_log_size: int = 6,
    max_columns: int = 6,
    debug_level: int = 1,
    fail_fast: bool = False,
) -> DifferentialSuiteResult:
    if n_cases <= 0:
        raise ValueError("n_cases must be positive")
    if max_log_size <= 0:
        raise ValueError("max_log_size must be positive")
    if max_columns <= 0:
        raise ValueError("max_columns must be positive")

    # Deterministic seeded RNG is required for reproducible differential vectors.
    rnd = random.Random(seed)  # nosec B311
    tracer = DebugTracer(level=debug_level, max_events=50000)
    oracle = oracle_backend if oracle_backend is not None else ReferenceKernelBackend()
    failures: list[DifferentialCaseResult] = []
    passed = 0

    for case_id in range(n_cases):
        req = _random_request(rnd, max_log_size=max_log_size, max_columns=max_columns)
        result = compare_case(
            case_id=case_id,
            request=req,
            candidate_backend=candidate_backend,
            oracle_backend=oracle,
            tracer=tracer,
        )
        if result.match:
            passed += 1
        else:
            failures.append(result)
            if fail_fast:
                break

    return DifferentialSuiteResult(
        total_cases=n_cases,
        passed_cases=passed,
        failed_cases=len(failures),
        failures=tuple(failures),
        debug_events=tracer.snapshot(),
    )


def _random_request(rnd: random.Random, *, max_log_size: int, max_columns: int) -> CommitLayerRequest:
    log_size = rnd.randint(1, max_log_size)
    n_rows = 1 << log_size
    n_columns = rnd.randint(1, max_columns)
    columns = [[rnd.randrange(0, 2**32) for _ in range(n_rows)] for _ in range(n_columns)]

    include_prev = rnd.choice([False, True])
    prev_layer = None
    if include_prev:
        prev_layer = [rnd.randbytes(32) for _ in range(1 << (log_size + 1))]

    return CommitLayerRequest.from_sequences(
        log_size=log_size,
        columns=columns,
        prev_layer_hashes=prev_layer,
        debug_level=0,
    )

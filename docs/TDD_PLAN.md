# TDD Plan

## Phase 1 (done in this scaffold)

1. Contract tests
- Reject invalid dimensions.
- Reject out-of-range values.
- Reject malformed previous-layer hashes.

2. Determinism tests
- Fixed vector with expected digests.
- Regression check that output is stable.

3. Operational safety tests
- Primary-backend failure with fallback success.
- Primary-backend failure without fallback.
- Shared library load hardening tests.

## Phase 2 (next)

1. Implement `mojo_blake2s_commit_layer` in Mojo.
2. Differential harness status:
- `compare_case` implemented for single-case parity checks.
- `run_randomized_suite` implemented for randomized corpus checks.
3. Fault injection status:
- non-zero return path covered via compiled shared-lib integration tests.
- debug sanitization in error path covered.
4. Remaining:
- edge-size corpus expansion for largest allowed contracts.
- strict STWO parity vectors once Mojo kernel is live.
- integrate additional prover hotspots beyond commitment path.

## Native bridge status

1. Rust `cdylib` backend implemented at `native/mojo_kernel_abi`.
2. Tests now validate:
- single-case parity with oracle
- randomized differential parity
- shared-library ABI success/error paths
- adaptive serial/parallel strategy boundary behavior
- prepared-request cache behavior (reuse, eviction, disable)
- M31 field batch linear-combination parity (reference vs native)
- M31 ABI overlap/range hard-fail checks

## Phase 3 (production hardening)

1. Performance gates:
- maintain deterministic parity
- prove end-to-end speedup over reference
- interleaved A/B sampling + bootstrap confidence intervals for speedup stability
- optional CPU affinity pinning for lower scheduling jitter on dedicated hosts
- multi-run aggregate CI perf gate (reduce hosted-runner noise flakiness)
- multi-seed CI fixtures (`seed + k*seed_step`) to reduce single-input overfitting
- strict finite-metric artifact serialization (`NaN`/`Inf` hard-fail)
2. Add benchmark artifacts per commit.
- implemented: `scripts/export_m31_benchmark_artifact.py` (JSON + Markdown + fingerprint)
3. Add reproducible CI matrix for x86_64/aarch64.
4. Add ABI fuzzing gates:
- deterministic fuzz regression tests in `cargo test`
- optional `cargo-fuzz` runtime target for malformed pointer/length stress

## Phase 4 (in progress)

1. Added M31 hotspot proxy path:
- strict request contract (`M31AxpyRequest`)
- deterministic reference implementation (`m31_axpy_reference`)
- native Rust ABI entrypoint (`mojo_m31_axpy`)
2. Added benchmark path:
- `scripts/benchmark_m31_axpy.py`
- warmup, trimmed mean, p95, Rayon and target-cpu controls

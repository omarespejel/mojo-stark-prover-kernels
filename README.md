# mojo-stark-prover-kernels

Production-oriented sandbox for accelerating STARK prover hotspots with Mojo kernels.

Current status:
- Strict kernel input contracts and validation.
- ABI handshake guard (`mojo_kernel_abi_version`) enforced at load time.
- SHA-256 artifact pinning (`MSPK_KERNEL_SHA256`) for shared-library integrity.
- Strict pinning mode (`MSPK_KERNEL_STRICT=1`) requiring hash-pinned artifacts.
- Deterministic Python reference kernel for layer commitment (`blake2s`).
- Deterministic Python reference kernel for M31 batch linear-combination (`axpy`).
- Backend runner with failover and debug tracing.
- Differential correctness harness (candidate backend vs oracle).
- Native Rust `cdylib` kernel implementing both ABIs (`mojo_blake2s_commit_layer`, `mojo_m31_axpy`).
- Heavy unit test coverage for contracts, determinism, and fallback behavior.
- Stub Mojo ABI entrypoint to lock interface before kernel implementation.

This repository is designed to let us ship in phases without compromising correctness:
1. Lock the ABI and data contracts.
2. Prove determinism and fallback behavior.
3. Replace reference backend with Mojo kernels and run differential tests.

## Layout

- `mojo_stark_prover_kernels/contracts.py`: request/result contracts and validation.
- `mojo_stark_prover_kernels/m31_axpy.py`: M31 batch linear-combination contract + reference kernel.
- `mojo_stark_prover_kernels/reference_blake2s_merkle.py`: deterministic reference kernel.
- `mojo_stark_prover_kernels/backends.py`: backend interfaces and Mojo shared-lib adapter stub.
- `mojo_stark_prover_kernels/differential.py`: randomized parity suite and mismatch reports.
- `mojo_stark_prover_kernels/native_backend.py`: build/load helpers for native Rust kernel.
- `mojo_stark_prover_kernels/runner.py`: production-style execution with fallback and trace capture.
- `tests/`: contract, determinism, and failover tests.
- `native/mojo_kernel_abi/`: Rust `cdylib` implementing the kernel ABI.
- `mojo/`: planned Mojo kernel entrypoints.
- `docs/`: architecture + TDD strategy.
- `scripts/run_native_fuzz.py`: deterministic/native fuzz workflow entrypoint.
- `scripts/benchmark_m31_axpy.py`: benchmark reference vs native M31 AXPY kernel.
- `scripts/check_m31_perf_gate.py`: regression gate for M31 performance thresholds.
- `scripts/export_m31_benchmark_artifact.py`: one-shot benchmark + gate + reproducible JSON/Markdown artifact export.

## Quick start

```bash
cd /Users/espejelomar/StarkNet/compiler-starknet/mojo-stark-prover-kernels
python3 -m unittest discover -s tests -v
```

## CI automation

- Workflow: `.github/workflows/benchmark-ci.yml`
- Triggers: every `push`, `pull_request`, and manual `workflow_dispatch`
- Enforced steps:
  - `PYTHONPATH=. pytest -q`
  - `cargo test -q` + `cargo clippy --all-targets -- -D warnings` in `native/mojo_kernel_abi`
  - benchmark export + perf gate via `scripts/export_m31_benchmark_artifact.py`
- Outputs per run:
  - JSON + Markdown benchmark artifacts uploaded via GitHub Actions artifacts
  - Markdown benchmark report appended to the job summary

Run local debug demo:

```bash
python3 scripts/debug_merkle_commit.py --log-size 3 --columns 3 --debug-level 2
```

Run differential suite (reference candidate):

```bash
python3 scripts/run_differential.py --backend reference --cases 50 --seed 11
```

Run differential suite (native Rust candidate):

```bash
python3 scripts/run_differential.py --backend native-rust --cases 50 --seed 11
```

Run benchmark (reference vs native kernel):

```bash
python3 scripts/benchmark_backends.py --log-size 9 --columns 32 --iters 30 --warmup-iters 10 --with-prev --target-cpu-native on
```

Run benchmark for M31 batch linear-combination hotspot:

```bash
python3 scripts/benchmark_m31_axpy.py --length 65536 --iters 20 --warmup-iters 10 --target-cpu-native on --rayon-threads 8 --interleaved on
```

Run perf regression gate for M31 hotspot:

```bash
python3 scripts/check_m31_perf_gate.py --length 65536 --iters 80 --warmup-iters 40 --target-cpu-native on --rayon-threads 8 --interleaved on --disable-gc --min-trimmed-speedup 1.05 --min-trimmed-speedup-ci-low 0.90
```

Export reproducible benchmark artifacts (JSON + Markdown):

```bash
python3 scripts/export_m31_benchmark_artifact.py --length 65536 --iters 80 --warmup-iters 40 --target-cpu-native on --rayon-threads 8 --interleaved on --disable-gc --output-dir reports/benchmarks
```

Optional (Linux): pin benchmark process to a stable CPU set to reduce scheduling noise:

```bash
python3 scripts/export_m31_benchmark_artifact.py --length 65536 --iters 80 --warmup-iters 40 --target-cpu-native on --rayon-threads 8 --interleaved on --disable-gc --affinity 2-5 --output-dir reports/benchmarks
```

Benchmark with explicit Rayon thread count (optional):

```bash
python3 scripts/benchmark_backends.py --log-size 9 --columns 32 --iters 30 --warmup-iters 10 --with-prev --rayon-threads 1 --target-cpu-native on
```

Disable host-specific codegen for portability comparisons:

```bash
python3 scripts/benchmark_backends.py --log-size 9 --columns 32 --iters 30 --warmup-iters 10 --with-prev --target-cpu-native off
```

Run deterministic native ABI fuzz smoke checks:

```bash
python3 scripts/run_native_fuzz.py --mode smoke
```

Run full native ABI fuzzing (requires `cargo-fuzz`):

```bash
python3 scripts/run_native_fuzz.py --mode full --fuzz-seconds 60
```

## Notes

- The current `blake2s` reference kernel is a deterministic contract baseline, not yet bit-for-bit STWO parity.
- The current M31 AXPY kernel is a realistic prover-side hotspot proxy for field batch linear-combination work.
- The native Rust `cdylib` is the immediate production bridge for ABI and performance validation while Mojo shared-library output is finalized.
- `NativeRust*Backend.build_and_create()` computes and pins the built artifact SHA-256 before loading.
- For production use, pin the artifact hash: `export MSPK_KERNEL_SHA256=<64-hex-sha256>`.
- To enforce pinned artifacts at load-time, set: `export MSPK_KERNEL_STRICT=1`.

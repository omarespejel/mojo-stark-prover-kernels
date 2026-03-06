# Architecture

## Goal

Accelerate prover hotspots with Mojo kernels while preserving deterministic correctness and safe failover.

## Components

1. `CommitLayerRequest` contract
- Hard bounds on log-size, columns, debug level, and value ranges.
- Validates all inputs before touching any backend.

2. `ReferenceKernelBackend`
- Deterministic Python implementation.
- Serves as test oracle and fallback path.

3. `M31AxpyRequest` + `m31_axpy_reference`
- Deterministic batch linear-combination over M31:
  - `out[i] = alpha*a[i] + beta*b[i] + c[i] (mod p)`
  - `p = 2^31 - 1`
- Used as a realistic prover-hotspot proxy with strict field-range checks.

4. `MojoSharedLibraryBackend`
- Planned production backend.
- Hardened loader:
  - rejects non-absolute paths by default
  - rejects symlink libraries
  - rejects world-writable libraries
- Bounded debug buffer.
- Bounded LRU cache for prepared FFI buffers to reduce per-call marshaling overhead.
- Exposes two ABIs when available:
  - `mojo_blake2s_commit_layer`
  - `mojo_m31_axpy`

5. `NativeRustKernelBackend`
- Production bridge backend using a Rust `cdylib` that implements the exact ABI.
- Purpose:
  - prove ABI contracts under heavy tests
  - provide immediate performance baseline against Python reference
  - keep differential guarantees while Mojo shared-lib path matures
  - enforce ABI hardening with deterministic fuzz regression + `cargo-fuzz` target
  - use adaptive execution mode (`serial` for small workloads, `parallel` for large workloads)
  - support optional host-tuned codegen (`-C target-cpu=native`) for local benchmark builds
  - enforce strict M31 field range checks in native ABI path (reject out-of-field inputs)

6. `ProverKernelRunner`
- Calls primary backend.
- On failure, can fail over to reference backend.
- Captures structured debug events and duration.

## Security assumptions

1. Kernel backend can fail arbitrarily.
2. Inputs can be malformed or intentionally oversized.
3. Dynamic library paths are high-risk and must be constrained.
4. C ABI pointer/length handling must be stress-tested via malformed-input fuzzing.
5. Dynamic kernels must fail hard on shape/range mismatches rather than silently coerce.

## Current trust model

1. Reference backend is trusted baseline for deterministic behavior.
2. Mojo backend is untrusted until differential tests prove parity.
3. Native Rust backend is trusted only after passing deterministic and randomized differential suites.

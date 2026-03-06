# Native ABI Fuzzing

This project includes two complementary fuzz layers for `mojo_blake2s_commit_layer`:

1. Deterministic fuzz regression tests (`cargo test`)
- Implemented in `native/mojo_kernel_abi/src/lib.rs` (`fuzz_abi_entrypoint` + test module).
- Runs on stable toolchains.
- Ensures malformed ABI-shaped inputs do not panic and return expected status classes.

2. `cargo-fuzz` target (`libFuzzer`)
- Target: `native/mojo_kernel_abi/fuzz/fuzz_targets/abi_entrypoint.rs`.
- Continuously mutates inputs against the same `fuzz_abi_entrypoint` helper.
- Best for long-running stress campaigns and corpus minimization.

## Commands

Smoke mode (deterministic tests only):

```bash
python3 scripts/run_native_fuzz.py --mode smoke
```

Full mode (deterministic tests + libFuzzer):

```bash
cargo install cargo-fuzz
python3 scripts/run_native_fuzz.py --mode full --fuzz-seconds 60
```

## Security intent

- Exercise hostile combinations of null pointers, pointer aliasing, misalignment, and inconsistent lengths.
- Keep ABI hardening checks pinned by regression tests.
- Catch panics/memory safety regressions before integrating optimized kernels.

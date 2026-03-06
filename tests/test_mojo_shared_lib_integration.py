from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from mojo_stark_prover_kernels.backends import BackendExecutionError, MojoSharedLibraryBackend
from mojo_stark_prover_kernels.contracts import CommitLayerRequest
from mojo_stark_prover_kernels.debug import DebugTracer
from mojo_stark_prover_kernels.m31_axpy import M31AxpyRequest


def _find_c_compiler() -> str | None:
    for candidate in ("cc", "clang", "gcc"):
        found = shutil.which(candidate)
        if found is not None:
            return found
    return None


def _compile_mock_lib(temp_dir: Path) -> Path:
    compiler = _find_c_compiler()
    if compiler is None:
        raise unittest.SkipTest("no C compiler available for integration test")

    c_source = temp_dir / "mock_kernel.c"
    c_source.write_text(
        textwrap.dedent(
            r"""
            #include <stdint.h>
            #include <string.h>

            uint32_t mojo_kernel_abi_version(void) {
                return 1u;
            }

            int32_t mojo_blake2s_commit_layer(
                uint32_t log_size,
                uint8_t* prev_layer_bytes,
                uint32_t prev_layer_len,
                uint32_t* columns_flat,
                uint32_t n_columns,
                uint32_t n_rows,
                uint8_t* out_hashes,
                uint32_t out_hashes_len,
                uint32_t debug_level,
                char* debug_buffer,
                uint32_t debug_buffer_len
            ) {
                (void)prev_layer_bytes;
                (void)prev_layer_len;
                (void)columns_flat;
                (void)n_rows;
                (void)debug_level;

                uint32_t hashes = (1u << log_size);
                if (out_hashes_len < hashes * 32u) {
                    const char* msg = "out_too_small";
                    if (debug_buffer_len > 0) {
                        strncpy(debug_buffer, msg, debug_buffer_len - 1);
                        debug_buffer[debug_buffer_len - 1] = '\0';
                    }
                    return 2;
                }

                if (n_columns == 13u) {
                    const char* msg = "rc_error\n\x01";
                    if (debug_buffer_len > 0) {
                        strncpy(debug_buffer, msg, debug_buffer_len - 1);
                        debug_buffer[debug_buffer_len - 1] = '\0';
                    }
                    return 9;
                }

                for (uint32_t i = 0; i < hashes; i++) {
                    for (uint32_t j = 0; j < 32; j++) {
                        out_hashes[i * 32u + j] = (uint8_t)((i + j + n_columns) & 0xFFu);
                    }
                }

                const char* ok = "ok\nline2\t\x01";
                if (debug_buffer_len > 0) {
                    strncpy(debug_buffer, ok, debug_buffer_len - 1);
                    debug_buffer[debug_buffer_len - 1] = '\0';
                }
                return 0;
            }
            """
        )
    )

    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    lib_path = temp_dir / f"libmock_kernel{suffix}"
    if sys.platform == "darwin":
        cmd = [compiler, "-dynamiclib", "-fPIC", str(c_source), "-o", str(lib_path)]
    else:
        cmd = [compiler, "-shared", "-fPIC", str(c_source), "-o", str(lib_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    lib_path.chmod(0o755)
    return lib_path


class MojoSharedLibIntegrationTests(unittest.TestCase):
    def test_commit_layer_success_path(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lib_path = _compile_mock_lib(Path(d))
            backend = MojoSharedLibraryBackend(lib_path, allow_relative_path=True)
            req = CommitLayerRequest.from_sequences(
                log_size=2,
                columns=[[1, 2, 3, 4], [10, 20, 30, 40]],
                debug_level=2,
            )
            tracer = DebugTracer(level=2)
            result = backend.commit_layer(req, tracer=tracer)

            self.assertEqual(len(result), 4)
            self.assertEqual(result[0], bytes(((j + 2) & 0xFF) for j in range(32)))

            events = tracer.snapshot()
            self.assertTrue(any("mojo backend debug: ok\\nline2" in e for e in events))

    def test_commit_layer_non_zero_return_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lib_path = _compile_mock_lib(Path(d))
            backend = MojoSharedLibraryBackend(lib_path, allow_relative_path=True)
            cols = [[i, i + 1] for i in range(13)]
            req = CommitLayerRequest.from_sequences(
                log_size=1,
                columns=cols,
                debug_level=1,
            )

            with self.assertRaises(BackendExecutionError) as ctx:
                backend.commit_layer(req)
            msg = str(ctx.exception)
            self.assertIn("non-zero status (9)", msg)
            self.assertIn("rc_error\\n", msg)

    def test_prepared_request_cache_reused(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lib_path = _compile_mock_lib(Path(d))
            backend = MojoSharedLibraryBackend(
                lib_path,
                allow_relative_path=True,
                cache_max_entries=2,
            )
            req = CommitLayerRequest.from_sequences(
                log_size=2,
                columns=[[1, 2, 3, 4], [5, 6, 7, 8]],
                debug_level=0,
            )

            _ = backend.commit_layer(req)
            _ = backend.commit_layer(req)

            self.assertEqual(len(backend._prepared_cache), 1)
            self.assertIn(req, backend._prepared_cache)

    def test_prepared_request_cache_eviction(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lib_path = _compile_mock_lib(Path(d))
            backend = MojoSharedLibraryBackend(
                lib_path,
                allow_relative_path=True,
                cache_max_entries=1,
            )
            req_a = CommitLayerRequest.from_sequences(
                log_size=1,
                columns=[[1, 2], [3, 4]],
                debug_level=0,
            )
            req_b = CommitLayerRequest.from_sequences(
                log_size=1,
                columns=[[9, 10], [11, 12]],
                debug_level=0,
            )

            _ = backend.commit_layer(req_a)
            _ = backend.commit_layer(req_b)

            self.assertEqual(len(backend._prepared_cache), 1)
            self.assertNotIn(req_a, backend._prepared_cache)
            self.assertIn(req_b, backend._prepared_cache)

    def test_prepared_request_cache_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lib_path = _compile_mock_lib(Path(d))
            backend = MojoSharedLibraryBackend(
                lib_path,
                allow_relative_path=True,
                cache_max_entries=0,
            )
            req = CommitLayerRequest.from_sequences(
                log_size=1,
                columns=[[1, 2], [3, 4]],
                debug_level=0,
            )

            _ = backend.commit_layer(req)
            _ = backend.commit_layer(req)

            self.assertEqual(len(backend._prepared_cache), 0)

    def test_m31_call_requires_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lib_path = _compile_mock_lib(Path(d))
            backend = MojoSharedLibraryBackend(lib_path, allow_relative_path=True)
            req = M31AxpyRequest.from_sequences(
                a=[1, 2],
                b=[3, 4],
                c=[5, 6],
                alpha=7,
                beta=11,
            )

            with self.assertRaises(BackendExecutionError) as ctx:
                _ = backend.m31_axpy(req)
            self.assertIn("missing symbol: mojo_m31_axpy", str(ctx.exception))

    def test_explicit_sha256_match_allows_loading(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lib_path = _compile_mock_lib(Path(d))
            expected = hashlib.sha256(lib_path.read_bytes()).hexdigest()
            backend = MojoSharedLibraryBackend(
                lib_path,
                allow_relative_path=True,
                expected_sha256=expected,
            )
            self.assertEqual(backend.shared_lib_path, lib_path.resolve(strict=True))

    def test_explicit_sha256_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lib_path = _compile_mock_lib(Path(d))
            with self.assertRaises(BackendExecutionError) as ctx:
                _ = MojoSharedLibraryBackend(
                    lib_path,
                    allow_relative_path=True,
                    expected_sha256="0" * 64,
                )
            self.assertIn("sha256 mismatch", str(ctx.exception))

    def test_env_sha256_used_when_explicit_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lib_path = _compile_mock_lib(Path(d))
            expected = hashlib.sha256(lib_path.read_bytes()).hexdigest()
            original = os.environ.get("MSPK_KERNEL_SHA256")
            try:
                os.environ["MSPK_KERNEL_SHA256"] = expected
                backend = MojoSharedLibraryBackend(lib_path, allow_relative_path=True)
                self.assertEqual(backend.shared_lib_path, lib_path.resolve(strict=True))
            finally:
                if original is None:
                    os.environ.pop("MSPK_KERNEL_SHA256", None)
                else:
                    os.environ["MSPK_KERNEL_SHA256"] = original

    def test_explicit_sha256_overrides_env_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lib_path = _compile_mock_lib(Path(d))
            expected = hashlib.sha256(lib_path.read_bytes()).hexdigest()
            original = os.environ.get("MSPK_KERNEL_SHA256")
            try:
                os.environ["MSPK_KERNEL_SHA256"] = "0" * 64
                backend = MojoSharedLibraryBackend(
                    lib_path,
                    allow_relative_path=True,
                    expected_sha256=expected,
                )
                self.assertEqual(backend.shared_lib_path, lib_path.resolve(strict=True))
            finally:
                if original is None:
                    os.environ.pop("MSPK_KERNEL_SHA256", None)
                else:
                    os.environ["MSPK_KERNEL_SHA256"] = original

    def test_env_strict_requires_sha256_pin(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lib_path = _compile_mock_lib(Path(d))
            original_strict = os.environ.get("MSPK_KERNEL_STRICT")
            original_sha = os.environ.get("MSPK_KERNEL_SHA256")
            try:
                os.environ["MSPK_KERNEL_STRICT"] = "1"
                os.environ.pop("MSPK_KERNEL_SHA256", None)
                with self.assertRaises(BackendExecutionError) as ctx:
                    _ = MojoSharedLibraryBackend(lib_path, allow_relative_path=True)
                self.assertIn("MSPK_KERNEL_STRICT requires", str(ctx.exception))
            finally:
                if original_strict is None:
                    os.environ.pop("MSPK_KERNEL_STRICT", None)
                else:
                    os.environ["MSPK_KERNEL_STRICT"] = original_strict
                if original_sha is None:
                    os.environ.pop("MSPK_KERNEL_SHA256", None)
                else:
                    os.environ["MSPK_KERNEL_SHA256"] = original_sha

    def test_env_strict_with_sha256_pin_allows_loading(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lib_path = _compile_mock_lib(Path(d))
            expected = hashlib.sha256(lib_path.read_bytes()).hexdigest()
            original_strict = os.environ.get("MSPK_KERNEL_STRICT")
            original_sha = os.environ.get("MSPK_KERNEL_SHA256")
            try:
                os.environ["MSPK_KERNEL_STRICT"] = "true"
                os.environ["MSPK_KERNEL_SHA256"] = expected
                backend = MojoSharedLibraryBackend(lib_path, allow_relative_path=True)
                self.assertEqual(backend.shared_lib_path, lib_path.resolve(strict=True))
            finally:
                if original_strict is None:
                    os.environ.pop("MSPK_KERNEL_STRICT", None)
                else:
                    os.environ["MSPK_KERNEL_STRICT"] = original_strict
                if original_sha is None:
                    os.environ.pop("MSPK_KERNEL_SHA256", None)
                else:
                    os.environ["MSPK_KERNEL_SHA256"] = original_sha


if __name__ == "__main__":
    unittest.main()

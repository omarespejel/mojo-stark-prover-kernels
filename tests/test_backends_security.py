from __future__ import annotations

import ctypes
import os
import pathlib
import sys
import tempfile
import unittest

from mojo_stark_prover_kernels.backends import (
    EXPECTED_KERNEL_ABI_VERSION,
    BackendExecutionError,
    MojoSharedLibraryBackend,
    _normalize_sha256_hex,
    _run_m31_kernel_self_test,
    _env_flag_enabled,
    _validate_kernel_abi_version,
    _validate_library_sha256,
    _validate_m31_result,
)
from mojo_stark_prover_kernels.m31_axpy import M31_PRIME


def _platform_shared_lib_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


class MojoSharedLibraryBackendSecurityTests(unittest.TestCase):
    def test_relative_path_rejected_by_default(self) -> None:
        with self.assertRaises(ValueError):
            MojoSharedLibraryBackend("relative/path/to/libmojo.so")

    def test_missing_file_rejected(self) -> None:
        missing = pathlib.Path("/tmp/this-file-should-not-exist-1234567890-abcdef.so")
        with self.assertRaises(FileNotFoundError):
            MojoSharedLibraryBackend(missing, allow_relative_path=True)

    def test_debug_buffer_bounds_validated(self) -> None:
        with self.assertRaises(ValueError):
            MojoSharedLibraryBackend("/tmp/nonexistent.so", debug_buffer_size=8)

    def test_cache_max_entries_bounds_validated(self) -> None:
        with self.assertRaises(ValueError):
            MojoSharedLibraryBackend("/tmp/nonexistent.so", cache_max_entries=-1)

    def test_world_writable_file_rejected_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            so_path = pathlib.Path(d) / f"libfake{_platform_shared_lib_suffix()}"
            so_path.write_bytes(b"not a shared library")
            so_path.chmod(0o666)
            with self.assertRaises(PermissionError):
                MojoSharedLibraryBackend(so_path, allow_relative_path=True)

    def test_group_writable_file_rejected_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            so_path = pathlib.Path(d) / f"libfake{_platform_shared_lib_suffix()}"
            so_path.write_bytes(b"not a shared library")
            so_path.chmod(0o664)
            with self.assertRaises(PermissionError):
                MojoSharedLibraryBackend(so_path, allow_relative_path=True)

    def test_symlink_rejected_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            dpath = pathlib.Path(d)
            target = dpath / f"libreal{_platform_shared_lib_suffix()}"
            target.write_bytes(b"not a shared library")
            target.chmod(0o600)
            symlink = dpath / f"liblink{_platform_shared_lib_suffix()}"
            symlink.symlink_to(target)
            with self.assertRaises(PermissionError):
                MojoSharedLibraryBackend(symlink, allow_relative_path=True)

    def test_non_library_extension_rejected_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bad_ext = pathlib.Path(d) / "libfake.txt"
            bad_ext.write_bytes(b"not a shared library")
            bad_ext.chmod(0o600)
            with self.assertRaises(PermissionError):
                MojoSharedLibraryBackend(bad_ext, allow_relative_path=True)

    def test_group_writable_parent_directory_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            unsafe_dir = pathlib.Path(d) / "unsafe"
            unsafe_dir.mkdir()
            unsafe_dir.chmod(0o775)

            so_path = unsafe_dir / f"libfake{_platform_shared_lib_suffix()}"
            so_path.write_bytes(b"not a shared library")
            so_path.chmod(0o600)

            with self.assertRaises(PermissionError):
                MojoSharedLibraryBackend(so_path, allow_relative_path=True)

    def test_group_writable_ancestor_directory_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            ancestor = root / "ancestor"
            ancestor.mkdir()
            ancestor.chmod(0o775)

            safe_parent = ancestor / "safe"
            safe_parent.mkdir()
            safe_parent.chmod(0o700)

            so_path = safe_parent / f"libfake{_platform_shared_lib_suffix()}"
            so_path.write_bytes(b"not a shared library")
            so_path.chmod(0o600)

            with self.assertRaises(PermissionError):
                MojoSharedLibraryBackend(so_path, allow_relative_path=True)

    def test_m31_result_length_mismatch_rejected(self) -> None:
        with self.assertRaises(BackendExecutionError):
            _validate_m31_result((1, 2), expected_len=3)

    def test_m31_result_out_of_range_rejected(self) -> None:
        with self.assertRaises(BackendExecutionError):
            _validate_m31_result((0, M31_PRIME), expected_len=2)

    def test_m31_result_valid_values_allowed(self) -> None:
        _validate_m31_result((0, M31_PRIME - 1), expected_len=2)

    def test_m31_kernel_self_test_accepts_good_kernel(self) -> None:
        class GoodKernel:
            def __call__(self, a, b, c, length, alpha, beta, out, out_len, *_):
                n = int(length.value)
                if n != int(out_len.value):
                    return 1
                alpha_i = int(alpha.value)
                beta_i = int(beta.value)
                a_ptr = ctypes.cast(a, ctypes.POINTER(ctypes.c_uint32))
                b_ptr = ctypes.cast(b, ctypes.POINTER(ctypes.c_uint32))
                c_ptr = ctypes.cast(c, ctypes.POINTER(ctypes.c_uint32))
                out_ptr = ctypes.cast(out, ctypes.POINTER(ctypes.c_uint32))
                for i in range(n):
                    out_ptr[i] = (alpha_i * a_ptr[i] + beta_i * b_ptr[i] + c_ptr[i]) % M31_PRIME
                return 0

        _run_m31_kernel_self_test(GoodKernel(), debug_buffer_size=256)

    def test_m31_kernel_self_test_rejects_bad_math(self) -> None:
        class BadKernel:
            def __call__(self, a, _b, _c, length, _alpha, _beta, out, out_len, *_):
                n = int(length.value)
                if n != int(out_len.value):
                    return 1
                a_ptr = ctypes.cast(a, ctypes.POINTER(ctypes.c_uint32))
                out_ptr = ctypes.cast(out, ctypes.POINTER(ctypes.c_uint32))
                for i in range(n):
                    out_ptr[i] = a_ptr[i]
                return 0

        with self.assertRaises(BackendExecutionError):
            _run_m31_kernel_self_test(BadKernel(), debug_buffer_size=256)

    def test_m31_kernel_self_test_rejects_noncanonical_output(self) -> None:
        class NonCanonicalKernel:
            def __call__(self, _a, _b, _c, length, _alpha, _beta, out, out_len, *_):
                n = int(length.value)
                if n != int(out_len.value):
                    return 1
                out_ptr = ctypes.cast(out, ctypes.POINTER(ctypes.c_uint32))
                for i in range(n):
                    out_ptr[i] = M31_PRIME
                return 0

        with self.assertRaises(BackendExecutionError):
            _run_m31_kernel_self_test(NonCanonicalKernel(), debug_buffer_size=256)

    def test_kernel_abi_version_accepts_expected(self) -> None:
        class FakeLib:
            @staticmethod
            def mojo_kernel_abi_version() -> int:
                return EXPECTED_KERNEL_ABI_VERSION

        _validate_kernel_abi_version(FakeLib())

    def test_kernel_abi_version_rejects_missing_symbol(self) -> None:
        class FakeLib:
            pass

        with self.assertRaises(BackendExecutionError):
            _validate_kernel_abi_version(FakeLib())

    def test_kernel_abi_version_rejects_mismatch(self) -> None:
        class FakeLib:
            @staticmethod
            def mojo_kernel_abi_version() -> int:
                return EXPECTED_KERNEL_ABI_VERSION + 1

        with self.assertRaises(BackendExecutionError):
            _validate_kernel_abi_version(FakeLib())

    def test_kernel_sha256_normalize_accepts_valid_input(self) -> None:
        value = "AA" * 32
        self.assertEqual(_normalize_sha256_hex(value), value.lower())

    def test_kernel_sha256_normalize_rejects_invalid_length(self) -> None:
        with self.assertRaises(BackendExecutionError):
            _normalize_sha256_hex("abc")

    def test_kernel_sha256_normalize_rejects_non_hex(self) -> None:
        with self.assertRaises(BackendExecutionError):
            _normalize_sha256_hex("g" * 64)

    def test_kernel_sha256_validate_accepts_match(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "libdummy.so"
            path.write_bytes(b"kernel-bytes")
            expected = "4e72696f3eefb3b2375c36063864c2635cf3b8c85a83296a9cc30b0534c16f4d"
            _validate_library_sha256(path, expected)

    def test_kernel_sha256_validate_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "libdummy.so"
            path.write_bytes(b"kernel-bytes")
            with self.assertRaises(BackendExecutionError):
                _validate_library_sha256(path, "0" * 64)

    def test_env_flag_enabled_true_values(self) -> None:
        original = os.environ.get("MSPK_TEST_FLAG")
        try:
            for value in ("1", "true", "TRUE", "yes", "on"):
                os.environ["MSPK_TEST_FLAG"] = value
                self.assertTrue(_env_flag_enabled("MSPK_TEST_FLAG"))
        finally:
            if original is None:
                os.environ.pop("MSPK_TEST_FLAG", None)
            else:
                os.environ["MSPK_TEST_FLAG"] = original

    def test_env_flag_enabled_false_values(self) -> None:
        original = os.environ.get("MSPK_TEST_FLAG")
        try:
            for value in ("0", "false", "no", "off", "random"):
                os.environ["MSPK_TEST_FLAG"] = value
                self.assertFalse(_env_flag_enabled("MSPK_TEST_FLAG"))
            os.environ.pop("MSPK_TEST_FLAG", None)
            self.assertFalse(_env_flag_enabled("MSPK_TEST_FLAG"))
        finally:
            if original is None:
                os.environ.pop("MSPK_TEST_FLAG", None)
            else:
                os.environ["MSPK_TEST_FLAG"] = original


if __name__ == "__main__":
    unittest.main()

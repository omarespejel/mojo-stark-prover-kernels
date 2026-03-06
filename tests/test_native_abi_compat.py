from __future__ import annotations

import ctypes
import hashlib
import shutil
import unittest

from mojo_stark_prover_kernels.backends import (
    EXPECTED_KERNEL_ABI_VERSION,
    MojoSharedLibraryBackend,
)
from mojo_stark_prover_kernels.native_backend import build_native_kernel


class NativeAbiCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("cargo") is None:
            raise unittest.SkipTest("cargo is required for native ABI compatibility tests")
        cls.artifact = build_native_kernel(release=True)

    def test_native_artifact_exports_expected_abi_version(self) -> None:
        lib = ctypes.CDLL(str(self.artifact))
        fn = lib.mojo_kernel_abi_version
        fn.argtypes = []
        fn.restype = ctypes.c_uint32
        self.assertEqual(fn(), EXPECTED_KERNEL_ABI_VERSION)

    def test_python_loader_accepts_native_artifact_with_sha_pin(self) -> None:
        expected = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        backend = MojoSharedLibraryBackend(
            self.artifact,
            allow_relative_path=True,
            expected_sha256=expected,
        )
        self.assertEqual(backend.shared_lib_path, self.artifact.resolve(strict=True))


if __name__ == "__main__":
    unittest.main()

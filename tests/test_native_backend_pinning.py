from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mojo_stark_prover_kernels.native_backend import (
    NativeRustKernelBackend,
    NativeRustM31Backend,
    _sha256_file,
)


class NativeBackendPinningTests(unittest.TestCase):
    def test_sha256_file_matches_known_digest(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "artifact.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                _sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_kernel_backend_build_and_create_pins_artifact_hash(self) -> None:
        artifact = Path("/tmp/libmock_kernel.so")
        digest = "aa" * 32
        with (
            patch(
                "mojo_stark_prover_kernels.native_backend.build_native_kernel_with_sha256",
                return_value=(artifact, digest),
            ),
            patch.object(NativeRustKernelBackend, "__init__", return_value=None) as init_mock,
        ):
            _ = NativeRustKernelBackend.build_and_create(release=True, debug_buffer_size=2048)
            init_mock.assert_called_once()
            call = init_mock.call_args
            if "shared_lib_path" in call.kwargs:
                self.assertEqual(call.kwargs["shared_lib_path"], artifact)
            elif call.args:
                self.assertEqual(call.args[-1], artifact)
            self.assertTrue(init_mock.call_args.kwargs["allow_relative_path"])
            self.assertEqual(init_mock.call_args.kwargs["debug_buffer_size"], 2048)
            self.assertEqual(init_mock.call_args.kwargs["expected_sha256"], digest)

    def test_m31_backend_build_and_create_pins_artifact_hash(self) -> None:
        artifact = Path("/tmp/libmock_m31.so")
        digest = "bb" * 32
        with (
            patch(
                "mojo_stark_prover_kernels.native_backend.build_native_kernel_with_sha256",
                return_value=(artifact, digest),
            ),
            patch.object(NativeRustM31Backend, "__init__", return_value=None) as init_mock,
        ):
            _ = NativeRustM31Backend.build_and_create(release=False, debug_buffer_size=1024)
            init_mock.assert_called_once()
            call = init_mock.call_args
            if "shared_lib_path" in call.kwargs:
                self.assertEqual(call.kwargs["shared_lib_path"], artifact)
            elif call.args:
                self.assertEqual(call.args[-1], artifact)
            self.assertTrue(init_mock.call_args.kwargs["allow_relative_path"])
            self.assertEqual(init_mock.call_args.kwargs["debug_buffer_size"], 1024)
            self.assertEqual(init_mock.call_args.kwargs["expected_sha256"], digest)


if __name__ == "__main__":
    unittest.main()

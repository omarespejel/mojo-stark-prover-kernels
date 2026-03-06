from __future__ import annotations

import unittest

from mojo_stark_prover_kernels.native_backend import _with_target_cpu_native


class NativeBackendFlagsTests(unittest.TestCase):
    def test_target_cpu_native_enabled_by_default(self) -> None:
        env = {}
        out = _with_target_cpu_native(env.copy())
        self.assertIn("RUSTFLAGS", out)
        self.assertIn("-C target-cpu=native", out["RUSTFLAGS"])

    def test_target_cpu_native_can_be_disabled(self) -> None:
        env = {"MSPK_ENABLE_TARGET_CPU_NATIVE": "0"}
        out = _with_target_cpu_native(env.copy())
        self.assertNotIn("RUSTFLAGS", out)

    def test_target_cpu_native_not_duplicated(self) -> None:
        env = {"RUSTFLAGS": "-C target-cpu=native"}
        out = _with_target_cpu_native(env.copy())
        self.assertEqual(out["RUSTFLAGS"], "-C target-cpu=native")

    def test_target_cpu_native_appends_existing_rustflags(self) -> None:
        env = {"RUSTFLAGS": "-C opt-level=3"}
        out = _with_target_cpu_native(env.copy())
        self.assertIn("-C opt-level=3", out["RUSTFLAGS"])
        self.assertIn("-C target-cpu=native", out["RUSTFLAGS"])


if __name__ == "__main__":
    unittest.main()

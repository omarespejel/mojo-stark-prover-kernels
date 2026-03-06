from __future__ import annotations

import ctypes
import shutil
import unittest

from mojo_stark_prover_kernels.backends import ReferenceKernelBackend
from mojo_stark_prover_kernels.contracts import CommitLayerRequest
from mojo_stark_prover_kernels.debug import DebugTracer
from mojo_stark_prover_kernels.differential import compare_case, run_randomized_suite
from mojo_stark_prover_kernels.native_backend import NativeRustKernelBackend


class NativeRustKernelBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("cargo") is None:
            raise unittest.SkipTest("cargo is required for native backend tests")
        cls.backend = NativeRustKernelBackend.build_and_create(release=True)

    def test_single_case_parity(self) -> None:
        req = CommitLayerRequest.from_sequences(
            log_size=3,
            columns=[
                [11, 12, 13, 14, 15, 16, 17, 18],
                [101, 102, 103, 104, 105, 106, 107, 108],
                [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            ],
            prev_layer_hashes=[bytes([i]) * 32 for i in range(16)],
            debug_level=1,
        )
        result = compare_case(
            case_id=0,
            request=req,
            candidate_backend=self.backend,
            oracle_backend=ReferenceKernelBackend(),
        )
        self.assertTrue(result.match)

    def test_randomized_differential_suite(self) -> None:
        suite = run_randomized_suite(
            candidate_backend=self.backend,
            oracle_backend=ReferenceKernelBackend(),
            seed=20260305,
            n_cases=40,
            max_log_size=7,
            max_columns=8,
            debug_level=1,
            fail_fast=False,
        )
        self.assertEqual(suite.failed_cases, 0)
        self.assertEqual(suite.passed_cases, suite.total_cases)

    def test_native_debug_event_emitted(self) -> None:
        req = CommitLayerRequest.from_sequences(
            log_size=1,
            columns=[[1, 2], [3, 4]],
            debug_level=1,
        )
        tracer = DebugTracer(level=2)
        _ = self.backend.commit_layer(req, tracer=tracer)
        events = tracer.snapshot()
        self.assertTrue(any("mojo backend debug: ok rows=2 cols=2 mode=serial" in e for e in events))

    def test_raw_abi_rejects_total_cells_above_contract_limit(self) -> None:
        log_size = 6
        n_rows = 1 << log_size
        n_columns = 4000
        total_cells = n_rows * n_columns

        columns = (ctypes.c_uint32 * total_cells)(*([7] * total_cells))
        out = (ctypes.c_uint8 * (n_rows * 32))()
        debug_buf = (ctypes.c_char * 512)()

        rc = self.backend._fn(
            ctypes.c_uint32(log_size),
            None,
            ctypes.c_uint32(0),
            ctypes.cast(columns, ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n_columns),
            ctypes.c_uint32(n_rows),
            ctypes.cast(out, ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_uint32(len(out)),
            ctypes.c_uint32(1),
            ctypes.cast(debug_buf, ctypes.POINTER(ctypes.c_char)),
            ctypes.c_uint32(len(debug_buf)),
        )

        debug_text = bytes(debug_buf).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        self.assertNotEqual(rc, 0)
        self.assertIn("total cells exceed configured max", debug_text)

    def test_raw_abi_rejects_overlapping_input_output_buffers(self) -> None:
        # Shared backing storage intentionally aliases columns and output buffers.
        storage = (ctypes.c_uint32 * 32)(*range(32))
        debug_buf = (ctypes.c_char * 512)()

        rc = self.backend._fn(
            ctypes.c_uint32(1),
            None,
            ctypes.c_uint32(0),
            ctypes.cast(storage, ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(2),
            ctypes.c_uint32(2),
            ctypes.cast(storage, ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_uint32(64),
            ctypes.c_uint32(1),
            ctypes.cast(debug_buf, ctypes.POINTER(ctypes.c_char)),
            ctypes.c_uint32(len(debug_buf)),
        )

        debug_text = bytes(debug_buf).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        self.assertNotEqual(rc, 0)
        self.assertIn("out_hashes overlaps input buffers", debug_text)

    def test_raw_abi_rejects_log_size_above_contract_limit(self) -> None:
        columns = (ctypes.c_uint32 * 1)(42)
        out = (ctypes.c_uint8 * 32)()
        debug_buf = (ctypes.c_char * 512)()

        rc = self.backend._fn(
            ctypes.c_uint32(21),
            None,
            ctypes.c_uint32(0),
            ctypes.cast(columns, ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(1),
            ctypes.c_uint32(1),
            ctypes.cast(out, ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_uint32(len(out)),
            ctypes.c_uint32(1),
            ctypes.cast(debug_buf, ctypes.POINTER(ctypes.c_char)),
            ctypes.c_uint32(len(debug_buf)),
        )

        debug_text = bytes(debug_buf).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        self.assertNotEqual(rc, 0)
        self.assertIn("log_size exceeds configured max", debug_text)

    def test_raw_abi_rejects_misaligned_columns_pointer(self) -> None:
        raw = (ctypes.c_uint8 * 64)()
        misaligned_columns = ctypes.cast(
            ctypes.byref(raw, 1),
            ctypes.POINTER(ctypes.c_uint32),
        )
        out = (ctypes.c_uint8 * 32)()
        debug_buf = (ctypes.c_char * 512)()

        rc = self.backend._fn(
            ctypes.c_uint32(0),
            None,
            ctypes.c_uint32(0),
            misaligned_columns,
            ctypes.c_uint32(1),
            ctypes.c_uint32(1),
            ctypes.cast(out, ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_uint32(len(out)),
            ctypes.c_uint32(1),
            ctypes.cast(debug_buf, ctypes.POINTER(ctypes.c_char)),
            ctypes.c_uint32(len(debug_buf)),
        )

        debug_text = bytes(debug_buf).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        self.assertNotEqual(rc, 0)
        self.assertIn("columns pointer is not u32-aligned", debug_text)

    def test_raw_abi_rejects_debug_buffer_len_above_cap(self) -> None:
        columns = (ctypes.c_uint32 * 1)(1)
        out = (ctypes.c_uint8 * 32)()

        rc = self.backend._fn(
            ctypes.c_uint32(0),
            None,
            ctypes.c_uint32(0),
            ctypes.cast(columns, ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(1),
            ctypes.c_uint32(1),
            ctypes.cast(out, ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_uint32(len(out)),
            ctypes.c_uint32(1),
            None,
            ctypes.c_uint32(1_000_001),
        )

        self.assertNotEqual(rc, 0)

    def test_raw_abi_checks_debug_buffer_len_before_other_errors(self) -> None:
        tiny_debug = (ctypes.c_char * 1)()

        rc = self.backend._fn(
            ctypes.c_uint32(0),
            None,
            ctypes.c_uint32(0),
            None,
            ctypes.c_uint32(0),
            ctypes.c_uint32(0),
            None,
            ctypes.c_uint32(0),
            ctypes.c_uint32(1),
            ctypes.cast(tiny_debug, ctypes.POINTER(ctypes.c_char)),
            ctypes.c_uint32(1_000_001),
        )

        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()

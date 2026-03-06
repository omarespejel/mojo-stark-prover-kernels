from __future__ import annotations

import ctypes
import random
import shutil
import unittest

from mojo_stark_prover_kernels.debug import DebugTracer
from mojo_stark_prover_kernels.m31_axpy import M31AxpyRequest, M31_PRIME, m31_axpy_reference
from mojo_stark_prover_kernels.native_backend import NativeRustM31Backend


class NativeRustM31BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("cargo") is None:
            raise unittest.SkipTest("cargo is required for native backend tests")
        cls.backend = NativeRustM31Backend.build_and_create(release=True)

    def test_single_case_parity(self) -> None:
        req = M31AxpyRequest.from_sequences(
            a=[11, 12, 13, 14],
            b=[101, 102, 103, 104],
            c=[1001, 1002, 1003, 1004],
            alpha=123,
            beta=321,
            debug_level=1,
        )
        expected = m31_axpy_reference(req)
        actual = self.backend.m31_axpy(req)
        self.assertEqual(actual, expected)
        self.assertTrue(all(isinstance(v, int) for v in actual))

    def test_randomized_parity(self) -> None:
        rnd = random.Random(20260305)
        for _ in range(30):
            size = rnd.randint(1, 1024)
            req = M31AxpyRequest.from_sequences(
                a=[rnd.randrange(0, M31_PRIME) for _ in range(size)],
                b=[rnd.randrange(0, M31_PRIME) for _ in range(size)],
                c=[rnd.randrange(0, M31_PRIME) for _ in range(size)],
                alpha=rnd.randrange(0, M31_PRIME),
                beta=rnd.randrange(0, M31_PRIME),
                debug_level=0,
            )
            self.assertEqual(self.backend.m31_axpy(req), m31_axpy_reference(req))

    def test_debug_event_emitted(self) -> None:
        req = M31AxpyRequest.from_sequences(
            a=[1, 2],
            b=[3, 4],
            c=[5, 6],
            alpha=7,
            beta=11,
            debug_level=1,
        )
        tracer = DebugTracer(level=2)
        _ = self.backend.m31_axpy(req, tracer=tracer)
        events = tracer.snapshot()
        self.assertTrue(any("mojo m31 debug: ok len=2 mode=serial" in e for e in events))

    def test_raw_abi_rejects_out_overlap(self) -> None:
        n = 4
        a = (ctypes.c_uint32 * n)(1, 2, 3, 4)
        b = (ctypes.c_uint32 * n)(5, 6, 7, 8)
        c = (ctypes.c_uint32 * n)(9, 10, 11, 12)
        debug_buf = (ctypes.c_char * 256)()

        rc = self.backend._m31_fn(
            ctypes.cast(a, ctypes.POINTER(ctypes.c_uint32)),
            ctypes.cast(b, ctypes.POINTER(ctypes.c_uint32)),
            ctypes.cast(c, ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n),
            ctypes.c_uint32(3),
            ctypes.c_uint32(5),
            ctypes.cast(a, ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n),
            ctypes.c_uint32(1),
            ctypes.cast(debug_buf, ctypes.POINTER(ctypes.c_char)),
            ctypes.c_uint32(len(debug_buf)),
        )

        debug_text = bytes(debug_buf).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        self.assertNotEqual(rc, 0)
        self.assertIn("output overlaps input", debug_text)

    def test_backend_rejects_invalid_values(self) -> None:
        req = M31AxpyRequest.from_sequences(
            a=[1],
            b=[2],
            c=[3],
            alpha=7,
            beta=11,
            debug_level=0,
        )
        # bypass dataclass immutability by constructing invalid payload through low-level ABI
        n = 1
        a = (ctypes.c_uint32 * n)(M31_PRIME)
        b = (ctypes.c_uint32 * n)(2)
        c = (ctypes.c_uint32 * n)(3)
        out = (ctypes.c_uint32 * n)()
        debug_buf = (ctypes.c_char * 256)()
        rc = self.backend._m31_fn(
            ctypes.cast(a, ctypes.POINTER(ctypes.c_uint32)),
            ctypes.cast(b, ctypes.POINTER(ctypes.c_uint32)),
            ctypes.cast(c, ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n),
            ctypes.c_uint32(req.alpha),
            ctypes.c_uint32(req.beta),
            ctypes.cast(out, ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n),
            ctypes.c_uint32(1),
            ctypes.cast(debug_buf, ctypes.POINTER(ctypes.c_char)),
            ctypes.c_uint32(len(debug_buf)),
        )
        debug_text = bytes(debug_buf).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        self.assertNotEqual(rc, 0)
        self.assertIn("input value outside m31 field", debug_text)

if __name__ == "__main__":
    unittest.main()

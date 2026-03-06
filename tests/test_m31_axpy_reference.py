from __future__ import annotations

import unittest

from mojo_stark_prover_kernels.m31_axpy import M31AxpyRequest, M31_PRIME, m31_axpy_reference


class M31AxpyReferenceTests(unittest.TestCase):
    def test_known_vector(self) -> None:
        req = M31AxpyRequest.from_sequences(
            a=[1, 2, 3],
            b=[10, 20, 30],
            c=[100, 200, 300],
            alpha=7,
            beta=9,
            debug_level=0,
        )
        out = m31_axpy_reference(req)
        self.assertEqual(
            out,
            (
                (7 * 1 + 9 * 10 + 100) % M31_PRIME,
                (7 * 2 + 9 * 20 + 200) % M31_PRIME,
                (7 * 3 + 9 * 30 + 300) % M31_PRIME,
            ),
        )

    def test_deterministic(self) -> None:
        req = M31AxpyRequest.from_sequences(
            a=[123, 456],
            b=[789, 321],
            c=[654, 987],
            alpha=17,
            beta=19,
        )
        lhs = m31_axpy_reference(req)
        rhs = m31_axpy_reference(req)
        self.assertEqual(lhs, rhs)


if __name__ == "__main__":
    unittest.main()

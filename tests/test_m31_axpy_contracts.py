from __future__ import annotations

import unittest

from mojo_stark_prover_kernels.m31_axpy import M31AxpyRequest, M31_PRIME


class M31AxpyRequestValidationTests(unittest.TestCase):
    def test_valid_request_passes(self) -> None:
        req = M31AxpyRequest.from_sequences(
            a=[1, 2, 3],
            b=[4, 5, 6],
            c=[7, 8, 9],
            alpha=11,
            beta=13,
        )
        req.validate()
        self.assertEqual(req.length, 3)

    def test_mismatched_lengths_rejected(self) -> None:
        with self.assertRaises(ValueError):
            M31AxpyRequest.from_sequences(
                a=[1, 2],
                b=[3],
                c=[4, 5],
                alpha=1,
                beta=1,
            )

    def test_empty_input_rejected(self) -> None:
        with self.assertRaises(ValueError):
            M31AxpyRequest.from_sequences(
                a=[],
                b=[],
                c=[],
                alpha=1,
                beta=1,
            )

    def test_value_out_of_range_rejected(self) -> None:
        with self.assertRaises(ValueError):
            M31AxpyRequest.from_sequences(
                a=[M31_PRIME],
                b=[1],
                c=[1],
                alpha=1,
                beta=1,
            )

    def test_scalar_out_of_range_rejected(self) -> None:
        with self.assertRaises(ValueError):
            M31AxpyRequest.from_sequences(
                a=[1],
                b=[1],
                c=[1],
                alpha=M31_PRIME,
                beta=1,
            )


if __name__ == "__main__":
    unittest.main()

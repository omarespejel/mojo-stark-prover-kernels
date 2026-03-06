from __future__ import annotations

import unittest

from mojo_stark_prover_kernels.contracts import CommitLayerRequest, MAX_LOG_SIZE


def _valid_columns(log_size: int = 2, n_columns: int = 2) -> list[list[int]]:
    n_rows = 1 << log_size
    return [[col * 100 + row for row in range(n_rows)] for col in range(n_columns)]


class CommitLayerRequestValidationTests(unittest.TestCase):
    def test_valid_request_passes(self) -> None:
        req = CommitLayerRequest.from_sequences(log_size=2, columns=_valid_columns())
        req.validate()
        self.assertEqual(req.n_rows, 4)

    def test_invalid_log_size_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CommitLayerRequest.from_sequences(
                log_size=MAX_LOG_SIZE + 1, columns=_valid_columns()
            )

    def test_invalid_debug_level_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CommitLayerRequest.from_sequences(
                log_size=2, columns=_valid_columns(), debug_level=999
            )

    def test_column_length_mismatch_rejected(self) -> None:
        bad_cols = [[1, 2, 3], [10, 20, 30, 40]]
        with self.assertRaises(ValueError):
            CommitLayerRequest.from_sequences(log_size=2, columns=bad_cols)

    def test_prev_hash_count_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CommitLayerRequest.from_sequences(
                log_size=2,
                columns=_valid_columns(),
                prev_layer_hashes=[b"\x00" * 32],
            )

    def test_prev_hash_size_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CommitLayerRequest.from_sequences(
                log_size=1,
                columns=_valid_columns(log_size=1),
                prev_layer_hashes=[b"\x00" * 31] * 4,
            )

    def test_value_out_of_u32_range_rejected(self) -> None:
        bad_cols = _valid_columns()
        bad_cols[0][0] = -1
        with self.assertRaises(ValueError):
            CommitLayerRequest.from_sequences(log_size=2, columns=bad_cols)

    def test_total_cells_limit_rejected(self) -> None:
        log_size = 16
        n_rows = 1 << log_size
        row = list(range(n_rows))
        with self.assertRaises(ValueError):
            CommitLayerRequest.from_sequences(
                log_size=log_size,
                columns=[row, row, row, row],
            )


if __name__ == "__main__":
    unittest.main()

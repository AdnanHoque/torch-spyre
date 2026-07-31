#!/usr/bin/env python3
"""Host-only tests for the activation-stationary decode resource model."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("design_a_model", HERE / "design_a_model.py")
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)


class DesignAModelTest(unittest.TestCase):
    def test_physical_m64_k1024_exactly_fills_xrf(self) -> None:
        report = MODEL.build_report(logical_m=1, physical_m=64, k=4096, n=4096)
        self.assertEqual(report.xrf_bytes_per_corelet, 128 * 1024)
        self.assertEqual(report.xrf_k_tile, 1024)
        self.assertEqual(report.stationary_a_tile_bytes_per_corelet, 128 * 1024)
        self.assertTrue(report.stationary_a_exact_xrf_fit)

    def test_m8_k1024_is_not_an_exact_xrf_fill(self) -> None:
        # An unpadded M8 x K1024 FP16 tile is 16 KiB, only one eighth of XRF.
        unpadded_bytes = 8 * 1024 * MODEL.FP16_BYTES
        self.assertEqual(unpadded_bytes, 16 * 1024)
        self.assertEqual(MODEL.XRF_BYTES_PER_CORELET // unpadded_bytes, 8)

    def test_design_a_does_not_reduce_unique_weight_hbm_bytes(self) -> None:
        report = MODEL.build_report(logical_m=1, physical_m=64, k=4096, n=4096)
        self.assertEqual(report.weight_bytes, 32 * 1024 * 1024)
        self.assertEqual(
            report.incumbent_unique_weight_hbm_bytes,
            report.candidate_unique_weight_hbm_bytes,
        )
        self.assertEqual(report.unique_weight_hbm_reduction_bytes, 0)

    def test_machine_balance_and_intensity(self) -> None:
        report = MODEL.build_report(logical_m=8, physical_m=64, k=4096, n=4096)
        self.assertEqual(report.machine_balance_flop_per_byte, 256)
        self.assertEqual(
            report.useful_weight_arithmetic_intensity_flop_per_byte,
            8,
        )
        self.assertEqual(
            report.physical_weight_arithmetic_intensity_flop_per_byte,
            64,
        )


if __name__ == "__main__":
    unittest.main()

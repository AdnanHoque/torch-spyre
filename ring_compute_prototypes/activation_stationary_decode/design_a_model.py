#!/usr/bin/env python3
"""First-principles resource model for activation-stationary decode matmul.

The candidate computes::

    C = A @ B
    C.T = B.T @ A.T

and sends the transposed problem through the stock weight-stationary PT
schedule.  In that transposed problem ``A.T`` occupies the XRF role and
``B.T`` occupies the West-to-East stream role.

This file is analytical only.  It does not claim that the compiler realizes
the dataflow or that the device reaches any modeled bound.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


FP16_BYTES = 2
PT_ROWS = 8
PT_COLUMNS = 8
PT_SIMD = 8
PT_MACS_PER_CORELET = PT_ROWS * PT_COLUMNS * PT_SIMD
CORELETS_PER_CORE = 2
CORES = 32
XRF_REGISTERS_PER_PT = 128
XRF_REGISTER_BYTES = 16
XRF_BYTES_PER_CORELET = (
    PT_ROWS * PT_COLUMNS * XRF_REGISTERS_PER_PT * XRF_REGISTER_BYTES
)
HMI_BYTES_PER_CYCLE = 256
CHIP_FLOPS_PER_CYCLE = (
    CORES * CORELETS_PER_CORE * PT_MACS_PER_CORELET * 2
)


@dataclass(frozen=True)
class DesignAReport:
    logical_m: int
    physical_m: int
    k: int
    n: int
    cores: int
    xrf_bytes_per_corelet: int
    xrf_k_tile: int
    stationary_a_tile_bytes_per_corelet: int
    stationary_a_exact_xrf_fit: bool
    logical_a_bytes: int
    physical_a_bytes: int
    weight_bytes: int
    output_bytes: int
    useful_flops: int
    physical_flops: int
    useful_weight_arithmetic_intensity_flop_per_byte: float
    physical_weight_arithmetic_intensity_flop_per_byte: float
    machine_balance_flop_per_byte: float
    weight_hmi_floor_cycles: float
    incumbent_unique_weight_hbm_bytes: int
    candidate_unique_weight_hbm_bytes: int
    unique_weight_hbm_reduction_bytes: int


def build_report(
    *,
    logical_m: int,
    physical_m: int,
    k: int,
    n: int,
    cores: int = CORES,
) -> DesignAReport:
    for name, value in {
        "logical_m": logical_m,
        "physical_m": physical_m,
        "k": k,
        "n": n,
        "cores": cores,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if logical_m > physical_m:
        raise ValueError("logical_m cannot exceed physical_m")
    if physical_m % (PT_COLUMNS * PT_SIMD):
        raise ValueError("physical_m must map exactly across the 64 PT column/SIMD lanes")

    xrf_k_tile, remainder = divmod(XRF_BYTES_PER_CORELET, physical_m * FP16_BYTES)
    if remainder:
        raise ValueError("physical M does not divide the FP16 XRF capacity")

    stationary_a_tile_bytes = physical_m * xrf_k_tile * FP16_BYTES
    weight_bytes = k * n * FP16_BYTES
    useful_flops = 2 * logical_m * k * n
    physical_flops = 2 * physical_m * k * n

    return DesignAReport(
        logical_m=logical_m,
        physical_m=physical_m,
        k=k,
        n=n,
        cores=cores,
        xrf_bytes_per_corelet=XRF_BYTES_PER_CORELET,
        xrf_k_tile=xrf_k_tile,
        stationary_a_tile_bytes_per_corelet=stationary_a_tile_bytes,
        stationary_a_exact_xrf_fit=stationary_a_tile_bytes
        == XRF_BYTES_PER_CORELET,
        logical_a_bytes=logical_m * k * FP16_BYTES,
        physical_a_bytes=physical_m * k * FP16_BYTES,
        weight_bytes=weight_bytes,
        output_bytes=logical_m * n * FP16_BYTES,
        useful_flops=useful_flops,
        physical_flops=physical_flops,
        useful_weight_arithmetic_intensity_flop_per_byte=useful_flops
        / weight_bytes,
        physical_weight_arithmetic_intensity_flop_per_byte=physical_flops
        / weight_bytes,
        machine_balance_flop_per_byte=CHIP_FLOPS_PER_CYCLE
        / HMI_BYTES_PER_CYCLE,
        weight_hmi_floor_cycles=weight_bytes / HMI_BYTES_PER_CYCLE,
        # The incumbent HMI/GTR schedule already reads every unique weight
        # byte once.  Design A changes the in-core path, not this source cut.
        incumbent_unique_weight_hbm_bytes=weight_bytes,
        candidate_unique_weight_hbm_bytes=weight_bytes,
        unique_weight_hbm_reduction_bytes=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-m", type=int, default=1)
    parser.add_argument("--physical-m", type=int, default=64)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--cores", type=int, default=CORES)
    args = parser.parse_args()
    print(
        json.dumps(
            asdict(
                build_report(
                    logical_m=args.logical_m,
                    physical_m=args.physical_m,
                    k=args.k,
                    n=args.n,
                    cores=args.cores,
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a production-shaped 4x8-to-32x1 LX SHUFFLE bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from sympy import Integer, Mod, Symbol, floor

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.bundle import generate_bundle
from torch_spyre._inductor.lx_relayout import (
    LXCollectiveKind,
    LXRelayoutPlan,
    _destination_size_ratio,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.spyre_kernel import (
    _materialize_explicit_lx_shuffle,
    simplify_op_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    producer_map = {
        str(core): {"0": core // 8, "1": core % 8} for core in range(32)
    }
    consumer_map = {str(core): {"0": core, "1": 0} for core in range(32)}
    ratio = _destination_size_ratio(
        producer_map,
        {"0": 4, "1": 8},
        consumer_map,
        {"0": 32, "1": 1},
    )
    assert ratio == 1

    m = Symbol("m")
    n = Symbol("n")
    plan = LXRelayoutPlan(
        source_name="granite_mlp_intermediate",
        consumer_name="pointwise",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={"0": 4, "1": 8},
        destination_device_dim_splits={"0": 32, "1": 1},
        collective_kind=LXCollectiveKind.ALL_TO_ALL,
        destination_size_ratio=ratio,
        destination_lx_address=0x44000,
    )
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 200, 64],
        device_coordinates=[m, floor(n / 64), Mod(n, 64)],
        allocation={"lx": 0x24000},
        name=plan.source_name,
    )
    consumer_spec = OpSpec(
        op="silu",
        is_reduction=False,
        iteration_space={m: (Integer(512), 32), n: (Integer(12800), 1)},
        args=[source_arg],
        op_info={},
    )
    materialized = _materialize_explicit_lx_shuffle(
        source_arg, consumer_spec, plan
    )
    assert materialized is not None
    shuffle_spec, _ = materialized
    simplify_op_spec(shuffle_spec)

    args.output.mkdir(parents=True, exist_ok=True)
    generate_bundle(
        "granite_mlp_alltoall_lx_relayout",
        str(args.output),
        [shuffle_spec],
        use_symbols=False,
    )
    print("shape: [512, 12800]")
    print("producer: 4x8")
    print("consumer: 32x1")
    print("fanin: 8")
    print("fanout: 8")
    print("transfers: 256")
    print(f"bundle: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Emit standalone restickify data-op SDSCs for Deeptools contract probing."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from sympy import Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.codegen.restickify_lx_dataop import (
    SUPPORTED_RESTICKIFY_DATA_OPS,
    combine_dataop_sdscs,
    generate_native_ptlx_consumer_endpoint_adapter_tile_sdsc,
    generate_native_ptlx_validgap_endpoint_tile_bridge_sdsc,
    generate_restickify_dataop_sdsc_from_spec,
    generate_streaming_ptlx_full_bridge_sdsc,
    generate_streaming_ptlx_tile_bridge_sdsc,
    generate_validgap_ptlx_consumer_endpoint_adapter_tile_sdsc,
)
from torch_spyre._inductor.codegen.restickify_ptlx_streaming import (
    default_core_mapping,
    generate_streaming_ptlx_artifact,
    plan_streaming_ptlx_tiles,
)
from torch_spyre._inductor.codegen.superdsc import SDSCArgs, SDSCSpec

_RESTICKIFY_LX_DATAOP_RESTICKIFY_OP_ENV = (
    "SPYRE_RESTICKIFY_LX_DATAOP_RESTICKIFY_OP"
)
_SUPPORTED_TWO_STEP_RESTICKIFY_OPS = frozenset(
    {"ReStickifyOpLx", "ReStickifyOpWithPTLx"}
)


def _two_step_restickify_op_name() -> str:
    op_name = os.environ.get(
        _RESTICKIFY_LX_DATAOP_RESTICKIFY_OP_ENV,
        "ReStickifyOpLx",
    )
    if op_name not in _SUPPORTED_TWO_STEP_RESTICKIFY_OPS:
        raise ValueError(
            f"{_RESTICKIFY_LX_DATAOP_RESTICKIFY_OP_ENV}={op_name!r} is not "
            "supported for the composed LX bridge; expected one of "
            f"{sorted(_SUPPORTED_TWO_STEP_RESTICKIFY_OPS)}"
        )
    if op_name not in SUPPORTED_RESTICKIFY_DATA_OPS:
        raise ValueError(
            f"{op_name!r} is not enabled in SUPPORTED_RESTICKIFY_DATA_OPS"
        )
    return op_name


def _core_mapping(dims: list[Symbol], split_dim: Symbol, num_cores: int):
    return {
        str(core): {
            str(dim): core if dim == split_dim else 0
            for dim in dims
        }
        for core in range(num_cores)
    }


def _parse_work_slices(values: list[str] | None, default: dict[str, int]) -> dict[str, int]:
    result = dict(default)
    for item in values or []:
        if ":" not in item:
            raise ValueError(f"expected DIM:SPLIT work-slice item, got {item!r}")
        dim, split = item.split(":", 1)
        result[dim] = int(split)
    return result


def _synthetic_spec(
    size: int,
    num_cores: int,
    output_split_dim: Symbol,
    output_stick_dim: Symbol,
    *,
    input_stick_dim: Symbol | None = None,
    input_start_address: int = 0,
    output_start_address: int = 1024 * 1024,
    input_layout_order: list[Symbol] | None = None,
    output_layout_order: list[Symbol] | None = None,
    input_strides: dict[Symbol, int] | None = None,
    output_strides: dict[Symbol, int] | None = None,
) -> SDSCSpec:
    # Data-op import in Deeptools expects canonical DSC dimension labels.
    # `mb_` acts as logical d0 and `out_` acts as logical d1 for this probe.
    d0 = Symbol("mb_")
    d1 = Symbol("out_")
    input_stick_dim = input_stick_dim or d1
    input_layout_order = input_layout_order or [d0, d1]
    output_layout_order = output_layout_order or [d1, d0]
    input_strides = input_strides or {d0: size, d1: 1}
    output_strides = output_strides or {d0: 1, d1: size}
    data_format = DataFormats.SEN169_FP16
    input_arg = SDSCArgs(
        layout="INPUT",
        data_format=data_format,
        scales={d0: 1, d1: 1},
        strides=input_strides,
        offsets={},
        max_dim_sizes={d0: -1, d1: -1},
        allocation={"lx": 0},
        start_address=input_start_address,
        backGap={},
    )
    output_arg = SDSCArgs(
        layout="OUTPUT",
        data_format=data_format,
        scales={d0: 1, d1: 1},
        strides=output_strides,
        offsets={},
        max_dim_sizes={d0: -1, d1: -1},
        allocation={"lx": 0},
        start_address=output_start_address,
        backGap={},
    )
    dims = [d0, d1]
    work_slices = {d0: 1, d1: 1}
    work_slices[output_split_dim] = num_cores
    return SDSCSpec(
        opfunc="ReStickifyOpHBM",
        execution_unit="sfp",
        data_format=data_format,
        num_inputs=1,
        iteration_space={d0: size, d1: size},
        num_cores=num_cores,
        work_slices=work_slices,
        core_id_to_work_slice={},
        core_id_to_work_slice_override=_core_mapping(dims, output_split_dim, num_cores),
        padding={},
        layouts={
            "INPUT": {
                "dim_order": input_layout_order,
                "stick_dim_order": input_stick_dim,
                "stick_size": 64,
            },
            "OUTPUT": {
                "dim_order": output_layout_order,
                "stick_dim_order": output_stick_dim,
                "stick_size": 64,
            },
        },
        args=[input_arg, output_arg],
        constants={},
        coordinate_masking={},
    )


def _two_step_lx_restickify_payload(
    mode: str,
    size: int,
    num_cores: int,
) -> dict:
    d0 = Symbol("mb_")
    d1 = Symbol("out_")
    direction = os.environ.get(
        "SPYRE_RESTICKIFY_LX_DATAOP_DIRECTION",
        "kernel-to-output",
    )
    single_op = os.environ.get("SPYRE_RESTICKIFY_LX_DATAOP_SINGLE_OP", "0") == "1"
    restickify_op_name = _two_step_restickify_op_name()
    if direction == "restickify-stcdp-restickify":
        producer_splits = {d0: num_cores, d1: 1}
        producer_mapping = _core_mapping([d0, d1], d0, num_cores)
        source_view_splits = {d0: num_cores, d1: 1}
        source_view_mapping = _core_mapping([d0, d1], d0, num_cores)
        transferred_splits = {d0: 1, d1: num_cores}
        transferred_mapping = _core_mapping([d0, d1], d1, num_cores)
        consumer_splits = {d0: num_cores, d1: 1}
        consumer_mapping = _core_mapping([d0, d1], d0, num_cores)

        source_view_start = 1024 * 1024
        transferred_start = 1280 * 1024
        output_start = 1536 * 1024
        physical_strides = {d0: size, d1: 1}
        transpose_source_strides = {d0: 1, d1: size}

        source_view_spec = _synthetic_spec(
            size,
            num_cores,
            output_split_dim=d0,
            output_stick_dim=d0,
            input_stick_dim=d1,
            input_start_address=0,
            output_start_address=source_view_start,
            input_layout_order=[d0, d1],
            output_layout_order=[d1, d0],
            input_strides=physical_strides,
            output_strides=transpose_source_strides,
        )
        source_view_payload = generate_restickify_dataop_sdsc_from_spec(
            0,
            source_view_spec,
            op_name=restickify_op_name,
            input_work_slices=producer_splits,
            input_core_to_work_slice=producer_mapping,
            output_work_slices=source_view_splits,
            output_core_to_work_slice=source_view_mapping,
        )

        transfer_spec = _synthetic_spec(
            size,
            num_cores,
            output_split_dim=d1,
            output_stick_dim=d0,
            input_stick_dim=d0,
            input_start_address=source_view_start,
            output_start_address=transferred_start,
            input_layout_order=[d1, d0],
            output_layout_order=[d1, d0],
            input_strides=transpose_source_strides,
            output_strides=transpose_source_strides,
        )
        transfer_payload = generate_restickify_dataop_sdsc_from_spec(
            1,
            transfer_spec,
            op_name="STCDPOpLx",
            input_work_slices=source_view_splits,
            input_core_to_work_slice=source_view_mapping,
            output_work_slices=transferred_splits,
            output_core_to_work_slice=transferred_mapping,
        )

        destination_spec = _synthetic_spec(
            size,
            num_cores,
            output_split_dim=d0,
            output_stick_dim=d1,
            input_stick_dim=d0,
            input_start_address=transferred_start,
            output_start_address=output_start,
            input_layout_order=[d1, d0],
            output_layout_order=[d0, d1],
            input_strides=transpose_source_strides,
            output_strides=physical_strides,
        )
        destination_payload = generate_restickify_dataop_sdsc_from_spec(
            2,
            destination_spec,
            op_name=restickify_op_name,
            input_work_slices=transferred_splits,
            input_core_to_work_slice=transferred_mapping,
            output_work_slices=consumer_splits,
            output_core_to_work_slice=consumer_mapping,
        )

        return combine_dataop_sdscs(
            f"0_{restickify_op_name}Stcdp{restickify_op_name}_{mode}_dataop",
            [source_view_payload, transfer_payload, destination_payload],
        )

    if direction == "stcdp-then-restickify":
        producer_splits = {d0: num_cores, d1: 1}
        producer_mapping = _core_mapping([d0, d1], d0, num_cores)
        intermediate_splits = {d0: 1, d1: num_cores}
        intermediate_mapping = _core_mapping([d0, d1], d1, num_cores)
        consumer_splits = {d0: num_cores, d1: 1}
        consumer_mapping = _core_mapping([d0, d1], d0, num_cores)

        intermediate_start = 1024 * 1024
        output_start = 1536 * 1024
        physical_strides = {d0: size, d1: 1}
        transpose_source_strides = {d0: 1, d1: size}

        stcdp_spec = _synthetic_spec(
            size,
            num_cores,
            output_split_dim=d1,
            output_stick_dim=d0,
            input_stick_dim=d1,
            input_start_address=0,
            output_start_address=intermediate_start,
            input_layout_order=[d0, d1],
            output_layout_order=[d1, d0],
            input_strides=physical_strides,
            output_strides=transpose_source_strides,
        )
        stcdp_payload = generate_restickify_dataop_sdsc_from_spec(
            0,
            stcdp_spec,
            op_name="STCDPOpLx",
            input_work_slices=producer_splits,
            input_core_to_work_slice=producer_mapping,
            output_work_slices=intermediate_splits,
            output_core_to_work_slice=intermediate_mapping,
        )

        restickify_spec = _synthetic_spec(
            size,
            num_cores,
            output_split_dim=d0,
            output_stick_dim=d1,
            input_stick_dim=d0,
            input_start_address=intermediate_start,
            output_start_address=output_start,
            input_layout_order=[d1, d0],
            output_layout_order=[d0, d1],
            input_strides=transpose_source_strides,
            output_strides=physical_strides,
        )
        restickify_payload = generate_restickify_dataop_sdsc_from_spec(
            1,
            restickify_spec,
            op_name=restickify_op_name,
            input_work_slices=intermediate_splits,
            input_core_to_work_slice=intermediate_mapping,
            output_work_slices=consumer_splits,
            output_core_to_work_slice=consumer_mapping,
        )

        return combine_dataop_sdscs(
            f"0_StcdpThen{restickify_op_name}_{mode}_dataop",
            [stcdp_payload, restickify_payload],
        )

    if direction == "output-to-kernel":
        input_splits = {d0: num_cores, d1: 1}
        input_mapping = _core_mapping([d0, d1], d0, num_cores)

        # The real HBM restickify in the computed transpose fixture converts
        # OUTPUT ([out, mb], stick mb) to KERNEL ([mb, out], stick out).  Keep
        # the ReStickifyOpLx output split off the input stick dimension, then
        # use a same-stick STCDP stage to restore the consumer's split.
        intermediate_splits = {d0: 1, d1: num_cores}
        intermediate_mapping = _core_mapping([d0, d1], d1, num_cores)

        final_split_dim = d0 if mode == "baseline" else d1
        final_splits = {d0: 1, d1: 1}
        final_splits[final_split_dim] = num_cores
        final_mapping = _core_mapping([d0, d1], final_split_dim, num_cores)

        intermediate_start = 1024 * 1024
        output_start = 1536 * 1024
        restickify_output_start = output_start if single_op else intermediate_start
        restickify_output_splits = final_splits if single_op else intermediate_splits
        restickify_output_mapping = final_mapping if single_op else intermediate_mapping
        input_strides = {d0: 1, d1: size}
        output_strides = {d0: size, d1: 1}

        restickify_spec = _synthetic_spec(
            size,
            num_cores,
            output_split_dim=d1,
            output_stick_dim=d1,
            input_stick_dim=d0,
            input_start_address=0,
            output_start_address=restickify_output_start,
            input_layout_order=[d1, d0],
            output_layout_order=[d0, d1],
            input_strides=input_strides,
            output_strides=output_strides,
        )
        restickify_payload = generate_restickify_dataop_sdsc_from_spec(
            0,
            restickify_spec,
            op_name=restickify_op_name,
            input_work_slices=input_splits,
            input_core_to_work_slice=input_mapping,
            output_work_slices=restickify_output_splits,
            output_core_to_work_slice=restickify_output_mapping,
        )
        if single_op:
            return combine_dataop_sdscs(
                f"0_{restickify_op_name}_{mode}_{direction}_single_dataop",
                [restickify_payload],
            )

        stcdp_spec = _synthetic_spec(
            size,
            num_cores,
            output_split_dim=final_split_dim,
            output_stick_dim=d1,
            input_stick_dim=d1,
            input_start_address=intermediate_start,
            output_start_address=output_start,
            input_layout_order=[d0, d1],
            output_layout_order=[d0, d1],
            input_strides=output_strides,
            output_strides=output_strides,
        )
        stcdp_payload = generate_restickify_dataop_sdsc_from_spec(
            1,
            stcdp_spec,
            op_name="STCDPOpLx",
            input_work_slices=intermediate_splits,
            input_core_to_work_slice=intermediate_mapping,
            output_work_slices=final_splits,
            output_core_to_work_slice=final_mapping,
        )

        return combine_dataop_sdscs(
            f"0_TwoStep{restickify_op_name}Stcdp_{mode}_{direction}_dataop",
            [restickify_payload, stcdp_payload],
        )

    if direction != "kernel-to-output":
        raise ValueError(
            "SPYRE_RESTICKIFY_LX_DATAOP_DIRECTION must be "
            "'kernel-to-output', 'output-to-kernel', or "
            "'stcdp-then-restickify', or "
            "'restickify-stcdp-restickify'"
        )

    input_splits = {d0: 1, d1: num_cores}
    input_mapping = _core_mapping([d0, d1], d1, num_cores)

    # ReStickifyOpLx requires each output piece to cover at least one full
    # input stick.  The input stick is `out`, so the intermediate restickified
    # tensor must not be split across `out` before the local restickify runs.
    # A following STCDP stage can still remap ownership for Stage 3B.
    intermediate_splits = {d0: num_cores, d1: 1}
    intermediate_mapping = _core_mapping([d0, d1], d0, num_cores)

    final_split_dim = d0 if mode == "baseline" else d1
    final_splits = {d0: 1, d1: 1}
    final_splits[final_split_dim] = num_cores
    final_mapping = _core_mapping([d0, d1], final_split_dim, num_cores)

    intermediate_start = 1024 * 1024
    output_start = 1536 * 1024

    restickify_spec = _synthetic_spec(
        size,
        num_cores,
        output_split_dim=d0,
        output_stick_dim=d0,
        input_start_address=0,
        output_start_address=intermediate_start,
    )
    restickify_payload = generate_restickify_dataop_sdsc_from_spec(
        0,
        restickify_spec,
        op_name=restickify_op_name,
        input_work_slices=input_splits,
        input_core_to_work_slice=input_mapping,
        output_work_slices=intermediate_splits,
        output_core_to_work_slice=intermediate_mapping,
    )

    restickified_strides = {d0: 1, d1: size}
    stcdp_spec = _synthetic_spec(
        size,
        num_cores,
        output_split_dim=final_split_dim,
        output_stick_dim=d0,
        input_stick_dim=d0,
        input_start_address=intermediate_start,
        output_start_address=output_start,
        input_layout_order=[d1, d0],
        output_layout_order=[d1, d0],
        input_strides=restickified_strides,
        output_strides=restickified_strides,
    )
    stcdp_payload = generate_restickify_dataop_sdsc_from_spec(
        1,
        stcdp_spec,
        op_name="STCDPOpLx",
        input_work_slices=intermediate_splits,
        input_core_to_work_slice=intermediate_mapping,
        output_work_slices=final_splits,
        output_core_to_work_slice=final_mapping,
    )

    return combine_dataop_sdscs(
        f"0_TwoStep{restickify_op_name}Stcdp_{mode}_dataop",
        [restickify_payload, stcdp_payload],
    )


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_scheduler(path: Path, output_dir: Path, scheduler: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{path.stem}.scheduled.json"
    log = output_dir / f"{path.stem}.scheduled.log"
    proc = subprocess.run(
        [scheduler, "-s", str(path), "-o", str(out)],
        text=True,
        capture_output=True,
        check=False,
    )
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{scheduler} failed for {path}; see {log}")
    return out


def _run_dcg(path: Path, output_dir: Path, dcg_standalone: str) -> tuple[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log = output_dir / f"{path.stem}.dcg.log"
    artifact_dir = output_dir / path.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [dcg_standalone, "-initSdsc", str(path), "-d", str(artifact_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc.returncode, log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--num-cores", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=("baseline", "stage3b"),
        action="append",
        default=None,
        help="baseline splits output on d0; stage3b keeps producer/output on d1",
    )
    parser.add_argument(
        "--op",
        choices=sorted(SUPPORTED_RESTICKIFY_DATA_OPS),
        action="append",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/restickify-lx-dataop-probe",
        help="Directory for generated SDSC JSON and optional scheduler outputs.",
    )
    parser.add_argument("--run-scheduler", action="store_true")
    parser.add_argument("--scheduler", default="L3DlOpsScheduler_standalone")
    parser.add_argument(
        "--stcdp-same-stick",
        action="store_true",
        help=(
            "Emit STCDPOpLx as a same-stick LX-LX movement control. "
            "Without this, STCDPOpLx is expected to reject restickify-shaped "
            "input/output stick changes."
        ),
    )
    parser.add_argument(
        "--run-dcg",
        action="store_true",
        help="Run dcg_standalone -initSdsc on each generated data-op artifact.",
    )
    parser.add_argument(
        "--two-step-lx-restickify",
        action="store_true",
        help=(
            "Emit a composed LX prototype: ReStickifyOpLx performs local "
            "layout conversion, then same-stick STCDPOpLx performs the "
            "cross-core movement needed by the requested mode."
        ),
    )
    parser.add_argument(
        "--streaming-ptlx-tile",
        action="store_true",
        help=(
            "Emit the first static streaming PT-LX tile bridge from the "
            "fragment-level descriptor."
        ),
    )
    parser.add_argument(
        "--native-endpoint-adapter-tile",
        action="store_true",
        help=(
            "With --streaming-ptlx-tile, emit one native PT-LX tile workspace "
            "to consumer LX endpoint adapter artifact."
        ),
    )
    parser.add_argument(
        "--validgap-endpoint-adapter-tile",
        action="store_true",
        help=(
            "With --streaming-ptlx-tile, emit one valid-gap shaped PT-LX "
            "workspace to consumer LX endpoint adapter artifact."
        ),
    )
    parser.add_argument(
        "--native-validgap-endpoint-tile",
        action="store_true",
        help=(
            "With --streaming-ptlx-tile, emit gather plus native PT-LX "
            "restickify plus valid-gap consumer endpoint adapter."
        ),
    )
    parser.add_argument(
        "--all-streaming-tiles",
        action="store_true",
        help="With --streaming-ptlx-tile, emit every materialized tile.",
    )
    parser.add_argument(
        "--full-streaming-bridge",
        action="store_true",
        help=(
            "With --streaming-ptlx-tile, emit one payload containing every "
            "materialized tile."
        ),
    )
    parser.add_argument("--tile-index", type=int, default=0)
    parser.add_argument(
        "--source-work-slices",
        action="append",
        default=None,
        help="Streaming source split item like mb:32; repeat for multiple dims.",
    )
    parser.add_argument(
        "--dest-work-slices",
        action="append",
        default=None,
        help="Streaming destination split item like out:8; repeat for multiple dims.",
    )
    parser.add_argument("--artifact-max-tiles", type=int, default=1)
    parser.add_argument("--dcg-standalone", default="dcg_standalone")
    args = parser.parse_args()

    if not spyre_config.restickify_lx_dataop:
        raise SystemExit(
            "set SPYRE_RESTICKIFY_LX_DATAOP=1 to use this diagnostic prototype"
        )

    d0 = Symbol("mb_")
    d1 = Symbol("out_")
    modes = args.mode or ["baseline", "stage3b"]
    ops = args.op or sorted(SUPPORTED_RESTICKIFY_DATA_OPS)
    output_dir = Path(args.output_dir)
    input_splits = {d0: 1, d1: args.num_cores}
    input_mapping = _core_mapping([d0, d1], d1, args.num_cores)

    rows = []
    if args.streaming_ptlx_tile:
        source_slices = _parse_work_slices(
            args.source_work_slices,
            {"mb": args.num_cores, "out": 1},
        )
        dest_slices = _parse_work_slices(
            args.dest_work_slices,
            {"mb": 4, "out": max(1, args.num_cores // 4)},
        )
        default_sample_limit = ((args.size + 63) // 64) ** 2
        sample_limit = (
            default_sample_limit
            if args.all_streaming_tiles
            else args.artifact_max_tiles
        )
        summary = plan_streaming_ptlx_tiles(
            size=args.size,
            source_work_slices=source_slices,
            source_core_mapping=default_core_mapping(source_slices),
            dest_work_slices=dest_slices,
            dest_core_mapping=default_core_mapping(dest_slices),
            sample_limit=sample_limit,
        )
        artifact = generate_streaming_ptlx_artifact(
            f"streaming_ptlx_{args.size}",
            summary,
            max_tiles=sample_limit,
        )
        root = next(iter(artifact.values()))
        materialized_tiles = root.get("tiles", []) or []
        if args.full_streaming_bridge:
            payload = generate_streaming_ptlx_full_bridge_sdsc(
                f"0_StreamingPTLXFullBridge_{args.size}",
                artifact,
            )
            path = output_dir / f"sdsc_streaming_ptlx_full_{args.size}.json"
            _write_json(path, payload)
            dcg_rc = None
            dcg_log = ""
            if args.run_dcg:
                dcg_rc, dcg_log_path = _run_dcg(
                    path,
                    output_dir / "dcg",
                    args.dcg_standalone,
                )
                dcg_log = str(dcg_log_path)
            row = {
                "mode": "streaming_ptlx_full_bridge",
                "op": "StreamingPTLXFullBridge",
                "size": args.size,
                "path": str(path),
                "dcg_rc": dcg_rc,
                "dcg_log": dcg_log,
                "source_work_slices": source_slices,
                "dest_work_slices": dest_slices,
                "tile_count": summary.total_tiles,
                "sample_tiles": len(summary.sample_tiles),
            }
            rows.append(row)
            print(
                f"streaming_ptlx_full_bridge: wrote {path}"
                + (f" dcg_rc={dcg_rc} dcg_log={dcg_log}" if args.run_dcg else "")
            )
            _write_json(output_dir / "summary.json", {"rows": rows})
            return 0

        tile_indices = (
            range(len(materialized_tiles))
            if args.all_streaming_tiles
            else [args.tile_index]
        )
        for tile_index in tile_indices:
            if args.native_endpoint_adapter_tile:
                payload = generate_native_ptlx_consumer_endpoint_adapter_tile_sdsc(
                    f"{tile_index}_NativePTLXEndpointAdapterTile_{args.size}",
                    artifact,
                    tile_index=tile_index,
                )
                path = output_dir / (
                    f"sdsc_native_ptlx_endpoint_adapter_tile_"
                    f"{args.size}_{tile_index}.json"
                )
                row_mode = "native_ptlx_endpoint_adapter_tile"
                row_op = "NativePTLXEndpointAdapterTile"
            elif args.validgap_endpoint_adapter_tile:
                payload = generate_validgap_ptlx_consumer_endpoint_adapter_tile_sdsc(
                    f"{tile_index}_ValidGapPTLXEndpointAdapterTile_{args.size}",
                    artifact,
                    tile_index=tile_index,
                )
                path = output_dir / (
                    f"sdsc_validgap_ptlx_endpoint_adapter_tile_"
                    f"{args.size}_{tile_index}.json"
                )
                row_mode = "validgap_ptlx_endpoint_adapter_tile"
                row_op = "ValidGapPTLXEndpointAdapterTile"
            elif args.native_validgap_endpoint_tile:
                payload = generate_native_ptlx_validgap_endpoint_tile_bridge_sdsc(
                    f"{tile_index}_NativePTLXValidGapEndpointTile_{args.size}",
                    artifact,
                    tile_index=tile_index,
                )
                path = output_dir / (
                    f"sdsc_native_ptlx_validgap_endpoint_tile_"
                    f"{args.size}_{tile_index}.json"
                )
                row_mode = "native_ptlx_validgap_endpoint_tile"
                row_op = "NativePTLXValidGapEndpointTile"
            else:
                payload = generate_streaming_ptlx_tile_bridge_sdsc(
                    f"{tile_index}_StreamingPTLXTileBridge_{args.size}",
                    artifact,
                    tile_index=tile_index,
                )
                path = output_dir / (
                    f"sdsc_streaming_ptlx_tile_{args.size}_{tile_index}.json"
                )
                row_mode = "streaming_ptlx_tile"
                row_op = "StreamingPTLXTileBridge"
            _write_json(path, payload)
            dcg_rc = None
            dcg_log = ""
            if args.run_dcg:
                dcg_rc, dcg_log_path = _run_dcg(
                    path,
                    output_dir / "dcg",
                    args.dcg_standalone,
                )
                dcg_log = str(dcg_log_path)
            tile = materialized_tiles[tile_index]
            row = {
                "mode": row_mode,
                "op": row_op,
                "size": args.size,
                "path": str(path),
                "dcg_rc": dcg_rc,
                "dcg_log": dcg_log,
                "source_work_slices": source_slices,
                "dest_work_slices": dest_slices,
                "tile_count": summary.total_tiles,
                "sample_tiles": len(summary.sample_tiles),
                "tile_index": tile_index,
                "tile_row": tile.get("tile_row"),
                "tile_col": tile.get("tile_col"),
                "fan_in": tile.get("fan_in"),
                "fan_out": tile.get("fan_out"),
            }
            rows.append(row)
            print(
                f"{row_mode}[{tile_index}]: wrote {path}"
                + (f" dcg_rc={dcg_rc} dcg_log={dcg_log}" if args.run_dcg else "")
            )
        _write_json(output_dir / "summary.json", {"rows": rows})
        return 0

    for mode in modes:
        if args.two_step_lx_restickify:
            payload = _two_step_lx_restickify_payload(
                mode,
                args.size,
                args.num_cores,
            )
            path = output_dir / f"sdsc_{mode}_TwoStepReStickifyLxStcdp_{args.size}.json"
            _write_json(path, payload)
            scheduled = ""
            if args.run_scheduler:
                scheduled = str(
                    _run_scheduler(path, output_dir / "scheduled", args.scheduler)
                )
            dcg_rc = None
            dcg_log = ""
            if args.run_dcg:
                dcg_rc, dcg_log_path = _run_dcg(
                    path,
                    output_dir / "dcg",
                    args.dcg_standalone,
                )
                dcg_log = str(dcg_log_path)
            row = {
                "mode": mode,
                "op": "TwoStepReStickifyLxStcdp",
                "size": args.size,
                "path": str(path),
                "scheduled_path": scheduled,
                "dcg_rc": dcg_rc,
                "dcg_log": dcg_log,
            }
            rows.append(row)
            print(
                f"{mode} TwoStepReStickifyLxStcdp: wrote {path}"
                + (f" scheduled={scheduled}" if scheduled else "")
                + (f" dcg_rc={dcg_rc} dcg_log={dcg_log}" if args.run_dcg else "")
            )
            continue

        output_split_dim = d0 if mode == "baseline" else d1
        output_splits = {d0: 1, d1: 1}
        output_splits[output_split_dim] = args.num_cores
        output_mapping = _core_mapping([d0, d1], output_split_dim, args.num_cores)
        for idx, op_name in enumerate(ops):
            output_stick_dim = (
                d1
                if op_name == "STCDPOpLx" and args.stcdp_same_stick
                else d0
            )
            spec = _synthetic_spec(
                args.size,
                args.num_cores,
                output_split_dim,
                output_stick_dim,
            )
            payload = generate_restickify_dataop_sdsc_from_spec(
                idx,
                spec,
                op_name=op_name,
                input_work_slices=input_splits,
                input_core_to_work_slice=input_mapping,
                output_work_slices=output_splits,
                output_core_to_work_slice=output_mapping,
            )
            path = output_dir / f"sdsc_{mode}_{op_name}_{args.size}.json"
            _write_json(path, payload)
            scheduled = ""
            if args.run_scheduler:
                scheduled = str(
                    _run_scheduler(path, output_dir / "scheduled", args.scheduler)
                )
            dcg_rc = None
            dcg_log = ""
            if args.run_dcg:
                dcg_rc, dcg_log_path = _run_dcg(
                    path,
                    output_dir / "dcg",
                    args.dcg_standalone,
                )
                dcg_log = str(dcg_log_path)
            row = {
                "mode": mode,
                "op": op_name,
                "size": args.size,
                "path": str(path),
                "scheduled_path": scheduled,
                "dcg_rc": dcg_rc,
                "dcg_log": dcg_log,
            }
            rows.append(row)
            print(
                f"{mode} {op_name}: wrote {path}"
                + (f" scheduled={scheduled}" if scheduled else "")
                + (f" dcg_rc={dcg_rc} dcg_log={dcg_log}" if args.run_dcg else "")
            )

    _write_json(output_dir / "summary.json", {"rows": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

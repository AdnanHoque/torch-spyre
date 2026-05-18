#!/usr/bin/env python3
"""Emit standalone restickify data-op SDSCs for Deeptools contract probing."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sympy import Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.codegen.restickify_lx_dataop import (
    SUPPORTED_RESTICKIFY_DATA_OPS,
    combine_dataop_sdscs,
    generate_restickify_dataop_sdsc_from_spec,
)
from torch_spyre._inductor.codegen.superdsc import SDSCArgs, SDSCSpec


def _core_mapping(dims: list[Symbol], split_dim: Symbol, num_cores: int):
    return {
        str(core): {
            str(dim): core if dim == split_dim else 0
            for dim in dims
        }
        for core in range(num_cores)
    }


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
    input_splits = {d0: 1, d1: num_cores}
    input_mapping = _core_mapping([d0, d1], d1, num_cores)

    intermediate_splits = {d0: 1, d1: num_cores}
    intermediate_mapping = _core_mapping([d0, d1], d1, num_cores)

    final_split_dim = d0 if mode == "baseline" else d1
    final_splits = {d0: 1, d1: 1}
    final_splits[final_split_dim] = num_cores
    final_mapping = _core_mapping([d0, d1], final_split_dim, num_cores)

    intermediate_start = 1024 * 1024
    output_start = 1536 * 1024

    restickify_spec = _synthetic_spec(
        size,
        num_cores,
        output_split_dim=d1,
        output_stick_dim=d0,
        input_start_address=0,
        output_start_address=intermediate_start,
    )
    restickify_payload = generate_restickify_dataop_sdsc_from_spec(
        0,
        restickify_spec,
        op_name="ReStickifyOpLx",
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
        f"0_TwoStepReStickifyLxStcdp_{mode}_dataop",
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

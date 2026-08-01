#!/usr/bin/env python3
"""Correctness, structure, and timing probe for LX-fed stationary-W matmul.

The producer and matmul use the same M partition, but the matmul additionally
splits N.  With the LX relayout planner enabled, each producer-resident A shard
is broadcast to the N owners and consumed by the BMM in the same bundle.  The
control uses the identical graph and work division with that planner disabled.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--route", choices=("lx", "hbm"), required=True)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--m-split", type=int, default=4)
    parser.add_argument("--n-split", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--expected-torch-head")
    parser.add_argument("--debug-plans", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def allocation_components(spec: dict[str, Any]) -> list[str]:
    by_index: dict[int, str] = {}
    for wrapper in spec.get("dscs_", []):
        dsc = next(iter(wrapper.values()))
        for node in dsc.get("scheduleTree_", []):
            if node.get("nodeType_") != "allocate":
                continue
            index = node.get("ldsIdx_")
            component = node.get("component_")
            if isinstance(index, int) and isinstance(component, str):
                by_index[index] = component
    return [by_index[index] for index in sorted(by_index)]


def artifact_report(cache: Path, backend_plan_dir: Path) -> dict[str, Any]:
    roots: list[dict[str, Any]] = []
    for path in sorted(cache.rglob("sdsc_*.json")):
        document = json.loads(path.read_text())
        if len(document) != 1:
            continue
        root_name, spec = next(iter(document.items()))
        canonical = copy.deepcopy(spec)
        canonical.pop("debug_handle_", None)
        roots.append(
            {
                "root_name": root_name,
                "op": root_name.split("_", 1)[1],
                "path": str(path),
                "num_cores": spec.get("numCoresUsed_"),
                "physical_core_ids": [
                    int(core) for core in spec.get("coreIdToDsc_", {})
                ],
                "work_slices": copy.deepcopy(spec.get("numWkSlicesPerDim_")),
                "allocation_components": allocation_components(spec),
                "lx_relayout_classifications": copy.deepcopy(
                    spec.get("lxRelayoutClassifications_", [])
                ),
                "canonical_sha256": hashlib.sha256(
                    json.dumps(
                        canonical, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
        )
    roots.sort(key=lambda row: int(row["root_name"].split("_", 1)[0]))
    bundles = sorted(cache.rglob("bundle.mlir"))
    require(len(bundles) == 1, f"expected one bundle, found {bundles}")
    bundle = bundles[0]
    backend_plans = (
        [
            {"path": str(path), **json.loads(path.read_text())}
            for path in sorted(backend_plan_dir.glob("*.json"))
        ]
        if backend_plan_dir.is_dir()
        else []
    )
    return {
        "bundle": str(bundle),
        "bundle_token": bundle.parent.name,
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "op_inventory": [row["op"] for row in roots],
        "roots": roots,
        "backend_plans": backend_plans,
    }


def correctness(actual: Any, expected: Any) -> dict[str, Any]:
    import torch

    cpu = actual.detach().cpu()
    delta = (cpu.float() - expected.float()).abs()
    return {
        "shape_exact": list(cpu.shape) == list(expected.shape),
        "finite": bool(torch.isfinite(cpu).all()),
        "allclose_rtol_5e2_atol_2_5e1": bool(
            torch.allclose(cpu, expected, rtol=5e-2, atol=2.5e-1)
        ),
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "actual_abs_max": float(cpu.float().abs().max()),
        "actual_abs_mean": float(cpu.float().abs().mean()),
        "expected_abs_max": float(expected.float().abs().max()),
        "expected_abs_mean": float(expected.float().abs().mean()),
    }


def trace_report(path: Path, bundle_token: str, runs: int) -> dict[str, Any]:
    events = json.loads(path.read_text()).get("traceEvents", [])
    kernels = [
        event
        for event in events
        if event.get("cat") == "kernel"
        and event.get("ph") == "X"
        and isinstance(event.get("dur"), (int, float))
    ]
    matched = [
        float(event["dur"])
        for event in kernels
        if bundle_token in str(event.get("name"))
    ]
    gate = len(kernels) == runs and len(matched) == runs and all(matched)
    return {
        "gate": gate,
        "kernel_event_count": len(kernels),
        "matched_event_count": len(matched),
        "kernel_names": sorted({str(event.get("name")) for event in kernels}),
        "device_us": (
            {
                "median": statistics.median(matched),
                "mean": statistics.fmean(matched),
                "min": min(matched),
                "max": max(matched),
                "samples": matched,
            }
            if matched
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    require(args.m % args.m_split == 0, "M must divide evenly by M split")
    require(args.m_split * args.n_split == 32, "probe uses all 32 cores")
    require(not args.run_dir.exists(), f"run directory exists: {args.run_dir}")
    cache = args.run_dir / "cache"
    trace_dir = args.run_dir / "trace"
    cache.mkdir(parents=True)
    trace_dir.mkdir()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    os.environ.setdefault("DXP_LX_FRAC_AVAIL", "0.2")

    import torch
    import torch_spyre

    try:
        from core.profiler import create_profiler
    except ModuleNotFoundError:
        from torch.profiler import ProfilerActivity, profile

        def create_profiler(
            torch_module: Any,
            trace_directory: str,
            *,
            profile_memory: bool,
            with_stack: bool,
        ) -> Any:
            return profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                profile_memory=profile_memory,
                with_stack=with_stack,
                on_trace_ready=torch_module.profiler.tensorboard_trace_handler(
                    trace_directory
                ),
            )

    from torch_spyre._inductor import config as spyre_config
    from torch_spyre._inductor import spyre_hint
    from torch_spyre._inductor.propagate_hints import _reset_counter
    from torch_spyre._inductor.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
        reset as reset_named_dims,
    )

    if args.debug_plans:
        import torch_spyre._inductor.lx_relayout as relayout_module
        import torch_spyre._inductor.scratchpad.allocator as allocator_module

        original_ratio = relayout_module._destination_size_ratio
        original_collect = allocator_module.collect_lx_relayout_plans
        original_same = relayout_module._same_core_placement
        original_view = relayout_module._per_core_view_on_buf

        def debug_ratio(*ratio_args: Any, **ratio_kwargs: Any) -> int | None:
            value = original_ratio(*ratio_args, **ratio_kwargs)
            print(
                "LX_RELAYOUT_GEOMETRY "
                + json.dumps(
                    {
                        "producer_map": ratio_args[0],
                        "producer_splits": ratio_args[1],
                        "consumer_map": ratio_args[2],
                        "consumer_splits": ratio_args[3],
                        "destination_size_ratio": value,
                    },
                    sort_keys=True,
                )
            )
            return value

        def debug_collect(*collect_args: Any, **collect_kwargs: Any) -> Any:
            plans = original_collect(*collect_args, **collect_kwargs)
            print(
                "LX_RELAYOUT_PLANS "
                + json.dumps(
                    [
                        {
                            "source": plan.source_name,
                            "consumer": plan.consumer_name,
                            "destination_size_ratio": plan.destination_size_ratio,
                        }
                        for plan in plans
                    ],
                    sort_keys=True,
                )
            )
            return plans

        def debug_view(*view_args: Any, **view_kwargs: Any) -> Any:
            value = original_view(*view_args, **view_kwargs)
            print(
                "LX_RELAYOUT_VIEW "
                + json.dumps(
                    {
                        "op": view_args[0].get_name(),
                        "buffer": view_args[2],
                        "view": repr(value[0]),
                        "has_partial_reduction": value[1],
                        "representable": value[2],
                    },
                    sort_keys=True,
                )
            )
            return value

        def debug_same(*same_args: Any, **same_kwargs: Any) -> bool:
            value = original_same(*same_args, **same_kwargs)
            print(
                "LX_RELAYOUT_SAME "
                + json.dumps(
                    {
                        "producer_cores": same_args[1],
                        "consumer_cores": same_args[3],
                        "same": value,
                    },
                    sort_keys=True,
                )
            )
            return value

        relayout_module._destination_size_ratio = debug_ratio
        relayout_module._per_core_view_on_buf = debug_view
        relayout_module._same_core_placement = debug_same
        allocator_module.collect_lx_relayout_plans = debug_collect

    torch_spyre._autoload()
    imported = Path(torch_spyre.__file__).resolve()
    torch_root = Path(
        os.popen(f"git -C {imported.parent} rev-parse --show-toplevel").read().strip()
    )
    torch_head = os.popen(f"git -C {torch_root} rev-parse HEAD").read().strip()
    if args.expected_torch_head:
        require(
            torch_head == args.expected_torch_head,
            f"torch head {torch_head} != {args.expected_torch_head}",
        )

    generator = torch.Generator().manual_seed(args.seed)
    # Express the cohort as a named batch dimension. This is the same
    # work-division-agnostic DDL geometry used by owner-local GQA: each cohort
    # owns one independent row panel while all cohorts share the 2-D weight.
    activation_cpu = (
        torch.randn(
            (args.m_split, args.m // args.m_split, args.k),
            dtype=torch.float16,
            generator=generator,
        )
        * 0.125
    )
    weight_cpu = (
        torch.randn((args.k, args.n), dtype=torch.float16, generator=generator) * 0.125
    )
    scale = 0.5
    expected = torch.matmul(activation_cpu * scale, weight_cpu)
    device = torch.device("spyre")
    activation = activation_cpu.to(device)
    weight = weight_cpu.to(device)

    consumer_work_div = {"cohort": args.m_split, "N": args.n_split}
    source_work_div = {"cohort": args.m_split}
    source_core_ids = tuple(shard * args.n_split for shard in range(args.m_split))

    class Graph(torch.nn.Module):
        def forward(self, a: Any, w: Any) -> Any:
            with spyre_hint(
                work_div=source_work_div,
                physical_core_ids=list(source_core_ids),
            ):
                a_lx = a * scale
            with spyre_hint(physical_core_order="work_div_inner_first"):
                with spyre_hint(work_div=consumer_work_div):
                    return torch.matmul(a_lx, w)

    reset_named_dims()
    _reset_counter()
    for name, size in (
        ("cohort", args.m_split),
        ("M", args.m // args.m_split),
        ("K", args.k),
        ("N", args.n),
    ):
        declare_tensor_dim(name, size)
    name_tensor_dims(activation, ["cohort", "M", "K"])
    name_tensor_dims(weight, ["K", "N"])
    patch = {
        "sencores": 32,
        "lx_planning": True,
        "lx_planner_relayout": args.route == "lx",
        "lx_matmul_operand_broadcast": args.route == "lx",
        "matmul_dataflow": "weight_stationary",
    }
    try:
        with spyre_config.patch(patch):
            compiled: Callable[..., Any] = torch.compile(
                Graph().to(device), fullgraph=True
            )
        with torch.no_grad(), spyre_config.patch(patch):
            compile_output = compiled(activation, weight)
            torch.spyre.synchronize()
    finally:
        reset_named_dims()
        _reset_counter()

    compile_correctness = correctness(compile_output, expected)
    for _ in range(args.warmups):
        with torch.no_grad(), spyre_config.patch(patch):
            compiled(activation, weight)
        torch.spyre.synchronize()

    profiler = create_profiler(
        torch, str(trace_dir), profile_memory=True, with_stack=False
    )
    walls_us: list[float] = []
    measured = None
    profiler.start()
    for _ in range(args.runs):
        started = time.perf_counter_ns()
        with torch.no_grad(), spyre_config.patch(patch):
            measured = compiled(activation, weight)
        torch.spyre.synchronize()
        walls_us.append((time.perf_counter_ns() - started) / 1000.0)
        profiler.step()
    profiler.stop()
    require(measured is not None, "no measured output")
    measured_correctness = correctness(measured, expected)

    artifacts = artifact_report(cache, args.run_dir / "backend_plans")
    shuffles = [row for row in artifacts["roots"] if "shuffle" in row["op"]]
    bmms = [row for row in artifacts["roots"] if "batchmatmul" in row["op"]]
    producers = [row for row in artifacts["roots"] if row["op"] == "mul"]
    bmm = bmms[0] if len(bmms) == 1 else None
    producer = producers[0] if len(producers) == 1 else None
    broadcast_contracts = (
        [
            row
            for row in bmm["lx_relayout_classifications"]
            if row.get("kind") == "matmul_operand_broadcast"
        ]
        if bmm is not None
        else []
    )
    backend_plans = [
        row
        for row in artifacts["backend_plans"]
        if row.get("artifact_kind") == "matmul_operand_broadcast_backend_plan"
    ]
    common_gates = {
        "one_producer": producer is not None,
        "one_bmm": bmm is not None,
        "producer_on_cohort_roots": producer is not None
        and producer["physical_core_ids"] == list(source_core_ids),
        "bmm_uses_expected_grid": bmm is not None
        and bmm["work_slices"].get("x") == args.m_split
        and bmm["work_slices"].get("out") == args.n_split
        and bmm["work_slices"].get("in") == 1
        and bmm["work_slices"].get("mb") == 1,
    }
    if args.route == "lx":
        route_gates = {
            "no_standalone_shuffle": not shuffles,
            "one_frontend_broadcast_contract": len(broadcast_contracts) == 1,
            "broadcast_targets_input_operand": len(broadcast_contracts) == 1
            and broadcast_contracts[0].get("consumer_operand_ds_type") == "INPUT"
            and broadcast_contracts[0].get("operand_index") == 0,
            "one_realized_backend_broadcast": len(backend_plans) == 1
            and backend_plans[0].get("realized") is True
            and backend_plans[0].get("physical_lowering_status")
            == "lowered_resident_input_fetch",
            "backend_grouped_fanout_exact": len(backend_plans) == 1
            and int(backend_plans[0].get("group_count", 0)) == args.m_split
            and int(backend_plans[0].get("replication_factor", 0)) == args.n_split
            and int(backend_plans[0].get("logical_transfer_count", 0)) == 32,
            "bmm_consumes_lx_activation": bmm is not None
            and bmm["allocation_components"][0] == "lx",
            "bmm_reads_weight_from_hbm": bmm is not None
            and bmm["allocation_components"][1] == "hbm",
            "no_restickify": not any(
                "restickify" in row["op"] for row in artifacts["roots"]
            ),
        }
    else:
        route_gates = {
            "no_shuffle": not shuffles,
            "no_lx_broadcast_contract": not broadcast_contracts,
            "no_backend_broadcast": not backend_plans,
            "bmm_reads_activation_from_hbm": bmm is not None
            and bmm["allocation_components"][0] == "hbm",
            "bmm_reads_weight_from_hbm": bmm is not None
            and bmm["allocation_components"][1] == "hbm",
        }
    structural_gates = {**common_gates, **route_gates}
    structural_gate = all(structural_gates.values())
    correctness_gate = all(
        row["shape_exact"] and row["finite"] and row["allclose_rtol_5e2_atol_2_5e1"]
        for row in (compile_correctness, measured_correctness)
    )
    traces = sorted(glob.glob(str(trace_dir / "*.pt.trace.json")))
    require(len(traces) == 1, f"expected one trace, found {traces}")
    trace = trace_report(Path(traces[0]), artifacts["bundle_token"], args.runs)
    report = {
        "schema": "lx_stationary_weight_matmul_probe_v2",
        "route": args.route,
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "source_work_div": source_work_div,
        "source_core_ids": list(source_core_ids),
        "consumer_work_div": consumer_work_div,
        "dataflow": {
            "activation": "producer output remains LX resident",
            "route": (
                "cohort-root STCDP LX-to-LX grouped fan-out"
                if args.route == "lx"
                else "HBM spill/reload control"
            ),
            "weight": "HBM-resident and N-owner stationary",
            "consume": (
                "STCDP input fetch writes the existing BMM LX operand allocation "
                "immediately before the DL schedule step in one device bundle"
                if args.route == "lx"
                else "producer spills to HBM and BMM reloads in one device bundle"
            ),
        },
        "torch_spyre_root": str(torch_root),
        "torch_spyre_head": torch_head,
        "correctness_gate": correctness_gate,
        "correctness": {
            "compile": compile_correctness,
            "measured": measured_correctness,
        },
        "structural_gate": structural_gate,
        "structural_gates": structural_gates,
        "artifacts": artifacts,
        "trace_path": traces[0],
        "trace": trace,
        "host_wall_diagnostic_us": {
            "median": statistics.median(walls_us),
            "mean": statistics.fmean(walls_us),
            "samples": walls_us,
        },
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "route": args.route,
                "correctness_gate": correctness_gate,
                "structural_gate": structural_gate,
                "trace_gate": trace["gate"],
                "device_median_us": (
                    trace["device_us"]["median"] if trace["device_us"] else None
                ),
                "op_inventory": artifacts["op_inventory"],
                "run_dir": str(args.run_dir),
            },
            sort_keys=True,
        )
    )
    if not (correctness_gate and structural_gate and trace["gate"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

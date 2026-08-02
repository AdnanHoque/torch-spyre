#!/usr/bin/env python3
"""Correctness, structure, and timing probe for LX-fed stationary-W matmul.

The producer and matmul use the same M partition, but the matmul additionally
splits N. Three matched routes hold the Torch graph and work division fixed:

lx
    Bind the planned grouped fan-out directly to the BMM input allocation.
lx_explicit
    Materialize the same plan as a generic S1 -> SHUFFLE -> S2 prefix.
hbm
    Disable relayout planning so the producer edge spills through HBM.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
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
    parser.add_argument(
        "--route", choices=("lx", "lx_explicit", "hbm"), required=True
    )
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument(
        "--projection-widths",
        help="comma-separated logical projection widths; defaults to --n",
    )
    parser.add_argument(
        "--projection-schedule",
        choices=("fused", "separate"),
        default="fused",
    )
    parser.add_argument("--grid", choices=("fixed", "auto"), default="fixed")
    parser.add_argument("--m-split", type=int, default=4)
    parser.add_argument("--n-split", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--timing-source", choices=("kineto", "aiupti"), default="kineto"
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--expected-torch-head")
    parser.add_argument("--debug-plans", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class _AiuptiActivity(ctypes.Structure):
    _fields_ = [("activity_kind", ctypes.c_uint8)]


class _AiuptiActivityCompute(ctypes.Structure):
    _fields_ = [
        ("activity_kind", ctypes.c_uint8),
        ("operation_kind", ctypes.c_uint8),
        ("device_id", ctypes.c_uint32),
        ("context_id", ctypes.c_uint32),
        ("stream_id", ctypes.c_uint32),
        ("correlation_id", ctypes.c_uint32),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("queued", ctypes.c_uint64),
        ("submitted", ctypes.c_uint64),
        ("local_memory_total", ctypes.c_size_t),
        ("name", ctypes.c_char * 128),
        ("cycles_ts1", ctypes.c_uint64),
        ("cycles_ts2", ctypes.c_uint64),
        ("cycles_ts3", ctypes.c_uint64),
        ("cycles_ts4", ctypes.c_uint64),
        ("cycles_ts5", ctypes.c_uint64),
    ]


class AiuptiComputeCollector:
    """Collect raw AIUPTI compute activities without a Kineto bridge."""

    _success = 0
    _compute_kind = 1
    _execute_kind = 2
    _buffer_bytes = 1 << 20

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.callback_errors: list[str] = []
        self.buffers: list[Any] = []
        self.dropped = 0
        self.lib = ctypes.CDLL("libaiupti.so", mode=ctypes.RTLD_GLOBAL)
        request_type = ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        )
        complete_type = ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
        )

        @request_type
        def request(
            buffer: Any, size: Any, max_num_records: Any
        ) -> None:
            allocation = ctypes.create_string_buffer(self._buffer_bytes)
            self.buffers.append(allocation)
            buffer[0] = ctypes.cast(allocation, ctypes.POINTER(ctypes.c_uint8))
            size[0] = self._buffer_bytes
            max_num_records[0] = 0

        @complete_type
        def complete(buffer: Any, _size: int, valid_size: int) -> None:
            try:
                while True:
                    record = ctypes.POINTER(_AiuptiActivity)()
                    result = self.get_next_record(
                        buffer, valid_size, ctypes.byref(record)
                    )
                    if result != self._success:
                        break
                    if not record:
                        self.callback_errors.append("AIUPTI returned a null record")
                        break
                    if record.contents.activity_kind != self._compute_kind:
                        continue
                    compute = ctypes.cast(
                        record, ctypes.POINTER(_AiuptiActivityCompute)
                    ).contents
                    self.records.append(
                        {
                            "operation_kind": int(compute.operation_kind),
                            "device_id": int(compute.device_id),
                            "context_id": int(compute.context_id),
                            "stream_id": int(compute.stream_id),
                            "correlation_id": int(compute.correlation_id),
                            "start_ns": int(compute.start),
                            "end_ns": int(compute.end),
                            "queued_ns": int(compute.queued),
                            "submitted_ns": int(compute.submitted),
                            "name": bytes(compute.name)
                            .split(b"\0", 1)[0]
                            .decode(errors="replace"),
                        }
                    )
            except BaseException as error:  # callbacks cannot propagate safely
                self.callback_errors.append(repr(error))

        self.request_callback = request
        self.complete_callback = complete
        self.register_callbacks = getattr(
            self.lib,
            "_Z31aiuptiActivityRegisterCallbacksPFvPPhPmS1_EPFvS_mmE",
        )
        self.enable = getattr(
            self.lib, "_Z20aiuptiActivityEnable19AIUpti_ActivityKind"
        )
        self.disable = getattr(
            self.lib, "_Z21aiuptiActivityDisable19AIUpti_ActivityKind"
        )
        self.flush = getattr(self.lib, "_Z24aiuptiFlushAllActivitiesv")
        self.get_next_record = getattr(
            self.lib, "_Z27aiuptiActivityGetNextRecordPhmPP15AIUpti_Activity"
        )
        self.get_dropped = getattr(
            self.lib, "_Z34aiuptiActivityGetNumDroppedRecordsPm"
        )
        self.register_callbacks.argtypes = [request_type, complete_type]
        self.enable.argtypes = [ctypes.c_int]
        self.disable.argtypes = [ctypes.c_int]
        self.get_next_record.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(_AiuptiActivity)),
        ]
        self.get_dropped.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
        for function in (
            self.register_callbacks,
            self.enable,
            self.disable,
            self.flush,
            self.get_next_record,
            self.get_dropped,
        ):
            function.restype = ctypes.c_int

    def start(self) -> None:
        require(
            self.register_callbacks(
                self.request_callback, self.complete_callback
            )
            == self._success,
            "failed to register AIUPTI callbacks",
        )
        require(
            self.enable(self._compute_kind) == self._success,
            "failed to enable AIUPTI compute activities",
        )

    def stop(self) -> None:
        require(self.flush() == self._success, "failed to flush AIUPTI activities")
        require(
            self.disable(self._compute_kind) == self._success,
            "failed to disable AIUPTI compute activities",
        )
        dropped = ctypes.c_size_t()
        require(
            self.get_dropped(ctypes.byref(dropped)) == self._success,
            "failed to query dropped AIUPTI records",
        )
        self.dropped = int(dropped.value)


def aiupti_report(
    collector: AiuptiComputeCollector, bundle_token: str, runs: int
) -> dict[str, Any]:
    executed = [
        record
        for record in collector.records
        if record["operation_kind"] == collector._execute_kind
        and record["end_ns"] > record["start_ns"]
    ]
    matched = [record for record in executed if bundle_token in record["name"]]
    durations = [
        (record["end_ns"] - record["start_ns"]) / 1000.0 for record in matched
    ]
    gate = (
        not collector.callback_errors
        and collector.dropped == 0
        and len(executed) == runs
        and len(matched) == runs
        and len(durations) == runs
    )
    return {
        "source": "AIUPTI compute-execute activity",
        "gate": gate,
        "kernel_event_count": len(executed),
        "matched_event_count": len(matched),
        "kernel_names": sorted({record["name"] for record in executed}),
        "dropped_record_count": collector.dropped,
        "callback_errors": collector.callback_errors,
        "raw_compute_records": collector.records,
        "device_us": (
            {
                "median": statistics.median(durations),
                "mean": statistics.fmean(durations),
                "min": min(durations),
                "max": max(durations),
                "samples": durations,
            }
            if durations
            else None
        ),
    }


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


def output_correctness(actual: Any, expected: Any) -> dict[str, Any]:
    actuals = actual if isinstance(actual, (tuple, list)) else (actual,)
    expecteds = expected if isinstance(expected, (tuple, list)) else (expected,)
    rows = [correctness(lhs, rhs) for lhs, rhs in zip(actuals, expecteds)]
    return {
        "output_count_exact": len(actuals) == len(expecteds),
        "outputs": rows,
        "allclose": len(actuals) == len(expecteds)
        and all(
            row["shape_exact"] and row["finite"] and row["allclose_rtol_5e2_atol_2_5e1"]
            for row in rows
        ),
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
    projection_widths = (
        tuple(int(value) for value in args.projection_widths.split(","))
        if args.projection_widths
        else (args.n,)
    )
    require(projection_widths, "at least one projection width is required")
    require(all(width > 0 for width in projection_widths), "widths must be positive")
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
    from torch._inductor import config as inductor_config

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
                record_shapes=True,
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
    logical_weight_cpus = tuple(
        torch.randn((args.k, width), dtype=torch.float16, generator=generator) * 0.125
        for width in projection_widths
    )
    scale = 0.5
    expected_parts = tuple(
        torch.matmul(activation_cpu * scale, weight) for weight in logical_weight_cpus
    )
    if args.projection_schedule == "fused":
        weight_cpus = (torch.cat(logical_weight_cpus, dim=-1),)
        expected = expected_parts[0] if len(expected_parts) == 1 else expected_parts
        n_dim_names = ("N",)
    else:
        weight_cpus = logical_weight_cpus
        expected = expected_parts
        n_dim_names = tuple(f"N{index}" for index in range(len(weight_cpus)))
    device = torch.device("spyre")
    activation = activation_cpu.to(device)
    weights = tuple(weight.to(device) for weight in weight_cpus)

    consumer_work_divs = tuple(
        {"cohort": args.m_split, dim_name: args.n_split} for dim_name in n_dim_names
    )
    source_work_div = {"cohort": args.m_split}
    source_core_ids = tuple(shard * args.n_split for shard in range(args.m_split))

    class Graph(torch.nn.Module):
        def forward(self, a: Any, *graph_weights: Any) -> Any:
            with spyre_hint(
                work_div=source_work_div,
                physical_core_ids=list(source_core_ids),
            ):
                a_lx = a * scale

            outputs = []
            for index, graph_weight in enumerate(graph_weights):
                if args.grid == "fixed":
                    with spyre_hint(physical_core_order="work_div_inner_first"):
                        with spyre_hint(work_div=consumer_work_divs[index]):
                            output = torch.matmul(a_lx, graph_weight)
                else:
                    output = torch.matmul(a_lx, graph_weight)
                outputs.append(output)
            if args.projection_schedule == "fused" and len(projection_widths) > 1:
                return torch.split(outputs[0], projection_widths, dim=-1)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

    reset_named_dims()
    _reset_counter()
    for name, size in (
        ("cohort", args.m_split),
        ("M", args.m // args.m_split),
        ("K", args.k),
    ):
        declare_tensor_dim(name, size)
    for name, weight in zip(n_dim_names, weight_cpus):
        declare_tensor_dim(name, weight.shape[-1])
    name_tensor_dims(activation, ["cohort", "M", "K"])
    for weight, dim_name in zip(weights, n_dim_names):
        name_tensor_dims(weight, ["K", dim_name])
    uses_lx_relayout = args.route in {"lx", "lx_explicit"}
    uses_attached_broadcast = args.route == "lx"
    patch = {
        "sencores": 32,
        "lx_planning": True,
        "lx_planner_relayout": uses_lx_relayout,
        "lx_matmul_operand_broadcast": uses_attached_broadcast,
        "matmul_dataflow": "weight_stationary",
    }
    try:
        with inductor_config.patch(
            {"profiler_mark_wrapper_call": True}
        ), spyre_config.patch(patch):
            compiled: Callable[..., Any] = torch.compile(
                Graph().to(device), fullgraph=True
            )
            with torch.no_grad():
                compile_output = compiled(activation, *weights)
                torch.spyre.synchronize()
    finally:
        reset_named_dims()
        _reset_counter()

    compile_correctness = output_correctness(compile_output, expected)
    for _ in range(args.warmups):
        with torch.no_grad(), spyre_config.patch(patch):
            compiled(activation, *weights)
        torch.spyre.synchronize()

    profiler = (
        create_profiler(
            torch, str(trace_dir), profile_memory=True, with_stack=False
        )
        if args.timing_source == "kineto"
        else None
    )
    aiupti_collector = (
        AiuptiComputeCollector() if args.timing_source == "aiupti" else None
    )
    walls_us: list[float] = []
    measured = None
    if profiler is not None:
        profiler.start()
    else:
        require(aiupti_collector is not None, "missing AIUPTI collector")
        aiupti_collector.start()
    for _ in range(args.runs):
        started = time.perf_counter_ns()
        with torch.no_grad(), spyre_config.patch(patch):
            measured = compiled(activation, *weights)
        torch.spyre.synchronize()
        walls_us.append((time.perf_counter_ns() - started) / 1000.0)
        if profiler is not None:
            profiler.step()
    if profiler is not None:
        profiler.stop()
    else:
        require(aiupti_collector is not None, "missing AIUPTI collector")
        aiupti_collector.stop()
    require(measured is not None, "no measured output")
    measured_correctness = output_correctness(measured, expected)

    artifacts = artifact_report(cache, args.run_dir / "backend_plans")
    shuffles = [row for row in artifacts["roots"] if "shuffle" in row["op"]]
    bmms = [row for row in artifacts["roots"] if "batchmatmul" in row["op"]]
    producers = [row for row in artifacts["roots"] if row["op"] == "mul"]
    producer = producers[0] if len(producers) == 1 else None
    expected_bmm_count = len(weights)
    broadcast_contracts = [
        contract
        for bmm in bmms
        for contract in bmm["lx_relayout_classifications"]
        if contract.get("kind") == "matmul_operand_broadcast"
    ]
    backend_plans = [
        row
        for row in artifacts["backend_plans"]
        if row.get("artifact_kind") == "matmul_operand_broadcast_backend_plan"
    ]
    expected_grid = args.grid == "fixed"
    common_gates = {
        "one_producer": producer is not None,
        "expected_bmm_count": len(bmms) == expected_bmm_count,
        "producer_on_cohort_roots": producer is not None
        and producer["physical_core_ids"] == list(source_core_ids),
        "bmms_use_expected_grid": (not expected_grid)
        or all(
            bmm["work_slices"].get("x") == args.m_split
            and bmm["work_slices"].get("out") == args.n_split
            and bmm["work_slices"].get("in") == 1
            and bmm["work_slices"].get("mb") == 1
            for bmm in bmms
        ),
        "auto_grid_uses_all_cores": expected_grid
        or all(bmm["num_cores"] == 32 for bmm in bmms),
    }
    if uses_attached_broadcast:
        route_gates = {
            "no_standalone_shuffle": not shuffles,
            "frontend_broadcast_per_bmm": len(broadcast_contracts)
            == expected_bmm_count,
            "broadcasts_target_input_operand": all(
                contract.get("consumer_operand_ds_type") == "INPUT"
                and contract.get("operand_index") == 0
                for contract in broadcast_contracts
            ),
            "realized_backend_broadcast_per_bmm": len(backend_plans)
            == expected_bmm_count
            and all(
                plan.get("realized") is True
                and plan.get("physical_lowering_status")
                == "lowered_resident_input_fetch"
                for plan in backend_plans
            ),
            "backend_grouped_fanout_exact": len(backend_plans) == expected_bmm_count
            and all(
                int(plan.get("group_count", 0)) == args.m_split
                and int(plan.get("replication_factor", 0)) == args.n_split
                and int(plan.get("logical_transfer_count", 0)) == 32
                for plan in backend_plans
            ),
            "bmms_consume_lx_activation": all(
                bmm["allocation_components"][0] == "lx" for bmm in bmms
            ),
            "bmms_read_weight_from_hbm": all(
                bmm["allocation_components"][1] == "hbm" for bmm in bmms
            ),
            "no_restickify": not any(
                "restickify" in row["op"] for row in artifacts["roots"]
            ),
        }
    elif args.route == "lx_explicit":
        route_gates = {
            "one_explicit_shuffle_per_bmm": len(shuffles) == expected_bmm_count,
            "no_attached_broadcast_contract": not broadcast_contracts,
            "no_attached_backend_broadcast": not backend_plans,
            "bmms_consume_materialized_lx_activation": all(
                bmm["allocation_components"][0] == "lx" for bmm in bmms
            ),
            "bmms_read_weight_from_hbm": all(
                bmm["allocation_components"][1] == "hbm" for bmm in bmms
            ),
            "shuffle_uses_lx_for_s1_and_s2": all(
                shuffle["allocation_components"] == ["lx", "lx"]
                for shuffle in shuffles
            ),
            "no_restickify": not any(
                "restickify" in row["op"] for row in artifacts["roots"]
            ),
        }
    else:
        route_gates = {
            "no_shuffle": not shuffles,
            "no_lx_broadcast_contract": not broadcast_contracts,
            "no_backend_broadcast": not backend_plans,
            "bmms_read_activation_from_hbm": all(
                bmm["allocation_components"][0] == "hbm" for bmm in bmms
            ),
            "bmms_read_weight_from_hbm": all(
                bmm["allocation_components"][1] == "hbm" for bmm in bmms
            ),
        }
    structural_gates = {**common_gates, **route_gates}
    structural_gate = all(structural_gates.values())
    correctness_gate = (
        compile_correctness["allclose"] and measured_correctness["allclose"]
    )
    if profiler is not None:
        traces = sorted(glob.glob(str(trace_dir / "*.pt.trace.json")))
        require(len(traces) == 1, f"expected one trace, found {traces}")
        trace_path: str | None = traces[0]
        trace = trace_report(
            Path(traces[0]), artifacts["bundle_token"], args.runs
        )
    else:
        require(aiupti_collector is not None, "missing AIUPTI collector")
        trace_path = None
        trace = aiupti_report(
            aiupti_collector, artifacts["bundle_token"], args.runs
        )
    report = {
        "schema": "lx_stationary_weight_matmul_probe_v4",
        "route": args.route,
        "timing_source": args.timing_source,
        "shape": {
            "m": args.m,
            "k": args.k,
            "projection_widths": list(projection_widths),
            "total_n": sum(projection_widths),
        },
        "projection_schedule": args.projection_schedule,
        "projection_bmm_count": expected_bmm_count,
        "logical_output_count": len(projection_widths),
        "grid": args.grid,
        "source_work_div": source_work_div,
        "source_core_ids": list(source_core_ids),
        "consumer_work_divs": (
            list(consumer_work_divs) if args.grid == "fixed" else None
        ),
        "dataflow": {
            "activation": (
                "producer output remains LX resident"
                if uses_lx_relayout
                else "producer output spills to HBM before the BMM"
            ),
            "route": (
                "cohort-root STCDP LX-to-LX grouped fan-out attached to BMM"
                if uses_attached_broadcast
                else (
                    "generic explicit S1-to-S2 LX shuffle before BMM"
                    if args.route == "lx_explicit"
                    else "HBM spill/reload control"
                )
            ),
            "weight": "HBM-resident and N-owner stationary",
            "consume": (
                "STCDP input fetch writes the existing BMM LX operand allocation "
                "immediately before the DL schedule step in one device bundle"
                if uses_attached_broadcast
                else (
                    "explicit shuffle writes a disjoint S2 allocation consumed by "
                    "the BMM in the same device bundle"
                    if args.route == "lx_explicit"
                    else "producer spills to HBM and BMM reloads in one device bundle"
                )
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
        "trace_path": trace_path,
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
                "projection_schedule": args.projection_schedule,
                "projection_widths": list(projection_widths),
                "grid": args.grid,
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

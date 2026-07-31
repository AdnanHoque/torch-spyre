#!/usr/bin/env python3
"""Matched timing for incumbent and activation-stationary matmul.

Both arms use the same activation, W[N,K], 32-core budget, compiler stack,
device, process, and I-C-C-I launch order. Work ownership can be fixed to N32
or selected independently by the ordinary planner.

The default ``paid`` boundary includes Design A's padding, activation
restickify, BMM, and output slice. The ``compute-only`` aligned-M oracle
preplaces each arm's A and W in its native layout and returns each BMM's native
C layout. It fails unless both emitted programs contain exactly one BMM root
and no conversion roots. Only Kineto ``cat == "kernel"`` complete-event
duration is performance evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable


CORES = 32
PHYSICAL_M = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--cores", type=int, default=CORES)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--core-frequency-mhz", type=float, default=1100.0)
    parser.add_argument(
        "--boundary",
        choices=("paid", "compute-only"),
        default="paid",
    )
    parser.add_argument(
        "--candidate-source",
        choices=("manual", "selector"),
        default="manual",
    )
    parser.add_argument(
        "--work-division",
        choices=("n32", "auto"),
        default="n32",
    )
    parser.add_argument("--incumbent-m-split", type=int, default=0)
    parser.add_argument("--incumbent-n-split", type=int, default=0)
    parser.add_argument("--incumbent-k-split", type=int, default=0)
    parser.add_argument("--candidate-m-split", type=int, default=0)
    parser.add_argument("--candidate-n-split", type=int, default=0)
    parser.add_argument("--candidate-k-split", type=int, default=0)
    parser.add_argument(
        "--candidate-core-order",
        choices=("auto", "row_major"),
        default="auto",
        help=(
            "core placement for an explicit candidate split; keep auto when "
            "attributing work division independently"
        ),
    )
    parser.add_argument(
        "--weight-layout",
        choices=("default", "stationary", "per-arm"),
        default="default",
        help=(
            "HBM layout used for W: default sticks K, stationary sticks N, "
            "and per-arm gives each algorithm its native preloaded layout"
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-torch-head")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_checked(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError(
            {
                "command": command,
                "returncode": completed.returncode,
                "output": completed.stdout,
            }
        )
    return completed.stdout.strip()


def stats(values: list[float]) -> dict[str, Any]:
    require(bool(values), "duration set is empty")
    return {
        "unit": "us",
        "count": len(values),
        "min_us": min(values),
        "median_us": statistics.median(values),
        "mean_us": statistics.fmean(values),
        "max_us": max(values),
        "durations_us": values,
    }


def correctness(torch: Any, actual: Any, expected: Any) -> dict[str, Any]:
    actual_f = actual.detach().cpu().float()
    expected_f = expected.detach().cpu().float()
    difference = (actual_f - expected_f).abs()
    return {
        "shape_exact": list(actual_f.shape) == list(expected_f.shape),
        "finite": bool(torch.isfinite(actual_f).all()),
        "allclose_rtol_5e2_atol_2_5e1": bool(
            torch.allclose(actual_f, expected_f, rtol=5e-2, atol=2.5e-1)
        ),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
    }


def inventory(cache: Path) -> dict[str, Any]:
    bundles = sorted(cache.rglob("bundle.mlir"))
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        roots = []
        for descriptor in sorted(bundle.parent.glob("sdsc_*.json")):
            suffix = descriptor.stem.removeprefix("sdsc_")
            if not suffix.isdigit():
                continue
            document = json.loads(descriptor.read_text())
            require(len(document) == 1, f"unexpected descriptor: {descriptor}")
            roots.append(next(iter(document)).split("_", 1)[1])
        ideal_cycles_path = bundle.parent / "perf" / "ideal_cycles.json"
        ideal_cycles = None
        if ideal_cycles_path.exists():
            entries = json.loads(ideal_cycles_path.read_text())
            totals = [
                int(entry["ideal_cycles"])
                for entry in entries
                if entry.get("sdsc_name") == "TOTAL"
            ]
            require(
                len(totals) == 1,
                f"expected one ideal-cycle total: {ideal_cycles_path}",
            )
            ideal_cycles = totals[0]
        rows.append(
            {
                "directory": str(bundle.parent),
                "bundle_sha256": sha256(bundle),
                "roots": roots,
                "ideal_cycles": ideal_cycles,
                "ideal_cycles_path": (
                    str(ideal_cycles_path) if ideal_cycles_path.exists() else None
                ),
            }
        )
    return {"bundles": rows}


def pt_service(
    artifacts: dict[str, Any],
    trace: dict[str, Any],
    *,
    m: int,
    k: int,
    n: int,
    core_frequency_mhz: float,
) -> dict[str, Any]:
    """Relate compiler ideal PT cycles to measured one-BMM device service."""
    require(core_frequency_mhz > 0, "core frequency must be positive")
    flops = 2 * m * k * n
    rows: dict[str, Any] = {}
    for role in ("incumbent", "candidate"):
        event_names = trace[f"{role}_names"]
        require(len(event_names) == 1, f"{role} must have one kernel identity")
        event_name = next(iter(event_names))
        matches = [
            row
            for row in artifacts["bundles"]
            if Path(row["directory"]).name in event_name
        ]
        require(len(matches) == 1, f"cannot map {role} event to one bundle")
        ideal_cycles = matches[0]["ideal_cycles"]
        require(ideal_cycles is not None, f"{role} ideal cycles were not emitted")
        median_us = trace[role]["median_us"]
        ideal_us = ideal_cycles / core_frequency_mhz
        rows[role] = {
            "ideal_cycles": ideal_cycles,
            "ideal_us_at_core_frequency": ideal_us,
            "device_service_median_us": median_us,
            "actual_cycles_at_core_frequency": median_us * core_frequency_mhz,
            "ideal_cycle_over_device_service_percent": 100.0
            * ideal_us
            / median_us,
            "effective_tflops": flops / (median_us * 1e6),
        }
    return {
        "scope": (
            "compiler ideal PT cycles divided by one-BMM Kineto device service; "
            "this is a utilization proxy, not a hardware PT-active counter"
        ),
        "core_frequency_mhz": core_frequency_mhz,
        "flops": flops,
        "arms": rows,
    }


def parse_trace(path: Path, blocks: int) -> dict[str, Any]:
    trace = json.loads(path.read_text())
    events = [
        event
        for event in trace.get("traceEvents", [])
        if event.get("cat") == "kernel"
        and isinstance(event.get("dur"), (int, float))
    ]
    events.sort(key=lambda event: float(event.get("ts", 0.0)))
    expected_count = blocks * 4
    event_count_gate = len(events) == expected_count
    incumbent_events = [
        event
        for index, event in enumerate(events)
        if index % 4 in (0, 3)
    ]
    candidate_events = [
        event
        for index, event in enumerate(events)
        if index % 4 in (1, 2)
    ]
    incumbent_names = Counter(str(event.get("name")) for event in incumbent_events)
    candidate_names = Counter(str(event.get("name")) for event in candidate_events)
    incumbent_us = [float(event["dur"]) for event in incumbent_events]
    candidate_us = [float(event["dur"]) for event in candidate_events]
    order_gate = (
        event_count_gate
        and len(incumbent_events) == blocks * 2
        and len(candidate_events) == blocks * 2
        and len(incumbent_names) == 1
        and len(candidate_names) == 1
        and set(incumbent_names) != set(candidate_names)
    )
    positive_gate = all(
        float(event["dur"]) > 0
        for event in events
    )
    incumbent_stats = stats(incumbent_us) if incumbent_us else None
    candidate_stats = stats(candidate_us) if candidate_us else None
    speedup = (
        incumbent_stats["median_us"] / candidate_stats["median_us"]
        if incumbent_stats and candidate_stats
        else None
    )
    paired_ratios = []
    if event_count_gate:
        for block in range(blocks):
            offset = block * 4
            outer = (
                float(events[offset]["dur"]) + float(events[offset + 3]["dur"])
            ) / 2.0
            inner = (
                float(events[offset + 1]["dur"])
                + float(events[offset + 2]["dur"])
            ) / 2.0
            paired_ratios.append(outer / inner)
    return {
        "gate": event_count_gate and order_gate and positive_gate,
        "gates": {
            "event_count_exact": event_count_gate,
            "icci_order_by_identity": order_gate,
            "positive_durations": positive_gate,
        },
        "event_filter": {"cat": "kernel"},
        "event_count": len(events),
        "incumbent_names": dict(incumbent_names),
        "candidate_names": dict(candidate_names),
        "incumbent": incumbent_stats,
        "candidate": candidate_stats,
        "incumbent_over_candidate_median_speedup": speedup,
        "paired_block_speedup": stats(paired_ratios) if paired_ratios else None,
    }


def main() -> None:
    args = parse_args()
    require(args.cores == CORES, "first timing contract is exact for 32 cores")
    require(args.m > 0, "logical M must be positive")
    require(args.k % 64 == 0 and args.n % args.cores == 0, "unaligned shape")
    require(args.warmups > 0 and args.blocks > 0, "timing counts must be positive")
    if args.boundary == "compute-only":
        require(
            args.m % PHYSICAL_M == 0,
            "compute-only oracle requires M aligned to 64",
        )
        require(
            args.weight_layout == "per-arm",
            "compute-only oracle requires each arm's native W layout",
        )
        require(
            args.candidate_source == "manual",
            "compute-only oracle directly expresses both BMM schedules",
        )
    def explicit_work_division(
        role: str, splits: tuple[int, int, int]
    ) -> dict[str, int] | None:
        if not any(splits):
            return None
        require(
            args.boundary == "compute-only",
            f"explicit {role} splits are supported by the compute-only oracle",
        )
        require(
            all(split > 0 for split in splits),
            f"{role} M/N/K splits must all be positive",
        )
        require(
            math.prod(splits) == args.cores,
            f"{role} M/N/K split product must equal the core count",
        )
        return dict(zip(("M", "N", "K"), splits, strict=True))

    incumbent_work_div = explicit_work_division(
        "incumbent",
        (
            args.incumbent_m_split,
            args.incumbent_n_split,
            args.incumbent_k_split,
        ),
    )
    candidate_work_div = explicit_work_division(
        "candidate",
        (
            args.candidate_m_split,
            args.candidate_n_split,
            args.candidate_k_split,
        )
    )

    run_dir = args.run_dir.resolve()
    require(not run_dir.exists(), f"run directory exists: {run_dir}")
    cache = run_dir / "cache"
    run_dir.mkdir(parents=True)
    cache.mkdir()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    if args.boundary == "compute-only":
        os.environ.setdefault("SENPERFORMANCE", "2")

    import torch
    import torch.nn.functional as functional
    import torch_spyre
    import torch_spyre._C as extension
    try:
        from core.profiler import create_profiler
    except ModuleNotFoundError:
        from torch.profiler import ProfilerActivity, profile

        def create_profiler(
            torch_module: Any,
            trace_dir: str,
            *,
            profile_memory: bool,
            with_stack: bool,
        ) -> Any:
            return profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                profile_memory=profile_memory,
                with_stack=with_stack,
                on_trace_ready=torch_module.profiler.tensorboard_trace_handler(
                    trace_dir
                ),
            )
    from torch_spyre.model_utils import _dma_to_spyre_dim_order_swapped
    from torch_spyre._inductor import config as spyre_config
    from torch_spyre._inductor import spyre_hint
    from torch_spyre._inductor.propagate_hints import _reset_counter
    from torch_spyre._inductor.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
        reset as reset_named_dims,
    )

    torch_spyre._autoload()
    torch._dynamo.reset()
    imported = Path(torch_spyre.__file__).resolve()
    torch_root = Path(
        run_checked(
            ["git", "-C", str(imported.parent), "rev-parse", "--show-toplevel"]
        )
    )
    torch_head = run_checked(["git", "-C", str(torch_root), "rev-parse", "HEAD"])
    if args.expected_torch_head:
        require(
            torch_head == args.expected_torch_head,
            f"torch head {torch_head} != {args.expected_torch_head}",
        )

    generator = torch.Generator().manual_seed(args.seed)
    activation_cpu = (
        torch.randn((args.m, args.k), dtype=torch.float16, generator=generator)
        * 0.125
    )
    weight_cpu = (
        torch.randn((args.n, args.k), dtype=torch.float16, generator=generator)
        * 0.125
    )
    expected = functional.linear(activation_cpu, weight_cpu)
    device = torch.device("spyre")
    default_activation = activation_cpu.to(device)
    m_stick_activation = _dma_to_spyre_dim_order_swapped(activation_cpu)
    default_weight = weight_cpu.to(device)
    stationary_weight = _dma_to_spyre_dim_order_swapped(weight_cpu)
    if args.boundary == "compute-only":
        incumbent_activation = default_activation
        candidate_activation = m_stick_activation
        incumbent_weight = stationary_weight
        candidate_weight = default_weight
    elif args.weight_layout == "default":
        incumbent_activation = default_activation
        candidate_activation = default_activation
        incumbent_weight = default_weight
        candidate_weight = default_weight
    elif args.weight_layout == "stationary":
        incumbent_activation = default_activation
        candidate_activation = default_activation
        incumbent_weight = stationary_weight
        candidate_weight = stationary_weight
    else:
        incumbent_activation = default_activation
        candidate_activation = default_activation
        incumbent_weight = stationary_weight
        candidate_weight = default_weight

    class Incumbent(torch.nn.Module):
        def forward(self, activation: Any, weight: Any) -> Any:
            if incumbent_work_div is not None:
                with spyre_hint(work_div=incumbent_work_div):
                    return functional.linear(activation, weight)
            if args.work_division == "n32":
                with spyre_hint(core_order="row_major"):
                    with spyre_hint(work_div={"N": args.cores}):
                        return functional.linear(activation, weight)
            return functional.linear(activation, weight)

    class Candidate(torch.nn.Module):
        def forward(self, activation: Any, weight: Any) -> Any:
            if args.boundary == "compute-only":
                # Keep C in the BMM-native [N, M] layout. Correctness
                # normalization happens outside the measured device program.
                if candidate_work_div is not None:
                    if args.candidate_core_order == "row_major":
                        with spyre_hint(core_order="row_major"):
                            with spyre_hint(work_div=candidate_work_div):
                                return torch.matmul(
                                    weight, activation.transpose(-2, -1)
                                )
                    else:
                        with spyre_hint(work_div=candidate_work_div):
                            return torch.matmul(
                                weight, activation.transpose(-2, -1)
                            )
                return torch.matmul(weight, activation.transpose(-2, -1))
            if args.candidate_source == "selector":
                if args.work_division == "n32":
                    with spyre_hint(core_order="row_major"):
                        with spyre_hint(work_div={"N": args.cores}):
                            return functional.linear(activation, weight)
                return functional.linear(activation, weight)
            if args.work_division == "n32":
                with spyre_hint(core_order="row_major"):
                    with spyre_hint(work_div={"N": args.cores}):
                        padded_m = (
                            (args.m + PHYSICAL_M - 1) // PHYSICAL_M
                        ) * PHYSICAL_M
                        padded = functional.pad(
                            activation, (0, 0, 0, padded_m - args.m)
                        )
                        return torch.matmul(
                            weight, padded.transpose(-2, -1)
                        ).transpose(-2, -1)[: args.m]
            padded_m = (
                (args.m + PHYSICAL_M - 1) // PHYSICAL_M
            ) * PHYSICAL_M
            padded = functional.pad(activation, (0, 0, 0, padded_m - args.m))
            return torch.matmul(
                weight, padded.transpose(-2, -1)
            ).transpose(-2, -1)[: args.m]

    config = {
        "sencores": args.cores,
        "lx_planning": False,
        "lx_planner_relayout": False,
    }

    def prepare_named_dims(arm_activation: Any, arm_weight: Any) -> None:
        reset_named_dims()
        _reset_counter()
        declare_tensor_dim("M", args.m)
        declare_tensor_dim("K", args.k)
        declare_tensor_dim("N", args.n)
        name_tensor_dims(arm_activation, ["M", "K"])
        name_tensor_dims(arm_weight, ["N", "K"])

    def compile_arm(
        module: torch.nn.Module,
        dataflow: str,
        arm_activation: Any,
        arm_weight: Any,
    ) -> tuple[Callable[..., Any], Any]:
        prepare_named_dims(arm_activation, arm_weight)
        arm_config = {**config, "matmul_dataflow": dataflow}
        try:
            with spyre_config.patch(arm_config):
                compiled = torch.compile(module.to(device), fullgraph=True)
            with torch.no_grad(), spyre_config.patch(arm_config):
                output = compiled(arm_activation, arm_weight)
                torch.spyre.synchronize()
            return compiled, output
        finally:
            reset_named_dims()
            _reset_counter()

    incumbent, incumbent_output = compile_arm(
        Incumbent(),
        "weight_stationary",
        incumbent_activation,
        incumbent_weight,
    )
    candidate, candidate_output = compile_arm(
        Candidate(),
        (
            "activation_stationary"
            if args.candidate_source == "selector" and args.boundary == "paid"
            else "weight_stationary"
        ),
        candidate_activation,
        candidate_weight,
    )
    normalized_candidate_output = (
        candidate_output.transpose(-2, -1)
        if args.boundary == "compute-only"
        else candidate_output
    )
    candidate_expected = (
        expected.transpose(-2, -1)
        if args.boundary == "compute-only"
        else expected
    )
    correctness_rows = {
        "incumbent": correctness(torch, incumbent_output, expected),
        "candidate_native": correctness(
            torch, candidate_output, candidate_expected
        ),
        "candidate_matches_incumbent": correctness(
            torch, normalized_candidate_output, incumbent_output
        ),
    }
    correctness_gate = all(
        row["shape_exact"]
        and row["finite"]
        and row["allclose_rtol_5e2_atol_2_5e1"]
        for row in correctness_rows.values()
    )
    require(correctness_gate, f"correctness failed: {correctness_rows}")
    device_layouts = {
        "incumbent": {
            "activation": str(incumbent_activation.device_tensor_layout()),
            "weight": str(incumbent_weight.device_tensor_layout()),
            "output": str(incumbent_output.device_tensor_layout()),
        },
        "candidate": {
            "activation": str(candidate_activation.device_tensor_layout()),
            "weight": str(candidate_weight.device_tensor_layout()),
            "output_native": str(candidate_output.device_tensor_layout()),
        },
    }

    arms = (
        ("incumbent", incumbent, incumbent_activation, incumbent_weight),
        ("candidate", candidate, candidate_activation, candidate_weight),
        ("candidate", candidate, candidate_activation, candidate_weight),
        ("incumbent", incumbent, incumbent_activation, incumbent_weight),
    )
    with torch.no_grad(), spyre_config.patch(config):
        for _ in range(args.warmups):
            for _, arm, arm_activation, arm_weight in arms:
                arm(arm_activation, arm_weight)
                torch.spyre.synchronize()

    trace_dir = run_dir / "trace"
    trace_dir.mkdir()
    host_wall = {"incumbent": [], "candidate": []}
    profiler = create_profiler(
        torch, str(trace_dir), profile_memory=True, with_stack=False
    )
    profiler.start()
    with torch.no_grad(), spyre_config.patch(config):
        for _ in range(args.blocks):
            for label, arm, arm_activation, arm_weight in arms:
                started = time.perf_counter_ns()
                arm(arm_activation, arm_weight)
                torch.spyre.synchronize()
                host_wall[label].append(
                    (time.perf_counter_ns() - started) / 1e3
                )
                profiler.step()
    profiler.stop()
    traces = sorted(trace_dir.glob("*.pt.trace.json"))
    require(len(traces) == 1, f"expected one Kineto trace, found {traces}")
    trace_path = traces[0]

    trace = parse_trace(trace_path, args.blocks)
    artifacts = inventory(cache)
    observed_inventories = {
        tuple(row["roots"]) for row in artifacts["bundles"]
    }
    if args.boundary == "compute-only":
        structural_gate = len(artifacts["bundles"]) == 2 and all(
            row["roots"] == ["batchmatmul"] and row["ideal_cycles"] is not None
            for row in artifacts["bundles"]
        )
    else:
        expected_inventories = {
            ("ReStickifyOpHBM", "batchmatmul"),
            ("identity", "identity", "ReStickifyOpHBM", "batchmatmul"),
        }
        structural_gate = (
            observed_inventories == expected_inventories
            if args.weight_layout == "default"
            else len(observed_inventories) == 2
            and all("batchmatmul" in roots for roots in observed_inventories)
        )

    tracked = run_checked(
        [
            "git",
            "-C",
            str(torch_root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ]
    ).splitlines()
    pt_service_report = (
        pt_service(
            artifacts,
            trace,
            m=args.m,
            k=args.k,
            n=args.n,
            core_frequency_mhz=args.core_frequency_mhz,
        )
        if args.boundary == "compute-only" and trace["gate"] and structural_gate
        else None
    )
    report = {
        "schema": "activation_stationary_decode_abba_v2",
        "status": (
            "pass"
            if correctness_gate and trace["gate"] and structural_gate
            else "fail"
        ),
        "shape": {
            "logical_m": args.m,
            "physical_m": (
                (args.m + PHYSICAL_M - 1) // PHYSICAL_M
            )
            * PHYSICAL_M,
            "k": args.k,
            "n": args.n,
        },
        "candidate_source": args.candidate_source,
        "boundary": args.boundary,
        "work_division": args.work_division,
        "incumbent_work_division": incumbent_work_div or "auto",
        "candidate_work_division": candidate_work_div or "auto",
        "candidate_core_order": args.candidate_core_order,
        "weight_layout": args.weight_layout,
        "device_layouts": device_layouts,
        "order": ["incumbent", "candidate", "candidate", "incumbent"],
        "warmups": args.warmups,
        "blocks": args.blocks,
        "paid_boundary": {
            "inputs_ready_on_device": True,
            "candidate_padding_inside_event": args.boundary == "paid",
            "candidate_restickify_inside_event": args.boundary == "paid",
            "candidate_bmm_inside_event": True,
            "candidate_output_slice_inside_event": args.boundary == "paid",
            "per_arm_native_activation_layout": args.boundary == "compute-only",
            "per_arm_native_weight_layout": args.boundary == "compute-only",
            "bmm_native_output_layout": args.boundary == "compute-only",
            "host_to_device_excluded": True,
            "compile_excluded": True,
        },
        "correctness_gate": correctness_gate,
        "correctness": correctness_rows,
        "structural_gate": structural_gate,
        "artifacts": artifacts,
        "pt_service": pt_service_report,
        "timing": {
            "source": "Kineto cat==kernel complete-event duration",
            "trace": trace,
            "trace_path": str(trace_path),
            "trace_sha256": sha256(trace_path),
            "host_wall_diagnostic_only": {
                label: stats(values) for label, values in host_wall.items()
            },
        },
        "provenance": {
            "probe": str(Path(__file__).resolve()),
            "probe_sha256": sha256(Path(__file__).resolve()),
            "torch_version": torch.__version__,
            "torch_spyre_root": str(torch_root),
            "torch_spyre_head": torch_head,
            "torch_spyre_tracked_status": tracked,
            "extension": str(Path(extension.__file__).resolve()),
            "extension_sha256": sha256(Path(extension.__file__).resolve()),
            "deeptools_path": os.environ.get("DEEPTOOLS_PATH"),
            "dxp_standalone": run_checked(["which", "dxp_standalone"]),
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "SENARCH",
                    "SENCORES",
                    "SENPERFORMANCE",
                    "DT_OPT",
                    "DXP_DEBUG",
                    "DXP_LX_FRAC_AVAIL",
                    "DXP_BACKEND_LX_FRAC_AVAIL",
                )
            },
        },
    }
    write_json(run_dir / "summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

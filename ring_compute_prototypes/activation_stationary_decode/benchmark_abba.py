#!/usr/bin/env python3
"""Matched paid timing for incumbent and activation-stationary decode matmul.

Both arms use the same activation, W[N,K], 32-core budget, compiler stack,
device, process, and I-C-C-I launch order. Work ownership can be fixed to N32
or selected independently by the ordinary planner. The candidate pays for
explicit zero padding, activation restickify, BMM, and output slicing. Only
Kineto ``cat == "kernel"`` complete-event duration is performance evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
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
        rows.append(
            {
                "directory": str(bundle.parent),
                "bundle_sha256": sha256(bundle),
                "roots": roots,
            }
        )
    return {"bundles": rows}


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
    require(0 < args.m <= PHYSICAL_M, "logical M must be in [1,64]")
    require(args.k % 64 == 0 and args.n % args.cores == 0, "unaligned shape")
    require(args.warmups > 0 and args.blocks > 0, "timing counts must be positive")

    run_dir = args.run_dir.resolve()
    require(not run_dir.exists(), f"run directory exists: {run_dir}")
    cache = run_dir / "cache"
    run_dir.mkdir(parents=True)
    cache.mkdir()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)

    import torch
    import torch.nn.functional as functional
    import torch_spyre
    import torch_spyre._C as extension
    from core.profiler import create_profiler
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
    activation = activation_cpu.to(device)
    weight = weight_cpu.to(device)

    class Incumbent(torch.nn.Module):
        def forward(self, activation: Any, weight: Any) -> Any:
            if args.work_division == "n32":
                with spyre_hint(core_order="row_major"):
                    with spyre_hint(work_div={"N": args.cores}):
                        return functional.linear(activation, weight)
            return functional.linear(activation, weight)

    class Candidate(torch.nn.Module):
        def forward(self, activation: Any, weight: Any) -> Any:
            if args.candidate_source == "selector":
                if args.work_division == "n32":
                    with spyre_hint(core_order="row_major"):
                        with spyre_hint(work_div={"N": args.cores}):
                            return functional.linear(activation, weight)
                return functional.linear(activation, weight)
            if args.work_division == "n32":
                with spyre_hint(core_order="row_major"):
                    with spyre_hint(work_div={"N": args.cores}):
                        padded = functional.pad(
                            activation, (0, 0, 0, PHYSICAL_M - args.m)
                        )
                        return torch.matmul(
                            weight, padded.transpose(-2, -1)
                        ).transpose(-2, -1)[: args.m]
            padded = functional.pad(
                activation, (0, 0, 0, PHYSICAL_M - args.m)
            )
            return torch.matmul(
                weight, padded.transpose(-2, -1)
            ).transpose(-2, -1)[: args.m]

    config = {
        "sencores": args.cores,
        "lx_planning": False,
        "lx_planner_relayout": False,
        "matmul_activation_layout": "reduction",
        "test_preseeded_lx_relayout": False,
        "test_lx_relayout_preseed_only": False,
    }

    def prepare_named_dims() -> None:
        reset_named_dims()
        _reset_counter()
        declare_tensor_dim("M", args.m)
        declare_tensor_dim("K", args.k)
        declare_tensor_dim("N", args.n)
        name_tensor_dims(activation, ["M", "K"])
        name_tensor_dims(weight, ["N", "K"])

    def compile_arm(
        module: torch.nn.Module, dataflow: str
    ) -> tuple[Callable[..., Any], Any]:
        prepare_named_dims()
        arm_config = {**config, "matmul_dataflow": dataflow}
        try:
            with spyre_config.patch(arm_config):
                compiled = torch.compile(module.to(device), fullgraph=True)
            with torch.no_grad(), spyre_config.patch(arm_config):
                output = compiled(activation, weight)
                torch.spyre.synchronize()
            return compiled, output
        finally:
            reset_named_dims()
            _reset_counter()

    incumbent, incumbent_output = compile_arm(
        Incumbent(), "weight_stationary"
    )
    candidate, candidate_output = compile_arm(
        Candidate(),
        (
            "activation_stationary"
            if args.candidate_source == "selector"
            else "weight_stationary"
        ),
    )
    correctness_rows = {
        "incumbent": correctness(torch, incumbent_output, expected),
        "candidate": correctness(torch, candidate_output, expected),
        "candidate_matches_incumbent": correctness(
            torch, candidate_output, incumbent_output
        ),
    }
    correctness_gate = all(
        row["shape_exact"]
        and row["finite"]
        and row["allclose_rtol_5e2_atol_2_5e1"]
        for row in correctness_rows.values()
    )
    require(correctness_gate, f"correctness failed: {correctness_rows}")

    arms = (
        ("incumbent", incumbent),
        ("candidate", candidate),
        ("candidate", candidate),
        ("incumbent", incumbent),
    )
    with torch.no_grad(), spyre_config.patch(config):
        for _ in range(args.warmups):
            for _, arm in arms:
                arm(activation, weight)
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
            for label, arm in arms:
                started = time.perf_counter_ns()
                arm(activation, weight)
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
    expected_inventories = {
        ("ReStickifyOpHBM", "batchmatmul"),
        ("identity", "identity", "ReStickifyOpHBM", "batchmatmul"),
    }
    observed_inventories = {
        tuple(row["roots"]) for row in artifacts["bundles"]
    }
    structural_gate = observed_inventories == expected_inventories

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
    report = {
        "schema": "activation_stationary_decode_abba_v1",
        "status": (
            "pass"
            if correctness_gate and trace["gate"] and structural_gate
            else "fail"
        ),
        "shape": {
            "logical_m": args.m,
            "physical_m": PHYSICAL_M,
            "k": args.k,
            "n": args.n,
        },
        "candidate_source": args.candidate_source,
        "work_division": args.work_division,
        "order": ["incumbent", "candidate", "candidate", "incumbent"],
        "warmups": args.warmups,
        "blocks": args.blocks,
        "paid_boundary": {
            "inputs_ready_on_device": True,
            "candidate_padding_inside_event": True,
            "candidate_restickify_inside_event": True,
            "candidate_bmm_inside_event": True,
            "candidate_output_slice_inside_event": True,
            "host_to_device_excluded": True,
            "compile_excluded": True,
        },
        "correctness_gate": correctness_gate,
        "correctness": correctness_rows,
        "structural_gate": structural_gate,
        "artifacts": artifacts,
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

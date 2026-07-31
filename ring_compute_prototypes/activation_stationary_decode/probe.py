#!/usr/bin/env python3
"""Fail-closed device probe for activation-stationary Granite decode matmul.

Two graph forms compute the same ``linear(A, W)`` result with ``W[N, K]``:

``incumbent``
    ``A @ W.T``.  Stock PT mapping streams A West-to-East and parks W in XRF.

``activation_stationary``
    ``(W @ A.T).T``.  The transposed problem asks the same stock PT mapping to
    stream W West-to-East and park A.T in XRF.

``activation_stationary_padded64``
    Zero-pad logical decode M to the PT's physical 64 rows before applying the
    activation-stationary form, then slice the logical rows from the result.
    This makes the physical execution contract explicit instead of relying on
    implicit backend padding for the transposed problem.

The first experiment deliberately reuses the stock batchmatmul compiler and
freezes the same original-N 32-way output ownership in both arms.  It is not a
production graph rewrite and makes no performance claim unless the emitted
descriptor, dynamic correctness, and exact device timing gates all pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable


FP16_BYTES = 2
CORES = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "incumbent",
            "activation_stationary",
            "activation_stationary_padded64",
        ),
    )
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--cores", type=int, default=CORES)
    parser.add_argument("--seed", type=int, default=20260730)
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


def tensor_sha256(torch: Any, tensor: Any) -> str:
    value = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_checked(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            {
                "command": command,
                "returncode": completed.returncode,
                "output": completed.stdout,
            }
        )
    return completed.stdout


def objects(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def allocations(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(node)
        for node in objects(spec.get("dscs_", []))
        if node.get("nodeType_") == "allocate"
    ]


def transfers(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(node)
        for node in objects(spec.get("dscs_", []))
        if node.get("nodeType_") == "transfer"
    ]


def gtr_histogram(transfer: dict[str, Any]) -> dict[str, int]:
    rows = transfer.get("coreIdToGTRInfo_") or {}
    histogram = Counter()
    for value in rows.values():
        histogram[
            f"group={value.get('groupId_')},sharers={value.get('numSharers_')}"
        ] += 1
    return dict(sorted(histogram.items()))


def artifact_report(cache: Path) -> dict[str, Any]:
    paths = [
        path
        for path in cache.rglob("sdsc_*.json")
        if path.name.removeprefix("sdsc_").removesuffix(".json").isdigit()
    ]
    roots: list[dict[str, Any]] = []
    for path in sorted(paths):
        document = json.loads(path.read_text())
        require(len(document) == 1, f"expected one root in {path}")
        root_name, spec = next(iter(document.items()))
        dscs = spec.get("dscs_") or []
        body = next(iter(dscs[0].values())) if len(dscs) == 1 else {}
        root_transfers = transfers(spec)
        roots.append(
            {
                "root_name": root_name,
                "op": root_name.split("_", 1)[1],
                "file": str(path),
                "file_sha256": sha256(path),
                "num_cores": spec.get("numCoresUsed_"),
                "work_slices": copy.deepcopy(spec.get("numWkSlicesPerDim_")),
                "core_id_to_work_slice": copy.deepcopy(
                    spec.get("coreIdToWkSlice_")
                ),
                "logical_shape": copy.deepcopy(body.get("N_")),
                "primary_ds_info": copy.deepcopy(body.get("primaryDsInfo_")),
                "compute_inputs": copy.deepcopy(body.get("inputs_")),
                "compute_outputs": copy.deepcopy(body.get("outputs_")),
                "allocations": [
                    {
                        key: copy.deepcopy(row.get(key))
                        for key in (
                            "name_",
                            "ldsIdx_",
                            "component_",
                            "layoutDimOrder_",
                            "stickDimOrder_",
                            "stickSize_",
                        )
                    }
                    for row in allocations(spec)
                ],
                "transfers": [
                    {
                        "name": row.get("name_"),
                        "gtr_histogram": gtr_histogram(row),
                    }
                    for row in root_transfers
                ],
            }
        )
    roots.sort(key=lambda row: int(row["root_name"].split("_", 1)[0]))
    bundles = sorted(cache.rglob("bundle.mlir"))
    require(len(bundles) == 1, f"expected one bundle, found {bundles}")
    return {
        "roots": roots,
        "op_inventory": [root["op"] for root in roots],
        "bundle": str(bundles[0]),
        "bundle_sha256": sha256(bundles[0]),
        "bundle_token": bundles[0].parent.name,
    }


def compare(torch: Any, actual: Any, expected: Any) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    actual_f = actual_cpu.float()
    expected_f = expected_cpu.float()
    difference = (actual_f - expected_f).abs()
    return {
        "shape_exact": list(actual_cpu.shape) == list(expected_cpu.shape),
        "allclose_rtol_5e2_atol_2_5e1": bool(
            torch.allclose(actual_f, expected_f, rtol=5e-2, atol=2.5e-1)
        ),
        "actual_finite": bool(torch.isfinite(actual_f).all()),
        "expected_finite": bool(torch.isfinite(expected_f).all()),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "actual_sha256": tensor_sha256(torch, actual_cpu),
        "expected_sha256": tensor_sha256(torch, expected_cpu),
    }


def make_inputs(torch: Any, args: argparse.Namespace) -> dict[str, tuple[Any, Any]]:
    generator = torch.Generator().manual_seed(args.seed)
    activation = torch.randn(
        (args.m, args.k), dtype=torch.float16, generator=generator
    ) * 0.125
    weight = torch.randn(
        (args.n, args.k), dtype=torch.float16, generator=generator
    ) * 0.125
    poison_activation = activation.clone()
    poison_activation[:, ::17] += 0.5
    poison_weight = weight.clone()
    poison_weight[::19, ::23] -= 0.375
    return {
        "positive": (activation, weight),
        "poison_activation": (poison_activation, weight),
        "poison_weight": (activation, poison_weight),
    }


def main() -> None:
    args = parse_args()
    require(args.cores == CORES, "first contract is exact for 32 cores")
    require(args.m > 0 and args.k > 0 and args.n > 0, "shape must be positive")
    if args.mode == "activation_stationary_padded64":
        require(args.m <= 64, "first padded activation-stationary contract is M <= 64")
    require(args.k % 64 == 0 and args.n % 64 == 0, "K and N must be stick aligned")
    require(args.n % args.cores == 0, "N must divide across all cores")
    run_dir = args.run_dir.resolve()
    require(not run_dir.exists(), f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    cache = run_dir / "cache"
    cache.mkdir()
    write_json(
        run_dir / "request.json",
        {
            "mode": args.mode,
            "shape": {"m": args.m, "k": args.k, "n": args.n},
            "cores": args.cores,
            "algebra": (
                "A @ W.T"
                if args.mode == "incumbent"
                else (
                    "(W @ A.T).T"
                    if args.mode == "activation_stationary"
                    else "(W @ zero_pad_M64(A).T).T[:M]"
                )
            ),
            "timing": "not_requested",
        },
    )

    import torch
    import torch.nn.functional as torch_functional
    import torch_spyre
    import torch_spyre._C as extension
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
        ).strip()
    )
    torch_head = run_checked(["git", "-C", str(torch_root), "rev-parse", "HEAD"]).strip()
    if args.expected_torch_head:
        require(
            torch_head == args.expected_torch_head,
            f"torch head {torch_head} != expected {args.expected_torch_head}",
        )

    profiles = make_inputs(torch, args)
    expected = {
        name: torch_functional.linear(activation, weight)
        for name, (activation, weight) in profiles.items()
    }
    device = torch.device("spyre")
    device_profiles = {
        name: (activation.to(device), weight.to(device))
        for name, (activation, weight) in profiles.items()
    }

    class Graph(torch.nn.Module):
        def forward(self, activation: Any, weight: Any) -> Any:
            with spyre_hint(core_order="row_major"):
                with spyre_hint(work_div={"N": args.cores}):
                    if args.mode == "incumbent":
                        return torch_functional.linear(activation, weight)
                    if args.mode == "activation_stationary_padded64":
                        activation = torch_functional.pad(
                            activation, (0, 0, 0, 64 - args.m)
                        )
                    return torch.matmul(
                        weight, activation.transpose(-2, -1)
                    ).transpose(-2, -1)[: args.m]

    def prepare(profile: str) -> tuple[Any, Any]:
        reset_named_dims()
        _reset_counter()
        declare_tensor_dim("M", args.m)
        declare_tensor_dim("K", args.k)
        declare_tensor_dim("N", args.n)
        activation, weight = device_profiles[profile]
        name_tensor_dims(activation, ["M", "K"])
        name_tensor_dims(weight, ["N", "K"])
        return activation, weight

    config = {
        "sencores": args.cores,
        "lx_planning": False,
        "lx_planner_relayout": False,
        "matmul_activation_layout": "reduction",
        "test_preseeded_lx_relayout": False,
        "test_lx_relayout_preseed_only": False,
    }
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    prepared = prepare("positive")
    try:
        with spyre_config.patch(config):
            compiled = torch.compile(Graph().to(device), fullgraph=True)
        with torch.no_grad(), spyre_config.patch(config):
            positive = compiled(*prepared)
            torch.spyre.synchronize()
            positive_cpu = positive.cpu().clone()
    finally:
        reset_named_dims()
        _reset_counter()

    checks = {"positive": compare(torch, positive_cpu, expected["positive"])}
    for profile in ("poison_activation", "poison_weight"):
        with torch.no_grad(), spyre_config.patch(config):
            result = compiled(*device_profiles[profile])
            torch.spyre.synchronize()
            result_cpu = result.cpu().clone()
        checks[profile] = compare(torch, result_cpu, expected[profile])

    dynamic = {
        "poison_activation_changes_output": checks["poison_activation"][
            "actual_sha256"
        ]
        != checks["positive"]["actual_sha256"],
        "poison_weight_changes_output": checks["poison_weight"]["actual_sha256"]
        != checks["positive"]["actual_sha256"],
    }
    correctness_pass = all(
        row["shape_exact"]
        and row["allclose_rtol_5e2_atol_2_5e1"]
        and row["actual_finite"]
        and row["expected_finite"]
        for row in checks.values()
    ) and all(dynamic.values())

    artifacts = artifact_report(cache)
    bmm_roots = [root for root in artifacts["roots"] if root["op"] == "batchmatmul"]
    require(len(bmm_roots) == 1, f"expected one batchmatmul: {artifacts['op_inventory']}")
    bmm = bmm_roots[0]
    if args.mode == "incumbent":
        expected_shape = {"mb_": args.m, "in_": args.k, "out_": args.n}
    else:
        expected_shape = {
            "mb_": args.n,
            "in_": args.k,
            "out_": (
                64 if args.mode == "activation_stationary_padded64" else args.m
            ),
        }
    expected_inventory = (
        ["identity", "identity", "ReStickifyOpHBM", "batchmatmul"]
        if args.mode == "activation_stationary_padded64"
        else ["ReStickifyOpHBM", "batchmatmul"]
    )
    primary = bmm["primary_ds_info"]
    candidate_roles = {
        "weight_is_west_stream_input": (
            primary.get("INPUT", {}).get("layoutDimOrder_") == ["mb", "in"]
            and primary.get("INPUT", {}).get("stickDimOrder_") == ["in"]
        ),
        "activation_is_xrf_kernel": (
            primary.get("KERNEL", {}).get("layoutDimOrder_") == ["in", "out"]
            and primary.get("KERNEL", {}).get("stickDimOrder_") == ["out"]
        ),
        "original_n_is_bmm_mb": (
            bmm["logical_shape"].get("mb_") == args.n
            if args.mode != "incumbent"
            else True
        ),
    }
    structural = {
        "one_batchmatmul": artifacts["op_inventory"].count("batchmatmul") == 1,
        "expected_root_inventory": artifacts["op_inventory"] == expected_inventory,
        "no_shuffle_root": "shuffle" not in artifacts["op_inventory"],
        "logical_transposed_shape_exact": all(
            bmm["logical_shape"].get(key) == value
            for key, value in expected_shape.items()
        ),
        "all_32_cores": bmm["num_cores"] == args.cores,
        "original_n_is_32_way_owned": (
            bmm["work_slices"].get("out") == args.cores
            if args.mode == "incumbent"
            else bmm["work_slices"].get("mb") == args.cores
        ),
        **(
            candidate_roles
            if args.mode != "incumbent"
            else {}
        ),
    }
    structural_pass = all(structural.values())

    tracked = sorted(
        line
        for line in run_checked(
            [
                "git",
                "-C",
                str(torch_root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ]
        ).splitlines()
        if line
    )
    summary = {
        "status": "pass" if correctness_pass and structural_pass else "fail",
        "mode": args.mode,
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "correctness": checks,
        "dynamic_gates": dynamic,
        "correctness_pass": correctness_pass,
        "structural_gates": structural,
        "structural_pass": structural_pass,
        "artifacts": artifacts,
        "provenance": {
            "probe": str(Path(__file__).resolve()),
            "probe_sha256": sha256(Path(__file__).resolve()),
            "torch_version": torch.__version__,
            "torch_spyre_root": str(torch_root),
            "torch_spyre_head": torch_head,
            "torch_spyre_tracked_status": tracked,
            "extension": str(Path(extension.__file__).resolve()),
            "extension_sha256": sha256(Path(extension.__file__).resolve()),
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
        "evidence_boundary": {
            "device_timing": "not_measured",
            "granite_e2e": "not_run",
            "compiler_realization_beyond_source_sdsc": "not_yet_audited",
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

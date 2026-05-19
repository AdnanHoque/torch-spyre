#!/usr/bin/env python3
# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Probe one live Torch-Spyre run with an LX-neighbor restickify frame splice."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from restickify_lx_neighbor_frame_splice import splice
import restickify_scenario_probe as scenario_probe
from restickify_scenario_probe import CASES, _sync, _tensor_to_cpu

scenario_probe.torch = torch


def _case_by_name(name: str):
    for case in CASES:
        if case.name == name:
            return case
    raise ValueError(f"unknown case {name!r}")


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _compile_frame_probe(binary: Path, repo_root: Path, env: dict[str, str]) -> dict[str, Any]:
    if binary.exists():
        return {"status": "cached", "binary": str(binary)}
    include_root = Path(env.get("SPYRE_DEEPTOOLS_EXTRA_HEADERS", "/tmp/deeptools-headers-stage80"))
    cmd = [
        "g++",
        "-std=c++17",
        f"-I{include_root}",
        f"-I{env.get('DEEPTOOLS_INSTALL_DIR', '/opt/ibm/spyre/deeptools')}/include",
        str(repo_root / "tools" / "dcg_inpfetch_senprog_probe.cpp"),
        f"-L{env.get('DEEPTOOLS_INSTALL_DIR', '/opt/ibm/spyre/deeptools')}/lib",
        "-ldip",
        "-ldcg",
        "-ldcg_fe",
        "-ldcg_be",
        "-ldsc",
        "-ldpc",
        "-lsharedtools",
        "-lsgr",
        "-lutil",
        "-lcommon",
        "-ljson11",
        f"-Wl,-rpath,{env.get('DEEPTOOLS_INSTALL_DIR', '/opt/ibm/spyre/deeptools')}/lib",
        "-o",
        str(binary),
    ]
    row = _run(cmd, repo_root, env)
    row["binary"] = str(binary)
    if row["returncode"] != 0:
        raise RuntimeError("frame probe compile failed:\n" + row["stderr_tail"])
    return row


def _prepare_and_splice(
    code_dir: Path,
    *,
    work_root: Path,
    consumer_core_map: str,
    repo_root: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=True)
    stage_dir = work_root / f"{code_dir.name}_inpfetch"
    frame_dir = work_root / f"{code_dir.name}_ifn_frame"
    binary = Path(env.get("SPYRE_RESTICKIFY_LX_IFN_FRAME_PROBE_BIN", "/tmp/stage122-dcg-inpfetch-frame-probe"))

    compile_row = _compile_frame_probe(binary, repo_root, env)
    stage_row = _run(
        [
            sys.executable,
            str(repo_root / "tools" / "restickify_input_fetch_neighbor_probe.py"),
            "--code-dir",
            str(code_dir),
            "--output-dir",
            str(stage_dir),
            "--adapt-scheduled-lx-neighbor",
            "--alias-mb-out-to-ij-in",
            "--consumer-core-map",
            consumer_core_map,
            "--use-lx-neighbor-descriptor",
            "--fail-on-error",
        ],
        repo_root,
        env,
    )
    if stage_row["returncode"] != 0:
        raise RuntimeError("InputFetchNeighbor staging failed:\n" + stage_row["stderr_tail"])
    summary = json.loads((stage_dir / "input_fetch_neighbor_summary.json").read_text())
    row = summary["rows"][0]
    adapted = row["adapted_scheduled_lx_neighbor"]
    frame_row = _run(
        [
            str(binary),
            adapted["consumer"],
            adapted["producer"],
            str(frame_dir),
        ],
        repo_root,
        env,
    )
    if frame_row["returncode"] != 0:
        raise RuntimeError("InputFetchNeighbor frame generation failed:\n" + frame_row["stderr_tail"])
    splice_row = splice(code_dir, frame_dir, run_dxp_debug=True)
    return {
        "status": "patched",
        "code_dir": str(code_dir),
        "compile_frame_probe": compile_row,
        "stage": stage_row,
        "frame": frame_row,
        "splice": splice_row,
        "mapping_edges": row.get("run", {}).get("mapping_edges", []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="computed_transpose_adds_then_matmul")
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--device", default="spyre")
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output-dir", default="/tmp/stage122-lx-neighbor-runtime")
    parser.add_argument("--consumer-core-map", choices=("identity", "reverse"), default="reverse")
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()

    case = _case_by_name(args.case)
    dtype = getattr(torch, args.dtype)
    inputs, shape = case.input_builder(args.size, dtype)
    dev_inputs = tuple(arg.to(args.device) if hasattr(arg, "to") else arg for arg in inputs)
    patches: list[dict[str, Any]] = []

    from torch_spyre.execution import kernel_runner

    original_run = kernel_runner.SpyreSDSCKernelRunner.run

    def patched_run(self, *run_args, **kwargs):  # noqa: ANN001
        code_dir = Path(self.code_dir)
        descriptor = code_dir / "restickify_lx_neighbor_edges.json"
        if descriptor.exists() and not (code_dir / ".stage122_lx_neighbor_spliced").exists():
            payload = json.loads(descriptor.read_text(encoding="utf-8"))
            if payload.get("edges"):
                patch = _prepare_and_splice(
                    code_dir,
                    work_root=out_dir,
                    consumer_core_map=args.consumer_core_map,
                    repo_root=repo_root,
                    env=env,
                )
                patches.append(patch)
            else:
                patch = {
                    "status": "skipped",
                    "code_dir": str(code_dir),
                    "reason": "descriptor-has-no-edges",
                }
            (code_dir / ".stage122_lx_neighbor_spliced").write_text(
                json.dumps(patch, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return original_run(self, *run_args, **kwargs)

    start = time.perf_counter()
    try:
        kernel_runner.SpyreSDSCKernelRunner.run = patched_run
        compiled = torch.compile(case.fn, backend=args.backend, dynamic=False)
        result = compiled(*dev_inputs)
        _sync()
    finally:
        kernel_runner.SpyreSDSCKernelRunner.run = original_run
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    correctness = "skipped"
    if not args.skip_correctness:
        expected = case.fn(*inputs)
        actual = _tensor_to_cpu(result)
        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)
        correctness = "passed"

    summary = {
        "status": "ok",
        "case": args.case,
        "shape": shape,
        "size": args.size,
        "elapsed_ms": elapsed_ms,
        "correctness": correctness,
        "patch_count": len(patches),
        "patches": patches,
    }
    (out_dir / "runtime_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

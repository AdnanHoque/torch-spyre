#!/usr/bin/env python3
# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardware smoke for the chunked PT-LX restickify splice path.

This probe keeps the normal Torch-Spyre runtime tensor binding path intact. It
monkeypatches ``SpyreSDSCKernelRunner.run`` only long enough to detect a
generated bundle with chunked PT-LX sidecars, export those chunks, materialize a
single bridge frame, splice that frame over the stock ``ReStickifyOpHBM`` frame
in the same code directory, and then call the original runner with the original
arguments.

Use cautiously. This launches hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _case_by_name(cases: list[Any], name: str):
    for case in cases:
        if case.name == name:
            return case
    raise ValueError(f"unknown case {name!r}")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> dict[str, Any]:
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


def _has_chunked_sidecars(code_dir: Path) -> bool:
    return any(code_dir.glob("restickify_lx_neighbor_streaming_bridge_edge_*_chunk*.json"))


def _patch_code_dir(
    code_dir: Path,
    *,
    output_root: Path,
    repo_root: Path,
    env: dict[str, str],
    retries: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    patch_root = output_root / f"{code_dir.name}_chunked_ptlx"
    manifest_dir = patch_root / "export_manifest"
    bridge_dir = patch_root / "bridge_frame"
    splice_summary = patch_root / "splice_summary.json"
    patch_root.mkdir(parents=True, exist_ok=True)

    export_row = _run(
        [
            sys.executable,
            str(repo_root / "tools" / "restickify_chunked_sidecar_export.py"),
            "--sidecar-dir",
            str(code_dir),
            "--output-dir",
            str(manifest_dir),
            "--retries",
            str(retries),
            "--timeout-seconds",
            str(timeout_seconds),
            "--require-no-hbm",
            "--fail-on-error",
        ],
        cwd=repo_root,
        env=env,
    )
    if export_row["returncode"] != 0:
        raise RuntimeError(
            "chunked sidecar export failed:\n"
            + export_row["stdout_tail"]
            + export_row["stderr_tail"]
        )

    bridge_row = _run(
        [
            sys.executable,
            str(repo_root / "tools" / "restickify_chunked_bridge_frame.py"),
            "--manifest",
            str(manifest_dir / "manifest.json"),
            "--output-dir",
            str(bridge_dir),
            "--require-no-hbm",
        ],
        cwd=repo_root,
        env=env,
    )
    if bridge_row["returncode"] != 0:
        raise RuntimeError(
            "chunked bridge frame materialization failed:\n"
            + bridge_row["stdout_tail"]
            + bridge_row["stderr_tail"]
        )

    splice_row = _run(
        [
            sys.executable,
            str(repo_root / "tools" / "restickify_lx_bridge_same_artifact_splice.py"),
            "--code-dir",
            str(code_dir),
            "--bridge-frame-dir",
            str(bridge_dir),
            "--summary",
            str(splice_summary),
            "--require-hbm-free",
        ],
        cwd=repo_root,
        env=env,
    )
    if splice_row["returncode"] != 0:
        raise RuntimeError(
            "same-artifact chunked bridge splice failed:\n"
            + splice_row["stdout_tail"]
            + splice_row["stderr_tail"]
        )

    manifest = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    bridge = json.loads((bridge_dir / "summary.json").read_text(encoding="utf-8"))
    splice = json.loads(splice_summary.read_text(encoding="utf-8"))
    return {
        "status": "patched",
        "code_dir": str(code_dir),
        "patch_root": str(patch_root),
        "export": {
            "returncode": export_row["returncode"],
            "manifest": str(manifest_dir / "manifest.json"),
            "chunk_count": manifest.get("chunk_count"),
            "successful_chunks": manifest.get("successful_chunks"),
            "failed_chunks": manifest.get("failed_chunks"),
            "selected_token_totals": manifest.get("selected_token_totals"),
        },
        "bridge": {
            "returncode": bridge_row["returncode"],
            "summary": str(bridge_dir / "summary.json"),
            "frame": bridge.get("frame", {}),
        },
        "splice": {
            "returncode": splice_row["returncode"],
            "summary": str(splice_summary),
            "original_bytes": splice.get("original_bytes"),
            "patched_bytes": splice.get("patched_bytes"),
            "restickify_position": splice.get("restickify_position"),
            "restickify_original_bytes": splice.get("restickify_original_bytes"),
            "bridge_frame_bytes": splice.get("bridge_frame_bytes"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="matmul_then_add")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="spyre")
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--output-dir", default="/tmp/stage325-chunked-runtime-smoke")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--export-retries", type=int, default=5)
    parser.add_argument("--export-timeout-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = os.environ.copy()
    env.setdefault("SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR", "1")
    env.setdefault("SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED", "1")
    env.setdefault("SPYRE_RESTICKIFY_LX_NEIGHBOR_STREAMING_BRIDGE", "1")
    env.setdefault("SPYRE_RESTICKIFY_PTLX_NATIVE_VALIDGAP_ENDPOINT_TILE_E2E", "1")
    env.setdefault("SPYRE_RESTICKIFY_PTLX_NATIVE_VALIDGAP_ENDPOINT_TILE_CHUNK_SIZE", "0")
    env.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
    os.environ.update(env)

    import torch

    import restickify_scenario_probe as scenario_probe
    from restickify_scenario_probe import CASES, _sync, _tensor_to_cpu

    scenario_probe.torch = torch

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case = _case_by_name(CASES, args.case)
    dtype = getattr(torch, args.dtype)
    inputs, shape = case.input_builder(args.size, dtype)
    dev_inputs = tuple(arg.to(args.device) if hasattr(arg, "to") else arg for arg in inputs)
    patches: list[dict[str, Any]] = []

    from torch_spyre.execution import kernel_runner

    original_run = kernel_runner.SpyreSDSCKernelRunner.run

    def patched_run(self, *run_args, **kwargs):  # noqa: ANN001
        code_dir = Path(self.code_dir)
        marker = code_dir / ".stage325_chunked_ptlx_spliced"
        if _has_chunked_sidecars(code_dir) and not marker.exists():
            patch = _patch_code_dir(
                code_dir,
                output_root=output_dir,
                repo_root=repo_root,
                env=env,
                retries=args.export_retries,
                timeout_seconds=args.export_timeout_seconds,
            )
            patches.append(patch)
            marker.write_text(
                json.dumps(patch, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return original_run(self, *run_args, **kwargs)

    start = time.perf_counter()
    status = "ok"
    error = ""
    correctness: str | dict[str, Any] = "skipped"
    try:
        kernel_runner.SpyreSDSCKernelRunner.run = patched_run
        compiled = torch.compile(case.fn, backend=args.backend, dynamic=False)
        result = compiled(*dev_inputs)
        _sync()
        if not args.skip_correctness:
            expected = case.fn(*inputs)
            actual = _tensor_to_cpu(result)
            torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)
            correctness = "passed"
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = repr(exc)
        if not args.skip_correctness and isinstance(exc, AssertionError):
            correctness = {"status": "failed", "error": repr(exc)}
    finally:
        kernel_runner.SpyreSDSCKernelRunner.run = original_run
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    summary = {
        "status": status,
        "error": error,
        "case": args.case,
        "size": args.size,
        "shape": shape,
        "elapsed_ms": elapsed_ms,
        "correctness": correctness,
        "patch_count": len(patches),
        "patches": patches,
    }
    _write_json(output_dir / "runtime_smoke_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

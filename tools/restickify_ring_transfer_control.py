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

"""Run a Deeptools control that emits HBM-free core-to-core ring traffic.

This is a diagnostic companion for the restickify locality work. It does not
lower a Torch-Spyre restickify. Instead, it runs Deeptools'
``UnicastTrafficGen`` built-in and records the generated ``senprog.txt`` token
signature. The useful control property is:

    HBM == 0 and L3LU/L3SU > 0

On AIU, L3LU/L3SU are the ring-facing load/store units. For a core-to-core
LX-to-LX transfer over RIU, the program is expected to use L3LU/L3SU for the
ring leg, while avoiding HBM instructions entirely.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


_TOKENS = ("HBM", "L3LU", "L3SU", "LXLU", "LXSU", "SFP", "PT")


def _find_dcg_standalone(explicit: str | None) -> str | None:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("DCG_STANDALONE")
    if env:
        candidates.append(env)
    deeptools = os.environ.get("DEEPTOOLS_INSTALL_DIR")
    if deeptools:
        candidates.append(str(Path(deeptools) / "bin" / "dcg_standalone"))
    candidates.append("/opt/ibm/spyre/deeptools/bin/dcg_standalone")
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return shutil.which("dcg_standalone")


def _count_senprog_tokens(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {token: text.count(token) for token in _TOKENS}


def _producer_consumer_edges(stdout: str) -> list[dict[str, int]]:
    edges: list[dict[str, int]] = []
    for line in stdout.splitlines():
        match = re.match(r"\s*(\d+)\s*-->\s*\[\s*(\d+)\s*\]\s*$", line)
        if match:
            edges.append(
                {
                    "producer_core": int(match.group(1)),
                    "consumer_core": int(match.group(2)),
                }
            )
    return edges


def _first_matching_lines(path: Path, pattern: str, limit: int) -> list[str]:
    if not path.exists():
        return []
    regex = re.compile(pattern)
    matches: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if regex.search(line):
            matches.append(line)
            if len(matches) >= limit:
                break
    return matches


def _run_unicast(args: argparse.Namespace) -> dict[str, Any]:
    binary = _find_dcg_standalone(args.dcg_standalone)
    if not binary:
        raise SystemExit("dcg_standalone not found; set DEEPTOOLS_INSTALL_DIR or --dcg-standalone")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "dataDSC"
    if data_dir.exists() and args.clean:
        shutil.rmtree(data_dir)

    cmd = [
        binary,
        "-o",
        "UnicastTrafficGen",
        "-numCores",
        str(args.num_cores),
        "-i_size",
        str(args.i_size),
        "-d",
        str(data_dir),
    ]
    if args.emit_senprog:
        cmd.append("-s")
    cmd.extend(["-v", str(args.verbose)])

    result = subprocess.run(
        cmd,
        cwd=out_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    stdout_path = out_dir / "stdout.txt"
    stderr_path = out_dir / "stderr.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")

    senprog = data_dir / "senprog.txt"
    token_counts = _count_senprog_tokens(senprog)
    l3_ring_tokens = token_counts.get("L3LU", 0) + token_counts.get("L3SU", 0)
    local_lx_tokens = token_counts.get("LXLU", 0) + token_counts.get("LXSU", 0)
    hbm_tokens = token_counts.get("HBM", 0)

    summary: dict[str, Any] = {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "data_dir": str(data_dir),
        "senprog": str(senprog) if senprog.exists() else "",
        "senprog_token_counts": token_counts,
        "producer_consumer_edges": _producer_consumer_edges(result.stdout),
        "hbm_free": hbm_tokens == 0,
        "has_ring_facing_l3_transfer": l3_ring_tokens > 0,
        "has_local_lx_transfer_tokens": local_lx_tokens > 0,
        "interpretation": (
            "HBM-free core-to-core RIU transfer control"
            if result.returncode == 0 and hbm_tokens == 0 and l3_ring_tokens > 0
            else "control did not produce the expected HBM-free L3 ring signature"
        ),
        "sample_l3_ring_lines": _first_matching_lines(senprog, r"L3_(?:LDU|STU)|ringDT", 12),
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/tmp/restickify-ring-transfer-control")
    parser.add_argument("--dcg-standalone", default=None)
    parser.add_argument("--num-cores", type=int, default=8)
    parser.add_argument("--i-size", type=int, default=64)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--emit-senprog", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    summary = _run_unicast(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["returncode"] != 0:
        raise SystemExit(summary["returncode"])


if __name__ == "__main__":
    main()

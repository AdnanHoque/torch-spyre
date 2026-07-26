#!/usr/bin/env python3
# Copyright 2026 The Torch-Spyre Authors.
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

"""Summarize allocation, plan, and emitted shuffle evidence for one source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def jsonl(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--source", default="buf29")
    parser.add_argument("--consumer", default="buf31")
    args = parser.parse_args()

    destination = f"__spyre_lx_relayout_destination__:{args.source}"
    found_plan = []
    for row in jsonl(args.run / "relayout_plans.jsonl") or []:
        for plan in row if isinstance(row, list) else [row]:
            if plan.get("source_name") == args.source:
                found_plan.append(plan)

    allocations = []
    for row in jsonl(args.run / "allocations.jsonl") or []:
        by_name = {item["name"]: item for item in row.get("buffers", [])}
        if args.source in by_name:
            allocations.append(
                {
                    "source": by_name.get(args.source),
                    "destination": by_name.get(destination),
                    "relayout_sources": row.get("relayout_sources", []),
                }
            )

    shuffles = sorted(args.run.glob("origsdsc_debug_*shuffle*.json"))
    print(json.dumps({"plans": found_plan, "allocations": allocations}, indent=2))
    print(f"emitted_shuffle_files={len(shuffles)}")
    for path in shuffles:
        text = path.read_text(encoding="utf-8", errors="replace")
        print(f"  {path.name}: STCDPOpLx={'STCDPOpLx' in text}")

    allocated = any(
        row["source"].get("address") is not None
        and row["destination"] is not None
        and row["destination"].get("address") is not None
        for row in allocations
    )
    exact_plan = any(plan.get("consumer_name") == args.consumer for plan in found_plan)
    return 0 if allocated and exact_plan and len(shuffles) == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Verify that the explicit SHUFFLE occupies the bundle slot before BMM."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root_op(path: Path) -> str:
    payload = json.loads(path.read_text())
    root = next(iter(payload.values()))
    dsc = next(iter(root["dscs_"][0].values()))
    return dsc["computeOp_"][0]["opFuncName"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration-dir", type=Path, required=True)
    parser.add_argument("--authoritative-shuffle", type=Path, required=True)
    parser.add_argument("--lowered-sdsc", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()

    bundle = (args.integration_dir / "bundle.mlir").read_text()
    filenames = re.findall(r'sdsc_filename = "([^"]+)"', bundle)
    operations = [root_op(args.integration_dir / name) for name in filenames]
    if operations != ["ReStickifyOpHBM", "shuffle", "batchmatmul"]:
        raise AssertionError(f"unexpected bundle order: {operations}")

    shuffle_path = args.integration_dir / filenames[1]
    if sha256(shuffle_path) != sha256(args.authoritative_shuffle):
        raise AssertionError("integration SHUFFLE differs from authoritative fixture")

    lowered = next(iter(json.loads(args.lowered_sdsc.read_text()).values()))
    schedules = lowered["coreIdToDscSchedule"]
    for core in range(32):
        rows = [int(step[0]) for step in schedules[str(core)]]
        if rows != list(range(8)):
            raise AssertionError(f"core {core}: bounded rows are not ordered 0..7: {rows}")

    report = {
        "status": "pass",
        "bundle_sequence": [
            {"index": index, "filename": name, "operation": operation}
            for index, (name, operation) in enumerate(zip(filenames, operations))
        ],
        "authoritative_shuffle_sha256": sha256(args.authoritative_shuffle),
        "integration_shuffle_sha256": sha256(shuffle_path),
        "bounded_rows_per_core": list(range(8)),
        "ordering": "producer -> eight bounded SHUFFLE rows -> consumer batchmatmul",
    }
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(
        """# Variant A bundle-order verification

Status: **pass**

| Bundle index | File | Operation |
|---:|---|---|
| 0 | `sdsc_0.json` | `ReStickifyOpHBM` producer |
| 1 | `sdsc_1.json` | explicit `SHUFFLE` marker |
| 2 | `sdsc_2.json` | consumer `batchmatmul` |

The integration SHUFFLE is byte-identical to the authoritative SHUFFLE-only
fixture. DXP replaces that bundle slot with data rows `0..7` on every core.
Because `sdsc_execute` operations are ordered in the bundle, all eight bounded
rows complete before the following consumer `batchmatmul` begins.
"""
    )


if __name__ == "__main__":
    main()

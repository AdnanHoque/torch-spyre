#!/usr/bin/env python3
"""Run the full oracle structural/negative gates under one placement arm."""

from __future__ import annotations

import json
import os
from pathlib import Path

import full_attention_prefix_structural_probe as base

from joint_oracle_target_patch import install


def main() -> None:
    contract = install(base)
    base.main()
    path = Path(os.environ["FULL_PREFIX_RUN_DIR"]) / "report.json"
    report = json.loads(path.read_text())
    report["factorial_placement_contract"] = contract
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

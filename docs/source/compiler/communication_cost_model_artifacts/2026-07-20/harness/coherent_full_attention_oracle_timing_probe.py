#!/usr/bin/env python3
"""Run one timing cell with default or coherent regional placement."""

from __future__ import annotations

import json
import os
from pathlib import Path

import full_attention_oracle_timing_probe as base

from coherent_joint_target_patch import install


def main() -> None:
    contract = install(base)
    base.main()
    path = Path(os.environ["FULL_ATTN_RUN_DIR"]) / "summary.json"
    summary = json.loads(path.read_text())
    summary["coherent_placement_contract"] = contract
    path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

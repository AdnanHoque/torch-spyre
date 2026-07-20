#!/usr/bin/env python3
"""Adversarial V/head/query gate for the coherent regional placement."""

from __future__ import annotations

import json
import os
from pathlib import Path

import full_attention_prefix_structural_probe as base

from coherent_joint_target_patch import install


def main() -> None:
    original_create_tensors = base.create_tensors

    def create_vcoded(*args, **kwargs):
        query, key, value = original_create_tensors(*args, **kwargs)
        token = base.torch.linspace(
            -0.5, 0.5, value.size(2), dtype=base.torch.float32
        ).view(1, 1, value.size(2), 1)
        channel = (
            base.torch.arange(value.size(3), dtype=base.torch.float32) % 8
        ).view(1, 1, 1, value.size(3))
        head = base.torch.tensor(
            [-0.75, -0.25, 0.25, 0.75], dtype=base.torch.float32
        ).view(1, value.size(1), 1, 1)
        coded = (head + 0.25 * token + 0.01 * channel).to(value.dtype)
        return query, key, coded.contiguous()

    base.create_tensors = create_vcoded
    contract = install(base)
    base.main()
    path = Path(os.environ["FULL_PREFIX_RUN_DIR"]) / "report.json"
    report = json.loads(path.read_text())
    report["coherent_placement_contract"] = contract
    report["value_pattern"] = (
        "V encodes head level, K-token ramp, and channel modulo 8; "
        "Q and K retain the base probe's deterministic +/-2 pattern"
    )
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()


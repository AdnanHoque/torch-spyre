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

"""Promotion gates for device-backed on-chip SDPA sweep matrices."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = REPO_ROOT / "tools" / "onchip_sdpa_sweep.py"


@dataclass(frozen=True)
class GateCase:
    name: str
    batch: int
    heads: int
    dim: int
    block_size: int
    lengths: tuple[int, ...]
    min_mixed_by_length: dict[int, int]
    layout_xform_lengths: tuple[int, ...]
    is_causal: bool = False


ONCHIP_LAYOUT_XFORM_CASES = (
    GateCase(
        name="b1h2d64_block64",
        batch=1,
        heads=2,
        dim=64,
        block_size=64,
        lengths=(64, 128, 256, 384, 512),
        min_mixed_by_length={64: 6, 128: 9, 256: 19, 384: 29, 512: 39},
        layout_xform_lengths=(128, 256, 384, 512),
    ),
    GateCase(
        name="b2h2d64_block64",
        batch=2,
        heads=2,
        dim=64,
        block_size=64,
        lengths=(128, 256),
        min_mixed_by_length={128: 7, 256: 15},
        layout_xform_lengths=(128, 256),
    ),
    GateCase(
        name="b1h2d64_block64_causal",
        batch=1,
        heads=2,
        dim=64,
        block_size=64,
        lengths=(128, 256),
        min_mixed_by_length={128: 8, 256: 16},
        layout_xform_lengths=(128, 256),
        is_causal=True,
    ),
    GateCase(
        name="b2h4d128_block64",
        batch=2,
        heads=4,
        dim=128,
        block_size=64,
        lengths=(128, 256),
        min_mixed_by_length={128: 7, 256: 15},
        layout_xform_lengths=(128, 256),
    ),
    GateCase(
        name="b1h4d64_block64",
        batch=1,
        heads=4,
        dim=64,
        block_size=64,
        lengths=(128, 256),
        min_mixed_by_length={128: 9, 256: 19},
        layout_xform_lengths=(128, 256),
    ),
    GateCase(
        name="b1h2d128_block64",
        batch=1,
        heads=2,
        dim=128,
        block_size=64,
        lengths=(128, 256),
        min_mixed_by_length={128: 9, 256: 19},
        layout_xform_lengths=(128, 256),
    ),
    GateCase(
        name="b1h2d64_block128",
        batch=1,
        heads=2,
        dim=64,
        block_size=128,
        lengths=(128, 256, 512),
        min_mixed_by_length={128: 3, 256: 9, 512: 19},
        layout_xform_lengths=(256, 512),
    ),
    GateCase(
        name="b1h2d64_block64_long",
        batch=1,
        heads=2,
        dim=64,
        block_size=64,
        lengths=(768, 1024),
        min_mixed_by_length={768: 59, 1024: 78},
        layout_xform_lengths=(768, 1024),
    ),
)


GATES = {
    "onchip_layout_xform": ONCHIP_LAYOUT_XFORM_CASES,
}


def _parse_csv(values: str) -> list[str]:
    return [value.strip() for value in values.split(",") if value.strip()]


def select_cases(gate: str, requested: str) -> list[GateCase]:
    cases = list(GATES[gate])
    if requested == "all":
        return cases
    by_name = {case.name: case for case in cases}
    selected = []
    for name in _parse_csv(requested):
        if name not in by_name:
            raise ValueError(f"unknown case {name!r}; valid={sorted(by_name)}")
        selected.append(by_name[name])
    return selected


def case_output_path(output_dir: Path, gate: str, case: GateCase) -> Path:
    return output_dir / f"{gate}-{case.name}.json"


def sweep_command(
    *,
    python: str,
    variant: str,
    case: GateCase,
    warmup: int,
    iters: int,
    timeout_s: float,
    cache_prefix: str,
    output_json: Path,
    seed: int,
    atol: float,
    rtol: float,
) -> list[str]:
    cmd = [
        python,
        str(SWEEP_SCRIPT),
        "--lengths",
        ",".join(str(length) for length in case.lengths),
        "--variants",
        variant,
        "--batch",
        str(case.batch),
        "--heads",
        str(case.heads),
        "--dim",
        str(case.dim),
        "--block-size",
        str(case.block_size),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--timeout-s",
        str(timeout_s),
        "--cache-prefix",
        f"{cache_prefix}-{case.name}",
        "--output-json",
        str(output_json),
        "--seed",
        str(seed),
        "--atol",
        str(atol),
        "--rtol",
        str(rtol),
    ]
    if case.is_causal:
        cmd.append("--is-causal")
    return cmd


def _has_layout_xform_consumer(row: dict) -> bool:
    for mixed in row.get("mixed_sdscs", []):
        flash = mixed.get("flash_pipeline") or {}
        if flash.get("layout_xform_pair_role") == "consumer":
            return True
    return False


def validate_rows(
    rows: list[dict],
    *,
    case: GateCase,
    variant: str,
    max_error: float,
    require_layout_xform: bool = True,
) -> list[str]:
    errors = []
    by_length = {}
    for row in rows:
        shape = row.get("shape") or {}
        if row.get("variant") == variant:
            by_length[shape.get("length")] = row

    for length in case.lengths:
        row = by_length.get(length)
        if row is None:
            errors.append(f"{case.name}: missing {variant} row for L={length}")
            continue
        shape = row.get("shape") or {}
        expected_shape = {
            "batch": case.batch,
            "heads": case.heads,
            "length": length,
            "dim": case.dim,
        }
        if shape != expected_shape:
            errors.append(f"{case.name}: L={length} shape={shape} expected={expected_shape}")
        if row.get("status") != "ok":
            errors.append(f"{case.name}: L={length} status={row.get('status')!r}")
        if row.get("block_size") != case.block_size:
            errors.append(
                f"{case.name}: L={length} block_size={row.get('block_size')} "
                f"expected={case.block_size}"
            )
        if row.get("is_causal") != case.is_causal:
            errors.append(
                f"{case.name}: L={length} is_causal={row.get('is_causal')} "
                f"expected={case.is_causal}"
            )
        max_abs_error = row.get("max_abs_error")
        if max_abs_error is None or max_abs_error > max_error:
            errors.append(
                f"{case.name}: L={length} max_abs_error={max_abs_error} "
                f"limit={max_error}"
            )
        mixed_count = len(row.get("mixed_sdscs", []))
        min_mixed = case.min_mixed_by_length[length]
        if mixed_count < min_mixed:
            errors.append(
                f"{case.name}: L={length} mixed={mixed_count} expected>={min_mixed}"
            )
        if (
            require_layout_xform
            and length in case.layout_xform_lengths
            and not _has_layout_xform_consumer(row)
        ):
            errors.append(f"{case.name}: L={length} missing layout-xform consumer")
    return errors


def _read_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _run_gate(args: argparse.Namespace) -> int:
    cases = select_cases(args.gate, args.cases)
    output_dir = Path(args.case_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    errors = []
    for case in cases:
        case_json = case_output_path(output_dir, args.gate, case)
        cmd = sweep_command(
            python=args.python,
            variant=args.variant,
            case=case,
            warmup=args.warmup,
            iters=args.iters,
            timeout_s=args.timeout_s,
            cache_prefix=args.cache_prefix,
            output_json=case_json,
            seed=args.seed,
            atol=args.atol,
            rtol=args.rtol,
        )
        print(shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        if not args.reuse_existing:
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                errors.append(f"{case.name}: sweep command returned {proc.returncode}")
        if not case_json.exists():
            errors.append(f"{case.name}: missing output json {case_json}")
            continue
        rows = _read_rows(case_json)
        all_rows.extend(rows)
        errors.extend(
            validate_rows(
                rows,
                case=case,
                variant=args.variant,
                max_error=args.max_error,
                require_layout_xform=not args.no_require_layout_xform,
            )
        )

    if args.output_json and not args.dry_run:
        Path(args.output_json).write_text(json.dumps(all_rows, indent=2, sort_keys=True))

    if errors:
        print("PROMOTION_GATE_FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    if not args.dry_run:
        print(
            f"PROMOTION_GATE_PASSED gate={args.gate} cases={len(cases)} "
            f"rows={len(all_rows)}",
            flush=True,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=sorted(GATES), default="onchip_layout_xform")
    parser.add_argument("--cases", default="all", help="'all' or comma-separated case names")
    parser.add_argument("--variant", default="onchip_master_layout_xform")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=480.0)
    parser.add_argument("--seed", type=int, default=0xA771)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--max-error", type=float, default=0.01)
    parser.add_argument("--cache-prefix", default="/tmp/sdpa-onchip-layout-xform-gate")
    parser.add_argument(
        "--case-output-dir",
        default="/tmp/sdpa-onchip-layout-xform-gate-json",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="validate existing per-case JSON files instead of rerunning sweeps",
    )
    parser.add_argument("--no-require-layout-xform", action="store_true")
    args = parser.parse_args(argv)
    return _run_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())

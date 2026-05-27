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

"""Run an isolated SDPA sweep for vanilla, flash-HBM, and on-chip variants.

The parent process spawns one child process per shape/variant so Torch-Spyre
configuration flags are read from a clean environment and the device runtime is
not reused across variants.  Each child compiles one SDPA shape, checks value
correctness against PyTorch CPU, measures repeated compiled Spyre calls, and
summarizes generated mixed SDSCs / senprog tokens from ``TORCHINDUCTOR_CACHE_DIR``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path


TAIL_CHARS = 8000


BASE_VARIANT_ENV = {
    "SPYRE_FLASH_ATTENTION_PREFILL": "0",
    "SPYRE_FLASH_ATTENTION_ONCHIP_SDPA": "0",
    "SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM": "0",
    "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "0",
    "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP": "0",
    "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_ARTIFACT": "0",
    "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE": "-1",
    "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_VALUE_FLOW_TILE": "-1",
    "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PAIR_TILE": "-1",
    "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PREFIX_FORCE": "0",
    "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE": "-1",
    "SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF": "0",
    "SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF": "0",
    "SPYRE_ONCHIP_HANDOFF_REALIZE": "0",
    "SPYRE_ONCHIP_ATTENTION_SCORE_HANDOFF": "0",
}


VARIANT_ENV = {
    "vanilla": {
        **BASE_VARIANT_ENV,
    },
    "flash_hbm": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
    },
    "pointwise_only": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
        "SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF": "1",
        "SPYRE_ONCHIP_HANDOFF_MIN_BYTES": "0",
    },
    "onchip": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
        "SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF": "1",
        "SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF": "1",
        "SPYRE_ONCHIP_HANDOFF_MIN_BYTES": "0",
    },
    "onchip_master": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_ONCHIP_SDPA": "1",
        "SPYRE_ONCHIP_HANDOFF_MIN_BYTES": "0",
    },
    "onchip_master_layout_xform": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_ONCHIP_SDPA": "1",
        "SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM": "1",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE": None,
        "SPYRE_ONCHIP_HANDOFF_MIN_BYTES": "0",
    },
    "warp_overlap_probe": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE": "0",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP": "1",
    },
    "warp_ifn_prefix_probe": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE": "0",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP": "1",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PREFIX_FORCE": "1",
    },
    "value_flow_tile0": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_VALUE_FLOW_TILE": "0",
    },
    "value_flow_tile1": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_VALUE_FLOW_TILE": "1",
    },
    "value_flow_tile2": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_VALUE_FLOW_TILE": "2",
    },
    "ifn_pair_tile0": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PAIR_TILE": "0",
    },
    "ifn_pair_tile1": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PAIR_TILE": "1",
    },
    "ifn_pair_tile2": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PAIR_TILE": "2",
    },
    "layout_xform_pair_tile2": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE": "2",
    },
    "layout_xform_pair_auto": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE": "-2",
    },
    "onchip_layout_xform": {
        **BASE_VARIANT_ENV,
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE": "1",
        "SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF": "1",
        "SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF": "1",
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE": "-2",
        "SPYRE_ONCHIP_HANDOFF_MIN_BYTES": "0",
    },
}


def _parse_csv(values: str) -> list[str]:
    return [value.strip() for value in values.split(",") if value.strip()]


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def _summarize_cache(cache_dir: Path) -> dict:
    mixed = []
    for path in sorted(cache_dir.rglob("sdsc_*.json")):
        if "/debug/" in str(path):
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or not data:
            continue
        name, body = next(iter(data.items()))
        if not isinstance(body, dict):
            continue
        opfuncs = body.get("opFuncsUsed_", []) or []
        datadscs = body.get("datadscs_", []) or []
        if not opfuncs and not datadscs:
            continue
        flash_pipeline = body.get("flashAttentionPipeline_")
        first_dataop = None
        if datadscs:
            dataop_name, dataop_body = next(iter(datadscs[0].items()))
            first_labeled = (dataop_body.get("labeledDs_") or [{}])[0]
            first_piece = (first_labeled.get("PieceInfo") or [{}])[0]
            placement = (first_piece.get("PlacementInfo") or [{}])[0]
            first_dataop = {
                "name": dataop_name,
                "layout": first_labeled.get("layoutDimOrder_"),
                "stick": first_labeled.get("stickDimOrder_"),
                "startAddr": placement.get("startAddr"),
                "memId": placement.get("memId"),
            }
        mixed.append(
            {
                "file": str(path.relative_to(cache_dir)),
                "name": name,
                "opFuncsUsed": opfuncs,
                "datadscs": len(datadscs),
                "first_dataop": first_dataop,
                "flash_pipeline": flash_pipeline,
            }
        )

    senprog = []
    for path in sorted(cache_dir.rglob("senprog.txt")):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        rel = str(path.parent.relative_to(cache_dir))
        if "debug/" not in rel:
            continue
        counts = {
            token: text.count(token)
            for token in ("HBM", "L3_LDU", "L3_STU", "LX_LDSTU", "PT", "SFP")
        }
        if any(counts.values()):
            senprog.append({"dir": rel, **counts})

    return {"mixed_sdscs": mixed, "senprog": senprog}


def _run_child(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F
    import torch_spyre  # noqa: F401
    from torch_spyre._inductor import config as spyre_config
    from torch_spyre.ops.fallbacks import FallbackWarning

    if args.forbid_fallbacks:
        warnings.simplefilter("error", FallbackWarning)

    torch.manual_seed(args.seed)
    try:
        torch._dynamo.reset_code_caches()
        torch._inductor.codecache.FxGraphCache.clear()
    except AttributeError:
        pass

    shape = (args.batch, args.heads, args.length, args.dim)
    q_cpu = torch.randn(shape, dtype=torch.float16)
    k_cpu = torch.randn(shape, dtype=torch.float16)
    v_cpu = torch.randn(shape, dtype=torch.float16)

    def sdpa(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=args.is_causal)

    reference = sdpa(q_cpu, k_cpu, v_cpu)
    q_dev = q_cpu.to("spyre")
    k_dev = k_cpu.to("spyre")
    v_dev = v_cpu.to("spyre")

    compiled = torch.compile(sdpa, backend="inductor")

    compile_start = time.perf_counter()
    result = compiled(q_dev, k_dev, v_dev)
    torch.spyre.synchronize()
    compile_run_ms = (time.perf_counter() - compile_start) * 1000.0

    result_cpu = result.cpu()
    torch.testing.assert_close(
        result_cpu,
        reference,
        equal_nan=True,
        atol=args.atol,
        rtol=args.rtol,
    )
    max_abs_error = float((result_cpu - reference).abs().max().item())

    for _ in range(args.warmup):
        compiled(q_dev, k_dev, v_dev)
        torch.spyre.synchronize()

    samples_ms = []
    for _ in range(args.iters):
        start = time.perf_counter()
        compiled(q_dev, k_dev, v_dev)
        torch.spyre.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(samples_ms)
    cache_dir = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    payload = {
        "status": "ok",
        "variant": args.variant,
        "shape": {
            "batch": args.batch,
            "heads": args.heads,
            "length": args.length,
            "dim": args.dim,
        },
        "block_size": spyre_config.flash_attention_prefill_block_size,
        "is_causal": args.is_causal,
        "fallbacks_forbidden": args.forbid_fallbacks,
        "block_size_env": os.environ.get(
            "SPYRE_FLASH_ATTENTION_PREFILL_BLOCK_SIZE", ""
        ),
        "cache_dir": str(cache_dir),
        "compile_run_ms": compile_run_ms,
        "warmup": args.warmup,
        "iters": args.iters,
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "p90_ms": _percentile(ordered, 0.90),
        "max_abs_error": max_abs_error,
        **_summarize_cache(cache_dir),
    }
    print("RESULT_JSON:" + json.dumps(payload, sort_keys=True))
    return 0


def _child_env(args: argparse.Namespace, variant: str, length: int) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in VARIANT_ENV[variant].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    if args.block_size > 0:
        env["SPYRE_FLASH_ATTENTION_PREFILL_BLOCK_SIZE"] = str(args.block_size)
    else:
        env.pop("SPYRE_FLASH_ATTENTION_PREFILL_BLOCK_SIZE", None)
    env["DXP_DEBUG"] = "1" if args.dxp_debug else "0"
    nonce = random.randint(0, 999999)
    cache_prefix = args.cache_prefix.rstrip("/")
    env["TORCHINDUCTOR_CACHE_DIR"] = (
        f"{cache_prefix}-{variant}-B{args.batch}-H{args.heads}"
        f"-L{length}-D{args.dim}-C{int(args.is_causal)}-{os.getpid()}-{nonce}"
    )
    return env


def _run_parent(args: argparse.Namespace) -> int:
    results = []
    script = Path(__file__).resolve()
    lengths = [int(value) for value in _parse_csv(args.lengths)]
    variants = _parse_csv(args.variants)
    for variant in variants:
        if variant not in VARIANT_ENV:
            raise ValueError(f"unknown variant {variant!r}; valid={sorted(VARIANT_ENV)}")

    for length in lengths:
        for variant in variants:
            cmd = [
                sys.executable,
                str(script),
                "--child",
                "--variant",
                variant,
                "--batch",
                str(args.batch),
                "--heads",
                str(args.heads),
                "--length",
                str(length),
                "--dim",
                str(args.dim),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--seed",
                str(args.seed + length),
                "--atol",
                str(args.atol),
                "--rtol",
                str(args.rtol),
            ]
            if args.is_causal:
                cmd.append("--is-causal")
            if args.forbid_fallbacks:
                cmd.append("--forbid-fallbacks")
            env = _child_env(args, variant, length)
            started = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=args.timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                elapsed_s = time.perf_counter() - started
                results.append(
                    {
                        "status": "timeout",
                        "variant": variant,
                        "is_causal": args.is_causal,
                        "fallbacks_forbidden": args.forbid_fallbacks,
                        "shape": {
                            "batch": args.batch,
                            "heads": args.heads,
                            "length": length,
                            "dim": args.dim,
                        },
                        "cache_dir": env["TORCHINDUCTOR_CACHE_DIR"],
                        "timeout_s": args.timeout_s,
                        "elapsed_s": elapsed_s,
                        "stdout_tail": (exc.stdout or "")[-TAIL_CHARS:],
                        "stderr_tail": (exc.stderr or "")[-TAIL_CHARS:],
                    }
                )
                _print_last_result(results[-1])
                continue
            elapsed_s = time.perf_counter() - started
            result_line = None
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT_JSON:"):
                    result_line = line[len("RESULT_JSON:"):]
            if result_line is None or proc.returncode != 0:
                results.append(
                    {
                        "status": "failed",
                        "variant": variant,
                        "is_causal": args.is_causal,
                        "fallbacks_forbidden": args.forbid_fallbacks,
                        "shape": {
                            "batch": args.batch,
                            "heads": args.heads,
                            "length": length,
                            "dim": args.dim,
                        },
                        "cache_dir": env["TORCHINDUCTOR_CACHE_DIR"],
                        "returncode": proc.returncode,
                        "elapsed_s": elapsed_s,
                        "stdout_tail": proc.stdout[-TAIL_CHARS:],
                        "stderr_tail": proc.stderr[-TAIL_CHARS:],
                    }
                )
            else:
                payload = json.loads(result_line)
                payload["elapsed_s"] = elapsed_s
                results.append(payload)
            _print_last_result(results[-1])

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2, sort_keys=True))
    return 1 if any(row.get("status") != "ok" for row in results) else 0


def _print_last_result(row: dict) -> None:
    shape = row.get("shape", {})
    prefix = (
        f"L={shape.get('length')} {row.get('variant')} "
        f"status={row.get('status')}"
    )
    if row.get("status") == "ok":
        mixed = len(row.get("mixed_sdscs", []))
        print(
            f"{prefix} median={row['median_ms']:.6f}ms "
            f"mean={row['mean_ms']:.6f}ms max_err={row['max_abs_error']:.6g} "
            f"mixed={mixed} cache={row['cache_dir']}",
            flush=True,
        )
    else:
        print(
            f"{prefix} rc={row.get('returncode')} cache={row.get('cache_dir')}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", default="onchip")
    parser.add_argument("--variants", default="vanilla,flash_hbm,onchip")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--lengths", default="128,256")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="flash prefill block size; 0 leaves the environment unset",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0xA771)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--is-causal", action="store_true")
    parser.add_argument(
        "--forbid-fallbacks",
        action="store_true",
        help=(
            "treat Torch-Spyre CPU fallback warnings as errors in child runs; "
            "useful as a readiness gate for removing fallback-only paths"
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--cache-prefix", default="/tmp/sdpa-onchip-sweep")
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--dxp-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)

    if args.child:
        return _run_child(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a one-layer FMS Granite block against Spyre with empty weights.

This probe is intentionally close to the historical Granite cost-model probe,
but it avoids global ``torch.manual_seed`` because some Spyre runtime overlays
do not expose the PrivateUse1 default-generator hooks.  The goal is to test the
compiler/runtime path for a real FMS block shape without copying checkpoint
weights to the AIU.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import pathlib
import re
import statistics
import sys
import time
import traceback
from typing import Any

import torch
import torch.nn as nn

try:
    import torch_spyre

    if hasattr(torch_spyre, "_autoload"):
        torch_spyre._autoload()
except Exception:
    torch_spyre = None


ANTONI_TRACE_KERNELS = [
    "sdsc_fused_mul_0",
    "sdsc_fused_linear_mul_rms_norm_sum_unsqueeze_view_0",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_1",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_transpose_unsqueeze_view_2",
    "sdsc_fused_linear_mul_rms_norm_silu_3",
    "sdsc_fused_add_linear_mul_4",
    "sdsc_fused_add_mean_mul_rsqrt_0",
    "sdsc_fused_bmm_transpose_unsqueeze_0",
    "sdsc_fused_div_0",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable_linear_mul_sum_transpose_unsqueeze_view_1",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_transpose_unsqueeze_view_2",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable_linear_unsqueeze_3",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_linear_transpose_unsqueeze_view_4",
    "sdsc_fused_add_linear_mul_rms_norm_silu_5",
    "sdsc_fused_add_linear_mul_silu_6",
    "sdsc_fused_linear_overwrite_slice_transpose_view_1",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_2",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_unsqueeze_view_3",
    "sdsc_fused_linear_mul_rms_norm_silu_4",
    "sdsc_fused_add_mul_5",
]


def _shape(value: Any) -> list[int] | None:
    return None if value is None else list(value.shape)


def _strip_hash(name: str) -> str:
    previous = None
    while previous != name:
        previous = name
        name = re.sub(r"(?<=_\d)_[a-z0-9]{1,8}(?:_[a-z0-9]{1,8})+_?$", "", name)
        name = re.sub(r"_[a-z0-9]{6,}_?$", "", name)
        name = re.sub(r"_[a-z0-9]{2}_[a-z0-9]{4,}$", "", name)
    return name


def _summarize_cache(cache_dir: pathlib.Path) -> list[dict[str, Any]]:
    kernels: list[dict[str, Any]] = []
    for directory in sorted(glob.glob(str(cache_dir / "inductor-spyre" / "sdsc_*"))):
        if not os.path.isdir(directory):
            continue
        item: dict[str, Any] = {
            "kernel_dir": os.path.basename(directory),
            "normalized": _strip_hash(os.path.basename(directory)),
            "splits": [],
        }
        for path in sorted(glob.glob(os.path.join(directory, "sdsc_*.json"))):
            data = json.loads(pathlib.Path(path).read_text())
            for op_name, op in data.items():
                split = op.get("numWkSlicesPerDim_")
                if split:
                    item["splits"].append(
                        {
                            "file": os.path.basename(path),
                            "op": op_name,
                            "split": split,
                        }
                    )
        kernels.append(item)
    return kernels


def _materialize_empty_spyre_params(module: nn.Module) -> None:
    for child in module.children():
        _materialize_empty_spyre_params(child)
    for name, param in list(module._parameters.items()):
        if param is None:
            continue
        module._parameters[name] = nn.Parameter(
            torch.empty(tuple(param.shape), device="spyre", dtype=torch.float16),
            requires_grad=False,
        )
    for name, buf in list(module._buffers.items()):
        if buf is None or not torch.is_floating_point(buf):
            continue
        module._buffers[name] = torch.empty(
            tuple(buf.shape), device="spyre", dtype=torch.float16
        )


def _selected_freqs(position_ids: torch.Tensor) -> torch.Tensor:
    batch, seq = position_ids.shape
    return torch.zeros((batch, seq, 2, 2, 64), device="spyre", dtype=torch.float16)


def _make_mask(batch: int, q_len: int, kv_len: int) -> torch.Tensor:
    return torch.empty((batch, q_len, kv_len), device="spyre", dtype=torch.float16)


def _sync(value: Any) -> None:
    if isinstance(value, tuple):
        for item in value:
            _sync(item)
        return
    if isinstance(value, list):
        for item in value:
            _sync(item)
        return
    if isinstance(value, torch.Tensor):
        value.cpu()


def _write_summary(out_dir: pathlib.Path, result: dict[str, Any]) -> None:
    lines = [
        "# Granite Block Layer Probe",
        "",
        f"- case: `{result['case']}`",
        f"- returncode: `{result['returncode']}`",
        f"- fused_weights: `{result['fused_weights']}`",
        f"- compile_block: `{result['compile_block']}`",
        f"- input_shape: `{result['input_shape']}`",
        f"- position_ids_shape: `{result['position_ids_shape']}`",
        f"- mask_shape: `{result['mask_shape']}`",
        f"- past_key_value_shape: `{result['past_key_value_shape']}`",
        f"- generated SDSC exact normalized match: `{result['cache_match']}`",
        f"- generated SDSC overlap: `{len(result['cache_overlap'])}/{len(ANTONI_TRACE_KERNELS)}`",
        "",
        "## Timing",
        "",
        f"- median_ms: `{result.get('median_ms')}`",
        f"- all_ms: `{result.get('all_ms')}`",
        "",
        "## Generated SDSCs",
        "",
        "| normalized kernel | split samples |",
        "|---|---|",
    ]
    for kernel in result["cache_summary"]:
        samples = [str(sample["split"]) for sample in kernel.get("splits", [])[:2]]
        lines.append(f"| `{kernel['normalized']}` | `{' ; '.join(samples)}` |")
    if result["cache_missing"]:
        lines += ["", "## Missing vs Antoni Trace", ""]
        lines += [f"- `{kernel}`" for kernel in result["cache_missing"]]
    if result["cache_extra"]:
        lines += ["", "## Extra vs Antoni Trace", ""]
        lines += [f"- `{kernel}`" for kernel in result["cache_extra"]]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def run_case(args: argparse.Namespace) -> tuple[pathlib.Path, dict[str, Any]]:
    if args.fms_root:
        sys.path.insert(0, args.fms_root)
    from fms.models.granite import Granite, _8b_config

    def log(message: str) -> None:
        print(f"[block-probe] {message}", flush=True)

    out_dir = args.run_root / f"block_{args.case}"
    cache_dir = out_dir / "cache"
    export_dir = out_dir / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    export_dir.mkdir(exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    os.environ["DTCOMPILER_EXPORT_DIR"] = str(export_dir)
    os.environ["DEEPRT_EXPORT_DIR"] = str(export_dir)

    log("constructing one-layer Granite")
    cfg = _8b_config.updated(
        nlayers=1,
        fused_weights=args.fused_weights,
        linear_config={"linear_type": "torch_linear"},
    )
    model = Granite(cfg).eval().to(torch.float16)
    log("materializing empty spyre parameters")
    _materialize_empty_spyre_params(model)
    block = model.base_model.layers[0]
    if args.compile_block:
        log("installing block compile wrapper")
        try:
            block.compile(mode="default", backend="inductor")
        except TypeError:
            block.compile(dynamic=False)

    batch, hidden = args.batch, args.hidden
    m = args.seq_len if args.case == "prefill" else args.decode_multiple
    kv_len = args.seq_len if args.case == "prefill" else args.seq_len + args.decode_multiple
    pos_start = 0 if args.case == "prefill" else args.seq_len
    log("creating inputs")
    x = torch.empty((batch, m, hidden), device="spyre", dtype=torch.float16)
    position_ids = torch.arange(pos_start, pos_start + m, dtype=torch.long).unsqueeze(0)
    mask = _make_mask(batch, m, kv_len if args.case != "prefill" else m)
    kwargs: dict[str, Any] = {
        "position_ids": position_ids,
        "use_cache": True,
        "attn_name": args.attn_name,
        "mask": mask,
        "selected_freqs": _selected_freqs(position_ids),
    }
    if args.case == "prefill":
        kwargs["past_key_value_state"] = None
    else:
        k = torch.empty((batch, 8, args.seq_len, 128), device="spyre", dtype=torch.float16)
        v = torch.empty((batch, 8, args.seq_len, 128), device="spyre", dtype=torch.float16)
        kwargs["past_key_value_state"] = (k, v)
    if args.case == "decode_fill":
        kwargs["is_filling_mode"] = True
        kwargs["tokens_in_current_block"] = 1
        kwargs["cache_update_position"] = args.seq_len + 1
        k = torch.empty((batch, 8, kv_len, 128), device="spyre", dtype=torch.float16)
        v = torch.empty((batch, 8, kv_len, 128), device="spyre", dtype=torch.float16)
        kwargs["past_key_value_state"] = (k, v)

    result: dict[str, Any] = {
        "case": args.case,
        "input_shape": _shape(x),
        "position_ids_shape": _shape(position_ids),
        "mask_shape": _shape(mask),
        "past_key_value_shape": None
        if kwargs["past_key_value_state"] is None
        else [_shape(tensor) for tensor in kwargs["past_key_value_state"]],
        "returncode": 0,
        "error": None,
        "fused_weights": args.fused_weights,
        "compile_block": args.compile_block,
        "attn_name": args.attn_name,
        "warmups": args.warmups,
        "iters": args.iters,
    }
    timings: list[float] = []
    try:
        for index in range(args.warmups + args.iters):
            log(f"calling block iteration {index + 1}/{args.warmups + args.iters}")
            start = time.time()
            out = block(x=x, **kwargs)
            _sync(out)
            elapsed_ms = (time.time() - start) * 1000.0
            if index >= args.warmups:
                timings.append(elapsed_ms)
        if isinstance(out, tuple):
            y, cache = out
            result["output_shape"] = _shape(y)
            result["cache_shape"] = [_shape(tensor) for tensor in cache]
        else:
            result["output_shape"] = _shape(out)
        result["median_ms"] = statistics.median(timings) if timings else None
        result["all_ms"] = [round(value, 3) for value in timings]
    except BaseException:
        result["returncode"] = 1
        result["error"] = traceback.format_exc()
        (out_dir / "error.txt").write_text(result["error"])
    finally:
        gc.collect()

    cache_summary = _summarize_cache(cache_dir)
    cache_norm = {kernel["normalized"] for kernel in cache_summary}
    golden = set(ANTONI_TRACE_KERNELS)
    result.update(
        {
            "cache_summary": cache_summary,
            "cache_overlap": sorted(cache_norm & golden),
            "cache_missing": sorted(golden - cache_norm),
            "cache_extra": sorted(cache_norm - golden),
            "cache_match": cache_norm == golden,
        }
    )
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    _write_summary(out_dir, result)
    return out_dir, result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fms-root")
    parser.add_argument("--run-root", required=True, type=pathlib.Path)
    parser.add_argument(
        "--case",
        choices=["prefill", "decode_expand", "decode_fill"],
        default="prefill",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--decode-multiple", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--attn-name", default="sdpa_causal")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--compile-block", action="store_true")
    parser.add_argument("--fused-weights", action="store_true", default=True)
    parser.add_argument("--unfused-weights", action="store_false", dest="fused_weights")
    return parser


def main() -> int:
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.recompile_limit = 1000
    torch._inductor.config.use_joint_graph_passes = False
    out_dir, result = run_case(_parser().parse_args())
    print(f"RESULT_JSON={out_dir / 'result.json'}", flush=True)
    print(f"SUMMARY_MD={out_dir / 'summary.md'}", flush=True)
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return int(result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())

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

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
import traceback
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the Spyre/DeepTools backend path for "
            "spyre::causal_score_bias_like."
        )
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--query-len", type=int, default=4)
    parser.add_argument("--key-len", type=int, default=64)
    parser.add_argument("--key-start", type=int, default=2)
    parser.add_argument(
        "--opfunc",
        default="causal_score_bias_like",
        help=(
            "Backend opFuncName to emit. The default exercises the real scaffold; "
            "existing names such as identity, maskbyindex, where3, or greaterthan "
            "are useful reuse probes."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="TORCHINDUCTOR_CACHE_DIR to use. A temporary directory is used by default.",
    )
    parser.add_argument("--print-values", action="store_true")
    return parser.parse_args()


def _summarize_tensor(tensor, *, print_values: bool) -> dict:
    flat = tensor.flatten()
    summary = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sum": float(tensor.float().sum()),
        "first_values": [float(v) for v in flat[:12]],
    }
    if tensor.ndim >= 4:
        summary["row2_first_values"] = [
            float(v) for v in tensor[0, 0, min(2, tensor.shape[-2] - 1), :8]
        ]
    if print_values:
        summary["values"] = tensor.tolist()
    return summary


def _constant_names(dsc: dict) -> list[str]:
    constant_info = dsc.get("constantInfo_", {})
    if not isinstance(constant_info, dict):
        return []
    return [
        const.get("name_")
        for const in constant_info.values()
        if isinstance(const, dict)
    ]


def _load_causal_mask_dataop_helper():
    helper = (
        Path(__file__).resolve().parents[1]
        / "torch_spyre"
        / "_inductor"
        / "codegen"
        / "causal_mask_dataop.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_causal_mask_dataop_helper", helper
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _causal_score_bias_contract(sdsc: dict, dsc: dict) -> dict:
    """Extract the backend contract from a generated causal-bias SDSC.

    This is intentionally JSON-only so it can run even when the backend aborts
    before execution. The values describe what a DeepTools implementation must
    honor for the score-layout-anchored primitive.
    """
    compute_op = dsc.get("computeOp_", [{}])[0]
    output_name = (compute_op.get("outputLabeledDs") or [""])[0].split("-")[0]
    output_lds = None
    for lds in dsc.get("labeledDs_", []):
        if lds.get("dsName_") == output_name:
            output_lds = lds
            break

    output_layout = {}
    if output_lds is not None:
        output_layout = dsc.get("primaryDsInfo_", {}).get(
            output_lds.get("dsType_"), {}
        )

    layout_dim_order = output_layout.get("layoutDimOrder_", [])
    stick_dim_order = output_layout.get("stickDimOrder_", [])
    work_slices = sdsc.get("numWkSlicesPerDim_", {})
    split_dims = {dim: split for dim, split in work_slices.items() if split != 1}
    supported_score_layout = (
        "x" in layout_dim_order
        and stick_dim_order == ["out"]
        and "keyStart" in _constant_names(dsc)
    )

    return {
        "opfunc": compute_op.get("opFuncName"),
        "input_count": len(compute_op.get("inputLabeledDs", [])),
        "output_count": len(compute_op.get("outputLabeledDs", [])),
        "constants": _constant_names(dsc),
        "num_cores": dsc.get("numCoresUsed_"),
        "iteration_sizes": dsc.get("N_", {}),
        "work_slices": work_slices,
        "split_dims": split_dims,
        "core_id_to_work_slice": sdsc.get("coreIdToWkSlice_", {}),
        "output_layout": {
            "labeled_ds": output_name,
            "layout_dim_order": layout_dim_order,
            "stick_dim_order": stick_dim_order,
            "stick_size": output_layout.get("stickSize_", []),
        },
        "inferred_query_dim": "x" if "x" in layout_dim_order else None,
        "inferred_key_dim": "out" if stick_dim_order == ["out"] else None,
        "supported_score_layout": supported_score_layout,
    }


def _metadata_from_sdsc_payload(
    path: Path, payload: dict, *, key_start: int | None = None
) -> dict:
    sdsc = next(iter(payload.values()))
    dsc = next(iter(sdsc["dscs_"][0].values()))
    compute_ops = dsc.get("computeOp_", [])
    opfuncs = [op.get("opFuncName") for op in compute_ops]
    metadata = {
        "path": str(path),
        "opfuncs": opfuncs,
        "constants": _constant_names(dsc),
        "inputs": compute_ops[0].get("inputLabeledDs", []) if compute_ops else [],
        "outputs": compute_ops[0].get("outputLabeledDs", []) if compute_ops else [],
    }
    if "causal_score_bias_like" in opfuncs:
        contract = _causal_score_bias_contract(sdsc, dsc)
        metadata["causal_score_bias_contract"] = contract
        if key_start is not None:
            helper = _load_causal_mask_dataop_helper()
            metadata["causal_idx_to_mask_candidate"] = (
                helper.build_causal_idx_to_mask_candidate(
                    contract,
                    key_start=key_start,
                )
            )
    return metadata


def _collect_sdsc_metadata(
    cache_dir: Path, *, key_start: int | None = None
) -> list[dict]:
    metadata = []
    for path in sorted(cache_dir.rglob("sdsc_*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            metadata.append({"path": str(path), "error": repr(exc)})
            continue
        try:
            metadata.append(
                _metadata_from_sdsc_payload(path, payload, key_start=key_start)
            )
        except Exception as exc:  # noqa: BLE001
            metadata.append({"path": str(path), "error": repr(exc)})
    return metadata


def _expected_bias(torch, scores_cpu, key_start: int):
    q_len = scores_cpu.size(-2)
    k_len = scores_cpu.size(-1)
    q = torch.arange(q_len, device=scores_cpu.device).unsqueeze(-1)
    k = torch.arange(k_len, device=scores_cpu.device).unsqueeze(0) + key_start
    bias = torch.zeros_like(scores_cpu)
    return bias.masked_fill(k > q, float("-inf"))


def main() -> int:
    args = _parse_args()
    cache_dir = Path(
        args.cache_dir
        or tempfile.mkdtemp(prefix=f"causal-score-bias-{args.opfunc}-")
    )
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)

    import torch
    import torch_spyre  # noqa: F401

    if args.opfunc != "causal_score_bias_like":
        from torch_spyre._inductor.spyre_kernel import PointwiseOp, SpyreOpFuncs

        def patched(scores, key_start):
            op_info = {"constants": {"keyStart": key_start}}
            return PointwiseOp(args.opfunc, [scores], op_info)

        SpyreOpFuncs.causal_score_bias_like = staticmethod(patched)

    def fn(scores):
        return torch.ops.spyre.causal_score_bias_like(scores, args.key_start)

    result = {
        "opfunc": args.opfunc,
        "cache_dir": str(cache_dir),
        "shape": [args.batch, args.heads, args.query_len, args.key_len],
    }

    try:
        scores_cpu = torch.arange(
            args.batch * args.heads * args.query_len * args.key_len,
            dtype=torch.float16,
        ).reshape(args.batch, args.heads, args.query_len, args.key_len)
        expected = _expected_bias(torch, scores_cpu, args.key_start)
        result["expected"] = _summarize_tensor(
            expected, print_values=args.print_values
        )
        scores = scores_cpu.to("spyre")
        out = torch.compile(fn, backend="inductor")(scores).to("cpu")
        result["status"] = "ok"
        result["output"] = _summarize_tensor(out, print_values=args.print_values)
        result["matches_expected"] = bool(torch.equal(out, expected))
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback_tail"] = traceback.format_exc().splitlines()[-20:]

    result["sdscs"] = _collect_sdsc_metadata(cache_dir, key_start=args.key_start)
    print("RESULT_JSON:" + json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

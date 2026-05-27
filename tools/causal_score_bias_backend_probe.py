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


def _collect_sdsc_metadata(cache_dir: Path) -> list[dict]:
    metadata = []
    for path in sorted(cache_dir.rglob("sdsc_*.json")):
        try:
            payload = json.loads(path.read_text())
            root = next(iter(payload.values()))
            dsc = next(iter(root["dscs_"][0].values()))
        except Exception as exc:  # noqa: BLE001
            metadata.append({"path": str(path), "error": repr(exc)})
            continue
        constants = []
        for const in dsc.get("constantInfo_", {}).values():
            constants.append(const.get("name_"))
        metadata.append(
            {
                "path": str(path),
                "opfuncs": [
                    op.get("opFuncName") for op in dsc.get("computeOp_", [])
                ],
                "constants": constants,
                "inputs": dsc.get("computeOp_", [{}])[0].get("inputLabeledDs", []),
                "outputs": dsc.get("computeOp_", [{}])[0].get("outputLabeledDs", []),
            }
        )
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

    result["sdscs"] = _collect_sdsc_metadata(cache_dir)
    print("RESULT_JSON:" + json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

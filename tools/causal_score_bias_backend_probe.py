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
import subprocess
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
    parser.add_argument(
        "--candidate-plan-json",
        default="",
        help=(
            "Optional path to write the causal IdxToMask+where3 emission plan "
            "derived from generated causal_score_bias_like SDSCs."
        ),
    )
    parser.add_argument(
        "--candidate-parser-probe-dir",
        default="",
        help=(
            "Optional directory where a parser-probe SDSC bundle for the "
            "candidate causal IdxToMask data-op should be written."
        ),
    )
    parser.add_argument(
        "--candidate-parser-probe-name",
        default="causal_idx_to_mask_parser_probe",
        help=(
            "SDSC name to request from the candidate parser-probe helper. "
            "The emitted file is sdsc_<name>.json."
        ),
    )
    parser.add_argument(
        "--run-candidate-parser-probe",
        action="store_true",
        help=(
            "After emitting --candidate-parser-probe-dir, run "
            "dxp_standalone --bundle -d <dir> and record the return code plus "
            "stdout/stderr tails in RESULT_JSON."
        ),
    )
    parser.add_argument("--print-values", action="store_true")
    args = parser.parse_args()
    if args.run_candidate_parser_probe and not args.candidate_parser_probe_dir:
        parser.error(
            "--run-candidate-parser-probe requires --candidate-parser-probe-dir"
        )
    return args


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


def _metadata_from_sdsc_payload(
    path: Path, payload: dict, *, key_start: int | None = None
) -> dict:
    sdsc = next(iter(payload.values()))
    dsc = next(iter(sdsc["dscs_"][0].values()))
    compute_ops = dsc.get("computeOp_", [])
    opfuncs = [op.get("opFuncName") for op in compute_ops]
    helper = _load_causal_mask_dataop_helper()
    metadata = {
        "path": str(path),
        "opfuncs": opfuncs,
        "constants": helper.constant_names_from_dsc(dsc),
        "inputs": compute_ops[0].get("inputLabeledDs", []) if compute_ops else [],
        "outputs": compute_ops[0].get("outputLabeledDs", []) if compute_ops else [],
    }
    if "causal_score_bias_like" in opfuncs:
        contract = helper.causal_score_bias_contract_from_sdsc(sdsc, dsc)
        metadata["causal_score_bias_contract"] = contract
        if key_start is not None:
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


def _causal_score_bias_payload(cache_dir: Path) -> tuple[Path | None, dict | None]:
    for path in sorted(cache_dir.rglob("sdsc_*.json")):
        try:
            payload = json.loads(path.read_text())
            sdsc = next(iter(payload.values()))
            dsc = next(iter(sdsc["dscs_"][0].values()))
            compute_ops = dsc.get("computeOp_", [])
        except Exception:  # noqa: BLE001
            continue
        opfuncs = [op.get("opFuncName") for op in compute_ops]
        if "causal_score_bias_like" in opfuncs:
            return path, payload
    return None, None


def _validate_sdsc_name(name: str) -> None:
    if not name:
        raise ValueError("candidate parser-probe SDSC name must be non-empty")
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(
            "candidate parser-probe SDSC name must not contain path separators"
        )


def _write_single_sdsc_bundle(bundle_dir: Path, sdsc_payload: dict) -> dict:
    if not isinstance(sdsc_payload, dict) or len(sdsc_payload) != 1:
        raise ValueError("parser-probe helper must return a single SDSC payload")

    sdsc_name = next(iter(sdsc_payload))
    _validate_sdsc_name(sdsc_name)
    file_name = f"sdsc_{sdsc_name}.json"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    sdsc_path = bundle_dir / file_name
    bundle_path = bundle_dir / "bundle.mlir"
    sdsc_path.write_text(json.dumps(sdsc_payload, indent=2, sort_keys=True))
    bundle_path.write_text(
        "\n".join(
            [
                "module {",
                "\tfunc.func @sdsc_bundle() {",
                "\t\tsdscbundle.sdsc_execute () "
                f"{{sdsc_filename={json.dumps(file_name)}}}",
                "\t\treturn",
                "\t}",
                "}",
                "",
            ]
        )
    )
    return {
        "name": sdsc_name,
        "sdsc_json": str(sdsc_path),
        "bundle_mlir": str(bundle_path),
    }


def _write_causal_idx_to_mask_parser_probe_bundle(
    bundle_dir: Path,
    payload: dict,
    *,
    key_start: int,
    name: str = "causal_idx_to_mask_parser_probe",
) -> dict:
    helper = _load_causal_mask_dataop_helper()
    builder = getattr(
        helper,
        "build_causal_idx_to_mask_parser_probe_sdsc",
        None,
    )
    if builder is None:
        raise AttributeError(
            "causal_mask_dataop.py does not expose "
            "build_causal_idx_to_mask_parser_probe_sdsc"
        )
    sdsc_payload = builder(payload, key_start=key_start, name=name)
    if isinstance(sdsc_payload, dict) and len(sdsc_payload) == 1:
        returned_name = next(iter(sdsc_payload))
        if returned_name != name:
            raise ValueError(
                "parser-probe helper returned SDSC name "
                f"{returned_name!r}, expected {name!r}"
            )
    return _write_single_sdsc_bundle(bundle_dir, sdsc_payload)


def _tail_lines(text: str, *, limit: int = 80) -> list[str]:
    return text.splitlines()[-limit:]


def _emit_candidate_parser_probe_bundle(
    cache_dir: Path,
    bundle_dir: Path,
    *,
    key_start: int,
    name: str,
    run: bool,
) -> dict:
    result = {
        "dir": str(bundle_dir),
        "requested_name": name,
        "run_requested": run,
    }
    try:
        _validate_sdsc_name(name)
        source_path, source_payload = _causal_score_bias_payload(cache_dir)
        if source_payload is None:
            result["status"] = "failed"
            result["error"] = "no generated causal_score_bias_like SDSC found"
            return result

        written = _write_causal_idx_to_mask_parser_probe_bundle(
            bundle_dir,
            source_payload,
            key_start=key_start,
            name=name,
        )
        result.update(written)
        result["source_sdsc"] = str(source_path)
        result["status"] = "emitted"

        if run:
            cmd = ["dxp_standalone", "--bundle", "-d", str(bundle_dir)]
            completed = subprocess.run(
                cmd,
                capture_output=True,
                check=False,
                text=True,
            )
            result["run"] = {
                "cmd": cmd,
                "rc": completed.returncode,
                "returncode": completed.returncode,
                "stdout_tail": _tail_lines(completed.stdout),
                "stderr_tail": _tail_lines(completed.stderr),
            }
            result["status"] = (
                "run_ok" if completed.returncode == 0 else "run_failed"
            )
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback_tail"] = traceback.format_exc().splitlines()[-20:]
    return result


def _candidate_emission_plans(metadata: list[dict], *, key_start: int) -> list[dict]:
    helper = _load_causal_mask_dataop_helper()
    plans = []
    for item in metadata:
        contract = item.get("causal_score_bias_contract")
        if not isinstance(contract, dict):
            continue
        plans.append(
            helper.build_causal_idx_to_mask_emission_plan(
                contract,
                key_start=key_start,
            )
        )
    return plans


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
    if args.candidate_plan_json:
        plans = _candidate_emission_plans(result["sdscs"], key_start=args.key_start)
        plan_path = Path(args.candidate_plan_json)
        plan_path.write_text(json.dumps({"plans": plans}, indent=2, sort_keys=True))
        result["candidate_plan_json"] = str(plan_path)
        result["candidate_plan_count"] = len(plans)
    if args.candidate_parser_probe_dir:
        result["candidate_parser_probe"] = _emit_candidate_parser_probe_bundle(
            cache_dir,
            Path(args.candidate_parser_probe_dir),
            key_start=args.key_start,
            name=args.candidate_parser_probe_name,
            run=args.run_candidate_parser_probe,
        )
    print("RESULT_JSON:" + json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run DXP with a preload shim that bypasses pre-DDC passes for restickify.

This is a validation probe for the Deeptools patch direction, not a production
execution path. It compiles a small ``LD_PRELOAD`` library that no-ops:

* ``Dsm::doCoreletSplitSdsc(SuperDsc*)``
* ``L3DlOpsScheduler::run(SuperDsc&)``

Then it runs ``dxp_standalone`` on a restickify SDSC bundle. If the fixture
compiles and the generated senprog contains LX/SFP/PT units but no L3/HBM
tokens, the result supports moving these generic pre-DDC passes out of the way
for restickify DDL-template inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


_SHIM_SRC = r'''
#include <iostream>
class SuperDsc;
class Dsm {
 public:
  static void doCoreletSplitSdsc(SuperDsc* sdsc);
};
class L3DlOpsScheduler {
 public:
  void run(SuperDsc& sdsc);
};
void Dsm::doCoreletSplitSdsc(SuperDsc*) {
  std::cerr << "[restickify-probe] skipped Dsm::doCoreletSplitSdsc via LD_PRELOAD\n";
}
void L3DlOpsScheduler::run(SuperDsc&) {
  std::cerr << "[restickify-probe] skipped L3DlOpsScheduler::run via LD_PRELOAD\n";
}
'''


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    stdout: Path,
    stderr: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    stdout.write_text(proc.stdout, encoding="utf-8")
    if stderr is not None:
        stderr.write_text(proc.stderr, encoding="utf-8")
    elif proc.stderr:
        stdout.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc.returncode


def _tool_path(bin_dir: Path, tool: str) -> str:
    candidate = bin_dir / tool
    if candidate.exists():
        return str(candidate)
    found = shutil.which(tool)
    if found:
        return found
    raise FileNotFoundError(f"could not find {tool}; pass --deeptools-bin")


def _strip_json_comments(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("//")]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bundle_mlir(sdsc_name: str) -> str:
    return (
        "module {\n"
        "  func.func @sdsc_bundle() {\n"
        f"    sdscbundle.sdsc_execute () {{sdsc_filename=\"{sdsc_name}\"}}\n"
        "    return\n"
        "  }\n"
        "}\n"
    )


def _compile_shim(output_dir: Path) -> Path:
    src = output_dir / "restickify_skip_dxp_preddc_shim.cpp"
    lib = output_dir / "librestickify_skip_dxp_preddc.so"
    src.write_text(_SHIM_SRC, encoding="utf-8")
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not cxx:
        raise FileNotFoundError("could not find g++, c++, or clang++")
    rc = _run(
        [cxx, "-shared", "-fPIC", "-std=c++17", str(src), "-o", str(lib)],
        cwd=output_dir,
        stdout=output_dir / "compile.out",
        stderr=output_dir / "compile.err",
    )
    if rc != 0:
        raise RuntimeError((output_dir / "compile.err").read_text(encoding="utf-8"))
    return lib


def _single_dsc(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = next(iter(payload.values()))
    dsc = next(iter(root["dscs_"][0].values()))
    return root, dsc


def _summarize_sdsc(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    root, dsc = _single_dsc(path)
    return {
        "num_cores_used": root.get("numCoresUsed_"),
        "schedule_node_count": len(dsc.get("scheduleTree_", [])),
        "data_stage_param_count": len(dsc.get("dataStageParam_", {})),
        "labeled_ds": [
            {
                "idx": lds.get("ldsIdx_"),
                "name": lds.get("dsName_"),
                "type": lds.get("dsType_"),
            }
            for lds in dsc.get("labeledDs_", [])
        ],
        "op_funcs": [op.get("opFuncName") for op in dsc.get("computeOp_", [])],
    }


def _summarize_senprog(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    tokens = ["LXLU", "LXSU", "SFP", "PT", "L3LU", "L3SU", "HBM"]
    return {
        "exists": path.exists(),
        "bytes": len(text),
        "token_counts": {token: text.count(token) for token in tokens},
        "contains_lx": "lx" in text.lower(),
        "contains_hbm": "hbm" in text.lower(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdsc", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--deeptools-bin",
        type=Path,
        default=Path("/opt/ibm/spyre/deeptools/bin"),
    )
    parser.add_argument("--senarch", default="rcudd1a")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = args.output_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    sdsc = bundle_dir / "sdsc.json"
    _strip_json_comments(args.sdsc, sdsc)
    (bundle_dir / "bundle.mlir").write_text(_bundle_mlir("sdsc.json"), encoding="utf-8")

    shim = _compile_shim(args.output_dir)
    dxp = _tool_path(args.deeptools_bin, "dxp_standalone")
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(shim)
    env["SENARCH"] = args.senarch
    env["DXP_VERBOSE"] = "0"
    rc = _run(
        [dxp, "--bundle", "-d", str(bundle_dir)],
        cwd=bundle_dir,
        stdout=bundle_dir / "dxp_preload.log",
        env=env,
    )

    debug_dir = bundle_dir / "debug" / "sdsc"
    summary = {
        "input_sdsc": str(args.sdsc),
        "shim": str(shim),
        "dxp_rc": rc,
        "dxp_ok": rc == 0,
        "dxp_log_tail": "\n".join(
            (bundle_dir / "dxp_preload.log")
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()[-20:]
        ),
        "pre_ddc_sdsc": _summarize_sdsc(debug_dir / "sdsc.out.json"),
        "post_ddc_sdsc": _summarize_sdsc(debug_dir / "sdsc.out.out.json"),
        "post_dip_sdsc": _summarize_sdsc(debug_dir / "sdsc.out.out.out.json"),
        "senprog": _summarize_senprog(debug_dir / "senprog.txt"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    marker = path / ".canonical_base_sha"
    return marker.read_text().strip() if marker.is_file() else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


env_root = Path(os.environ["SPYRE_ENV_ROOT"])
torch_root = Path(os.environ["TORCH_SPYRE_ROOT"])
perf_root = Path(os.environ["SPYRE_PERF_SUITE_ROOT"])
dxp = shutil.which("dxp_standalone")
dxp_path = Path(dxp) if dxp else None
extension = torch_root / "torch_spyre/_C.so"
restickify_lx = Path(os.environ["DEEPTOOLS_PATH"]) / "ddc/ddl_templates/restickify_lx.ddl"

import torch
import torch_spyre

report = {
    "python": sys.executable,
    "torch_version": torch.__version__,
    "torch_file": torch.__file__,
    "torch_spyre_file": torch_spyre.__file__,
    "torch_spyre_sha": git_head(torch_root),
    "perf_suite_sha": git_head(perf_root),
    "dxp_standalone": dxp,
    "dxp_sha256": sha256(dxp_path) if dxp_path else None,
    "torch_spyre_extension_sha256": sha256(extension),
    "restickify_lx_template": str(restickify_lx),
    "environment_root": str(env_root),
    "feature_flag": os.environ.get("SPYRE_LX_PLANNER_RELAYOUT", "0"),
}

print(json.dumps(report, indent=2, sort_keys=True))

if sys.executable != os.environ["SPYRE_PERF_PYTHON"]:
    raise SystemExit("wrong Python interpreter selected")
if not str(torch_spyre.__file__).startswith(str(torch_root)):
    raise SystemExit("wrong Torch-Spyre checkout selected")
if dxp != str(env_root / "deeptools-install/bin/dxp_standalone"):
    raise SystemExit("wrong DXP executable selected")
if report["dxp_sha256"] != "68269b28b10851f7a3e2ba8ad1a98b931128265ef40814345461cdf01a73721a":
    raise SystemExit("DXP executable does not match the canonical build")
if report["torch_spyre_extension_sha256"] != "b449a232ec1c07046eb64153d9672447242734005a3f822678f665aabe835c99":
    raise SystemExit("Torch-Spyre extension does not match the canonical build")
if not restickify_lx.is_file():
    raise SystemExit("Deeptools restickify_lx.ddl runtime template is missing")

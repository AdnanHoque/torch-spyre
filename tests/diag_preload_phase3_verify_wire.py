# Copyright 2025 The Torch-Spyre Authors.
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

"""Phase 3 wire-verification probe.

Compile a single nn.Linear matmul through torch.compile, then check
whether `loadmodel_to_spad_dsg.txt` in the resulting dxp bundle is empty
(no codegen change taking effect) or populated (codegen change emitted
isStatic_=1 and DSM materialized a preload graph).

Run twice:
  SPYRE_PRELOAD_STATIC=0  (baseline) — expect empty preload dsg
  SPYRE_PRELOAD_STATIC=1  (knob on)  — expect non-empty preload dsg

Same shape, same model, only the env var differs.

Predicted outcome:
  - `loadmodel_to_spad_dsg.txt` contains only the schema sentinels
    `I { }`, `O { }`, `T { }` and no `N <id> ...` node lines.
  - `execute_dsg.txt` contains real nodes (the matmul kernel itself).

Procedure:
  1. Snapshot existing bundle dirs under the inductor cache so we can
     identify which dir was created by *this* compile.
  2. Compile a single matmul under torch.compile.
  3. Diff the bundle dirs to locate the freshly created one.
  4. Read every `*_dsg.txt` in the new dir, count node lines, dump
     contents.
  5. Print a verdict.

Usage:
  python3 tests/diag_preload_phase1.py [M] [N] [K]

Defaults to M=128 N=4096 K=4096 — a representative LLM q-projection
shape that exercises the prefill matmul path.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TORCH_SPYRE_DOWNCAST_WARN", "0")

# Allow running this script directly from the repo root: ensure the repo
# root is on sys.path so `import torch_spyre` resolves the editable source
# tree, not whatever (if anything) is installed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

import torch_spyre  # noqa: E402, F401

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402
from torch._inductor.runtime.runtime_utils import cache_dir as _ind_cache_dir  # noqa: E402

# Disable Inductor's FX graph cache so the env-var difference between
# baseline and knob-on runs actually re-traces. Without this, the second
# run can hit a cached artifact from the first and silently skip codegen.
import torch._inductor.config as _icfg  # noqa: E402

_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False


SPYRE_BUNDLES_ROOT = Path(_ind_cache_dir()) / "inductor-spyre"

# A single dsengraph file with no real nodes is exactly these lines:
EMPTY_DSG = ("I { }", "O { }", "T { }")


def snapshot_bundles() -> set[str]:
    if not SPYRE_BUNDLES_ROOT.exists():
        return set()
    return {p.name for p in SPYRE_BUNDLES_ROOT.iterdir() if p.is_dir()}


def find_new_bundles(before: set[str]) -> list[Path]:
    after = snapshot_bundles()
    return sorted(SPYRE_BUNDLES_ROOT / name for name in (after - before))


class _LinearModel(torch.nn.Module):
    def __init__(self, K: int, N: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(K, N, bias=False, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def compile_and_run_one_matmul(M: int, N: int, K: int) -> None:
    """Compile an nn.Linear so the weight is a real nn.Parameter
    (requires_grad=True) — the discriminator the codegen patch keys on."""
    model = _LinearModel(K, N).to("spyre")
    x = torch.randn(M, K, dtype=torch.float16, device="spyre")

    torch._dynamo.reset()
    compiled = torch.compile(model, dynamic=False)

    out = compiled(x)
    _ts.synchronize()
    _ = out.shape


def classify_dsg(path: Path) -> dict:
    text = path.read_text()
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    node_lines = [ln for ln in lines if ln.startswith("N ")]
    edge_lines = [ln for ln in lines if ln.startswith("E ")]
    attr_lines = [ln for ln in lines if ln.startswith("A ")]
    is_empty = tuple(lines) == EMPTY_DSG
    return {
        "path": str(path),
        "byte_size": path.stat().st_size,
        "line_count": len(lines),
        "node_count": len(node_lines),
        "edge_count": len(edge_lines),
        "attr_count": len(attr_lines),
        "is_skeleton_empty": is_empty,
        "first_lines": lines[:8],
    }


def inspect_bundle(bundle_dir: Path) -> dict:
    dsg_files = sorted(bundle_dir.glob("*_dsg.txt"))
    return {
        "bundle_dir": str(bundle_dir),
        "files": [classify_dsg(p) for p in dsg_files],
    }


def main() -> int:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 4096

    print(f"Phase 1 probe — single matmul ({M}x{K}) @ ({K}x{N}) fp16 on spyre")
    print(f"Spyre bundle root: {SPYRE_BUNDLES_ROOT}")

    before = snapshot_bundles()
    print(f"Existing bundle count before compile: {len(before)}")

    compile_and_run_one_matmul(M, N, K)

    new_bundles = find_new_bundles(before)
    print(f"New bundle dirs created by this compile: {len(new_bundles)}")
    if not new_bundles:
        print("ERROR: no new bundle dir was created.")
        print("Possibilities:")
        print("  - matmul served from torch.compile FX-cache (try torch._dynamo.reset()).")
        print("  - compile path didn't reach SpyreAsyncCompile.sdsc().")
        return 1

    report = {
        "shape": {"M": M, "N": N, "K": K, "dtype": "fp16"},
        "bundles_created": len(new_bundles),
        "per_bundle": [inspect_bundle(b) for b in new_bundles],
    }

    print()
    print("=" * 70)
    print("PER-BUNDLE INSPECTION")
    print("=" * 70)
    for binfo in report["per_bundle"]:
        print(f"\nBundle: {binfo['bundle_dir']}")
        for f in binfo["files"]:
            tag = "EMPTY" if f["is_skeleton_empty"] else "HAS NODES"
            print(
                f"  [{tag}] {Path(f['path']).name}: "
                f"{f['line_count']} lines, "
                f"{f['node_count']} N, {f['edge_count']} E, {f['attr_count']} A"
            )

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    preload_files = [
        f
        for binfo in report["per_bundle"]
        for f in binfo["files"]
        if Path(f["path"]).name == "loadmodel_to_spad_dsg.txt"
    ]
    if not preload_files:
        print("UNEXPECTED: no loadmodel_to_spad_dsg.txt in any new bundle.")
        return 2

    all_empty = all(f["is_skeleton_empty"] for f in preload_files)
    has_any_nodes = any(f["node_count"] > 0 for f in preload_files)

    if all_empty and not has_any_nodes:
        print("CONFIRMED: loadmodel_to_spad_dsg.txt is empty in every bundle.")
        print("Hypothesis from kickoff doc validated:")
        print("  torch_spyre emits zero preload nodes.")
        print("  The cross-call weight preload mechanism never fires.")
        verdict_code = 0
    elif has_any_nodes:
        print("UNEXPECTED: loadmodel_to_spad_dsg.txt contains preload nodes.")
        print("Hypothesis is wrong — preload IS being generated.")
        print("Investigate why measured first-iter behavior doesn't reflect this.")
        verdict_code = 3
    else:
        print("MIXED: preload graph is non-empty but has no N nodes (only headers).")
        verdict_code = 4

    print()
    print("RAW JSON REPORT (machine-readable):")
    print(json.dumps(report, indent=2))
    return verdict_code


if __name__ == "__main__":
    sys.exit(main())

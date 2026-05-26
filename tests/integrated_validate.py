"""Integrated validation: compile each validated shape using the cost-model
planner (from this worktree) and check the resulting SDSC split matches what
the offline validator predicted.

Sets up sys.path BEFORE importing torch_spyre so the worktree's
work_division.py is loaded (the editable finder still maps torch_spyre to the
main repo by default; we override by removing it). The .so files are
symlinked from the main repo's build.

Run: python3 tests/integrated_validate.py
"""

from __future__ import annotations
import os
import sys
import glob
import json


# --- Inject worktree path BEFORE torch_spyre is loaded ---
WORKTREE = "/tmp/pr-mixed-splits-cost-model"
sys.path.insert(0, WORKTREE)
# Remove the .venv editable finder so the regular import picks up the worktree
sys.meta_path = [
    f for f in sys.meta_path
    if not (type(f).__name__.endswith("EditableFinder") and "torch_spyre" in repr(f))
]

# Force a clean inductor cache so every compile re-runs the planner
os.system("rm -rf /tmp/torchinductor_adnan")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
os.environ["SPYRE_2D_MN_SPLIT"] = "1"        # enable the cost-model planner
os.environ["DXP_LX_FRAC_AVAIL"] = "0.8"

import torch  # noqa: E402

# Sanity: confirm we picked up the worktree's work_division
import torch_spyre._inductor.work_division as wd  # noqa: E402
print(
    f"work_division loaded from: {wd.__file__}",
    "  has _cost_model_matmul_planner:", hasattr(wd, "_cost_model_matmul_planner"),
)
assert hasattr(wd, "_cost_model_matmul_planner"), "worktree's planner not loaded"


def split_of() -> dict[str, int]:
    """Pull the (mb, x, out, in) core_fold from the most recent SDSC."""
    fs = sorted(
        glob.glob("/tmp/torchinductor_adnan/**/sdsc_0_batchmatmul.json", recursive=True),
        key=os.path.getmtime,
    )
    if not fs:
        return {}
    bmm = json.load(open(fs[-1]))["0_batchmatmul"]["dscs_"][0]["batchmatmul"]
    fold = {}
    for node in bmm["scheduleTree_"]:
        for dim, info in node.get("coordinates_", {}).get("coordInfo", {}).items():
            for a in info.get("folds", {}).get("dim_prop_attr", []):
                if a.get("label_") == "core_fold":
                    fold[dim] = max(fold.get(dim, 1), a.get("factor_", 1))
    return fold


def run_case(label, fn, args, expected_fold):
    os.system("rm -rf /tmp/torchinductor_adnan")
    c = torch.compile(fn, fullgraph=True)
    _ = c(*args).cpu()
    got = split_of()
    keys = ("mb", "x", "out", "in")
    got_str = "{" + ", ".join(f"{k}={got.get(k, 1)}" for k in keys if k in got or k in expected_fold) + "}"
    exp_str = "{" + ", ".join(f"{k}={expected_fold.get(k, 1)}" for k in keys if k in expected_fold) + "}"
    ok = all(got.get(k, 1) == expected_fold.get(k, 1) for k in expected_fold)
    print(f"{label:<26}  got={got_str:<30}  expected={exp_str:<30}  {'PASS' if ok else 'FAIL'}")


d = torch.device("spyre")

# QO bs=1: expect x=8 (M), out=4 (N), in=1 (K)
x = torch.randn(1, 512, 4096, dtype=torch.float16).to(d)
W = torch.empty(4096, 4096, dtype=torch.float16); torch.nn.init.kaiming_uniform_(W); W = W.to(d)
run_case("QO [1,512,4096]x[4096,4096]",
         lambda a, b: torch.nn.functional.linear(a, b.T), (x, W),
         {"x": 8, "out": 4, "in": 1})

# MoE gate/up: expect mb=1, x=4, out=8, in=1
A = torch.randn(8, 128, 2048, dtype=torch.float16).to(d)
B = torch.randn(8, 2048, 8192, dtype=torch.float16).to(d)
run_case("MoE gate/up [8,128,2048]x[8,2048,8192]",
         lambda a, b: torch.bmm(a, b), (A, B),
         {"mb": 1, "x": 4, "out": 8, "in": 1})

# bmm large-K: expect mb=1, x=8, out=4, in=1
A = torch.randn(8, 512, 4096, dtype=torch.float16).to(d)
B = torch.randn(8, 4096, 512, dtype=torch.float16).to(d)
run_case("bmm largeK [8,512,4096]x[8,4096,512]",
         lambda a, b: torch.bmm(a, b), (A, B),
         {"mb": 1, "x": 8, "out": 4, "in": 1})

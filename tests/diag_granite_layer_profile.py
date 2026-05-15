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

"""Profile HBM-byte share by restickify category on a granite-like layer.

Uses only matmul + transpose patterns (no silu/view/expand) to avoid a
torch post-grad pass bug. Granite-realistic shapes. Compiles with
LX_PLANNING=True, allow_all_ops_in_lx_planning=True, sencores=32.

Captures the freshly-generated SDSC bundle dirs and aggregates by:
  HBM-LOAD restickify  (weight prep — graph-input → restickify → compute op)
  FUNDAMENTAL restickify (post-compute relayout — compute op → restickify → ...)
  matmul/pointwise compute

Reports restickify share + projected layer speedup under three ring cost models.
"""

import json
import os
import sys
import time
from collections import defaultdict
from glob import glob
from unittest.mock import patch

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch_spyre

torch_spyre._autoload()

from torch._inductor import config as t_inductor_config
from torch_spyre._inductor import config as ts_config

CACHE = "/tmp/torchinductor_adnan/inductor-spyre"

# Granite 3.3 8B dimensions (slightly simplified: full H for K/V too,
# avoiding GQA expand+flatten ops that trigger the FakeTensor bug).
H = 4096
INTERMEDIATE = 12800
M_PREFILL = 128
M_DECODE = 1


def dev(*shape, dtype=torch.float16):
    return torch.rand(shape, dtype=dtype, device="spyre")


def granite_attention(x, Wq, Wk, Wv, Wo):
    """Attention with the canonical FUNDAMENTAL signature (matmul→transposed-
    matmul). x: [M, H]; Wq, Wk, Wv, Wo: [H, H]."""
    q = x @ Wq.t()                          # HBM-LOAD on Wq
    k = x @ Wk.t()                          # HBM-LOAD on Wk
    v = x @ Wv.t()                          # HBM-LOAD on Wv
    scores = q @ k.transpose(-1, -2)        # FUNDAMENTAL on k
    attn = scores @ v
    return attn @ Wo.t()                    # HBM-LOAD on Wo


def granite_mlp(x, W1, W2):
    """MLP without silu — two matmuls. Captures the weight-restickify mass
    in MLP. x: [M, H]; W1: [I, H]; W2: [H, I]."""
    h = x @ W1.t()                          # HBM-LOAD on W1
    return h @ W2.t()                       # HBM-LOAD on W2


def granite_full(x, Wq, Wk, Wv, Wo, W1, W2):
    """Combined attention + MLP."""
    a = granite_attention(x, Wq, Wk, Wv, Wo)
    m = granite_mlp(x, W1, W2)
    return a + m


# ---- bundle analysis ------------------------------------------------------

def list_dirs():
    return set(glob(f"{CACHE}/sdsc_*"))


def sdsc_info(path):
    """Return (op_name, sdsc_idx, total_hbm_bytes) for one SDSC json."""
    try:
        d = json.load(open(path))
    except Exception:
        return None, -1, 0
    outer = next(iter(d))
    inner = d[outer]
    try:
        idx = int(os.path.basename(path).split("_")[1])
    except Exception:
        idx = -1
    if not inner.get("dscs_"):
        return outer, idx, 0
    dsc0 = inner["dscs_"][0]
    op_key = next(iter(dsc0))
    op = dsc0[op_key]
    co = op.get("computeOp_", [{}])[0]
    op_name = co.get("opFuncName", op_key)
    N = op.get("N_", {})
    primary = op.get("primaryDsInfo_", {})
    total = 0
    for lds in op.get("labeledDs_", []):
        df = lds.get("dataFormat_", "SEN169_FP16")
        esz = 1 if "INT8" in df or "FP8" in df else 2
        info = primary.get(lds.get("dsType_", ""), {})
        ext = 1
        for dim in info.get("layoutDimOrder_", []):
            v = N.get(dim + "_") or N.get(dim, 1)
            if isinstance(v, (int, float)):
                ext *= int(v)
        total += ext * esz
    return op_name, idx, total


def analyze_bundles(dirs):
    """Classify each restickify as HBM-LOAD or FUNDAMENTAL by position in
    its bundle, and sum HBM bytes by category."""
    cat_bytes = defaultdict(int)
    cat_count = defaultdict(int)
    bundle_summary = []
    for d in dirs:
        sdscs = []
        for p in sorted(glob(f"{d}/sdsc_*.json")):
            n, idx, b = sdsc_info(p)
            if n:
                sdscs.append((idx, n, b))
        sdscs.sort()
        seen_compute = False
        b_hbm = b_fund = b_other = 0
        for idx, name, b in sdscs:
            if "ReStickify" in name:
                if seen_compute:
                    cat_bytes["fundamental"] += b
                    cat_count["fundamental"] += 1
                    b_fund += b
                else:
                    cat_bytes["hbm_load"] += b
                    cat_count["hbm_load"] += 1
                    b_hbm += b
            else:
                cat_bytes["other_compute"] += b
                b_other += b
                seen_compute = True
        if sdscs:
            bundle_summary.append((os.path.basename(d), b_hbm, b_fund, b_other))
    return cat_bytes, cat_count, bundle_summary


def run_case(label, fn, args, sencores=32, lx_planning=True):
    print(f"\n=== compiling: {label} (sc={sencores}, lx={lx_planning}) ===")
    before = list_dirs()
    patchers = [
        t_inductor_config.patch("force_disable_caches", True),
        ts_config.patch("lx_planning", lx_planning),
        ts_config.patch("allow_all_ops_in_lx_planning", lx_planning),
        ts_config.patch("sencores", sencores),
    ]
    for p in patchers:
        p.__enter__()
    torch.compiler.reset()
    t0 = time.time()
    err = None
    try:
        compiled = torch.compile(fn, fullgraph=True)
        try:
            compiled(*args)
        except Exception as e:
            err = f"post-compile exec: {type(e).__name__}"
    except Exception as e:
        err = f"COMPILE FAILED: {type(e).__name__}: {str(e)[:120]}"
    finally:
        torch.compiler.reset()
        for p in reversed(patchers):
            p.__exit__(None, None, None)
    new = sorted(list_dirs() - before, key=os.path.getmtime)
    print(f"  {len(new)} new bundle dirs in {time.time()-t0:.1f}s "
          f"({err or 'ok'})")
    return new


def report(title, cat_bytes, cat_count, bundle_summary):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    if bundle_summary:
        print(f"\n  {'bundle':<55} {'HBM-LOAD':>9} {'FUND':>6} {'other':>7}  (MB)")
        for n, h, f, o in bundle_summary[:12]:
            print(f"  {n[:55]:<55} {h/1e6:9.2f} {f/1e6:6.2f} {o/1e6:7.2f}")
    total = sum(cat_bytes.values())
    if total == 0:
        print("  (no bundles)")
        return
    print(f"\n  category               bytes         share")
    print(f"  {'-'*50}")
    for cat in ["hbm_load", "fundamental", "other_compute"]:
        b = cat_bytes.get(cat, 0)
        n = cat_count.get(cat, 0)
        suf = f" ({n} SDSCs)" if n else ""
        print(f"  {cat:<22} {b/1e6:9.2f} MB  ({100*b/total:5.1f}%){suf}")
    print(f"  {'TOTAL':<22} {total/1e6:9.2f} MB")
    fund = cat_bytes.get("fundamental", 0)
    hbm = cat_bytes.get("hbm_load", 0)
    print(f"\n  Ring opportunity = FUNDAMENTAL share = {100*fund/total:.1f}%")
    print(f"  Weight-prep opportunity = HBM-LOAD share = {100*hbm/total:.1f}%")
    print(f"\n  Speedup projection (HBM-bound):")
    for alpha, lbl in [(1.0, "conservative (ring_bw = HBM_bw)"),
                       (0.5, "moderate (ring_bw = 2x HBM_bw)"),
                       (0.0, "optimistic (ring negligible)")]:
        # ring-only: saves (2-alpha)/2 fraction of FUNDAMENTAL HBM
        fund_save = fund * (2.0 - alpha) / 2.0
        ring_only = total / max(total - fund_save, 1)
        # ring + LX-side weight restickify: also saves write half of HBM-LOAD
        both_save = fund_save + hbm * 0.5
        both = total / max(total - both_save, 1)
        print(f"    alpha={alpha:.1f} {lbl}:")
        print(f"      ring-only           : {ring_only:.2f}x")
        print(f"      ring + LX-weight-rs : {both:.2f}x")


def main():
    print("=== Granite-3.3 8B-like layer profile (matmul+transpose patterns only) ===")
    print(f"  H={H} INTERMEDIATE={INTERMEDIATE}  M_prefill={M_PREFILL} M_decode={M_DECODE}")
    print(f"  fp16, sencores=32, LX_PLANNING=1, allow_all_ops=1")

    cases = []
    # Prefill
    x = dev(M_PREFILL, H)
    Wq, Wk, Wv, Wo = dev(H, H), dev(H, H), dev(H, H), dev(H, H)
    W1, W2 = dev(INTERMEDIATE, H), dev(H, INTERMEDIATE)
    new = run_case("attention_prefill_M128", granite_attention, (x, Wq, Wk, Wv, Wo))
    cases.append(("attention_prefill_M128", new))
    new = run_case("mlp_prefill_M128", granite_mlp, (x, W1, W2))
    cases.append(("mlp_prefill_M128", new))
    new = run_case("full_layer_prefill_M128", granite_full,
                   (x, Wq, Wk, Wv, Wo, W1, W2))
    cases.append(("full_layer_prefill_M128", new))

    # Decode
    xd = dev(M_DECODE, H)
    new = run_case("attention_decode_M1", granite_attention, (xd, Wq, Wk, Wv, Wo))
    cases.append(("attention_decode_M1", new))
    new = run_case("mlp_decode_M1", granite_mlp, (xd, W1, W2))
    cases.append(("mlp_decode_M1", new))

    # Per-case reports
    for label, dirs in cases:
        if dirs:
            cb, cc, bs = analyze_bundles(dirs)
            report(f"{label}: {len(dirs)} bundles", cb, cc, bs)

    # Aggregate (prefill only — decode tensors are too tiny to compare meaningfully)
    prefill_dirs = []
    for label, dirs in cases:
        if "prefill" in label:
            prefill_dirs.extend(dirs)
    if prefill_dirs:
        cb, cc, bs = analyze_bundles(prefill_dirs)
        report(f"AGGREGATE prefill ({len(prefill_dirs)} bundles)", cb, cc, [])


if __name__ == "__main__":
    main()

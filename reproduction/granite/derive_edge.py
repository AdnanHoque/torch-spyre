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

"""Derive the bmm-out -> mul-in asymmetric reshard geometry from a granite bundle.

The static labeledDs PieceInfo is EMPTY in the cached SDSC JSONs; DCG computes
the per-core placement internally. But the COMPILED bundle's scheduleTree_
allocate-node startAddressCoreCorelet_.data_ exposes the per-core HBM bases, and
the allocate-node coordinates_.coordInfo folds expose the dim factorization. From
those we DERIVE (never guess):

  producer (bmm OUTPUT idx2): which cores own which out-bands, and each band's
    out-extent (sticks) -- the 8 bands, owner = base core of each in-group.
  consumer (mul INPUT idx1): which cores own which out-pieces, each piece's
    out-extent -- the 25 pieces, owner = core k.

Cross-check: pieces tile the out dim with no gap/overlap on each side; the flat
element count reconciles (out_total identical on both sides).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def dl_op(doc):
    body = doc[next(iter(doc))]
    dsc = body["dscs_"][0]
    return body, dsc[next(iter(dsc))]


def alloc_node(dlop, idx, comp="hbm"):
    for n in dlop["scheduleTree_"]:
        if n.get("nodeType_") == "allocate" and n.get("ldsIdx_") == idx:
            if comp and n.get("component_") != comp:
                continue
            return n
    return None


def per_core_bases(node):
    data = node.get("startAddressCoreCorelet_", {}).get("data_", {})
    out = {}
    for k, v in data.items():
        c = int(k.strip("[]").split(",")[0])
        out[c] = int(v)
    return out


def fold_total(node, dim):
    """Logical extent of ``dim`` = product of its dim_prop_attr fold factors."""
    ci = node["coordinates_"]["coordInfo"][dim]
    prod = 1
    for a in ci["folds"]["dim_prop_attr"]:
        prod *= a["factor_"]
    return prod


def core_fold_factor(node, dim):
    for a in node["coordinates_"]["coordInfo"][dim]["folds"]["dim_prop_attr"]:
        if a["label_"] == "core_fold":
            return a["factor_"]
    return 1


def derive_side(bundle_dir, sdsc_file, lds_idx, split_dim, word=2, stick=64):
    """Return derived geometry for one endpoint (producer or consumer).

    owners: per-piece owner core (in piece order, sorted by start coordinate)
    starts/lens: per-piece [start, start+len) on split_dim (in elements)
    out_total, row_total, n_pieces, stride_bytes.
    """
    doc = load(Path(bundle_dir) / sdsc_file)
    _, dlop = dl_op(doc)
    node = alloc_node(dlop, lds_idx, "hbm")
    if node is None:
        raise RuntimeError(f"no hbm allocate node for {sdsc_file} idx{lds_idx}")
    layout = node["layoutDimOrder_"]
    bases = per_core_bases(node)
    out_total = fold_total(node, split_dim)
    n_pieces = core_fold_factor(node, split_dim)
    row_dim = [d for d in layout if d != split_dim][0]
    row_total = fold_total(node, row_dim)
    # Per-piece out-extent (uniform core_fold split of out_total).
    chunk = out_total // n_pieces  # elements per piece on split_dim
    # Owner core per band: group bases by value, in increasing base order ->
    # logical band order. Owner = the FIRST core listed for that base.
    base_to_cores = {}
    for c in sorted(bases):
        base_to_cores.setdefault(bases[c], []).append(c)
    ordered_bases = sorted(base_to_cores)  # ascending base = ascending band
    assert len(ordered_bases) == n_pieces, (
        f"{sdsc_file} idx{lds_idx}: {len(ordered_bases)} distinct bases != "
        f"core_fold {n_pieces}"
    )
    stride_bytes = (ordered_bases[1] - ordered_bases[0]) if n_pieces > 1 else 0
    owners, starts, lens = [], [], []
    for band, base in enumerate(ordered_bases):
        owners.append(base_to_cores[base][0])
        starts.append(band * chunk)
        lens.append(chunk)
    return {
        "sdsc": sdsc_file, "lds_idx": lds_idx, "layout": layout,
        "split_dim": split_dim, "row_dim": row_dim,
        "out_total": out_total, "row_total": row_total,
        "n_pieces": n_pieces, "chunk": chunk,
        "chunk_sticks": chunk // stick, "stride_bytes": stride_bytes,
        "owners": owners, "starts": starts, "lens": lens,
        "owner_groups": {b: base_to_cores[b] for b in ordered_bases},
    }


def tiles_exactly(starts, lens, length):
    assert starts[0] == 0
    for i in range(1, len(starts)):
        assert starts[i] == starts[i - 1] + lens[i - 1], "gap/overlap"
    assert starts[-1] + lens[-1] == length, "does not reach length"
    return True


def derive_granite_edge(bundle_dir, prod_file, prod_idx, cons_file, cons_idx,
                        split_dim="out"):
    prod = derive_side(bundle_dir, prod_file, prod_idx, split_dim)
    cons = derive_side(bundle_dir, cons_file, cons_idx, split_dim)
    assert prod["out_total"] == cons["out_total"], (
        f"out_total mismatch prod={prod['out_total']} cons={cons['out_total']}"
    )
    tiles_exactly(prod["starts"], prod["lens"], prod["out_total"])
    tiles_exactly(cons["starts"], cons["lens"], cons["out_total"])
    return prod, cons


if __name__ == "__main__":
    bdir = sys.argv[1]
    prod, cons = derive_granite_edge(
        bdir, "sdsc_1_batchmatmul.json", 2, "sdsc_2_mul.json", 1, "out")
    print("=== PRODUCER (bmm OUTPUT idx2) ===")
    print(json.dumps({k: v for k, v in prod.items() if k != "owner_groups"},
                     indent=1))
    print("owner_groups:", prod["owner_groups"])
    print()
    print("=== CONSUMER (mul INPUT idx1) ===")
    print(json.dumps({k: v for k, v in cons.items() if k != "owner_groups"},
                     indent=1))
    print()
    print("RECONCILE: out_total =", prod["out_total"], "row =", prod["row_total"],
          "| prod", prod["n_pieces"], "bands x", prod["chunk_sticks"], "sticks",
          "| cons", cons["n_pieces"], "pieces x", cons["chunk_sticks"], "sticks")

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
"""Derive + classify the dispatch->consumer same-stick handoff edge from a
compiled MoE bundle. Reads the per-core HBM allocate node folds (the same source
derive_placement uses) and reports the real per-core placement: split dim, stick
dim, rows/core, and same-stick / same-shard verdict."""

import json
import sys


def dl_op(doc):
    b = doc[next(iter(doc))]
    dsc = b["dscs_"][0]
    return b, dsc[next(iter(dsc))]


def hbm_base(op, idx):
    for n in op["scheduleTree_"]:
        if n.get("nodeType_") == "allocate" and n.get("ldsIdx_") == idx:
            return next(iter(n["startAddressCoreCorelet_"]["data_"].values()))
    return None


def folds(op, idx, dim):
    for n in op["scheduleTree_"]:
        if n.get("nodeType_") == "allocate" and n.get("ldsIdx_") == idx:
            fa = n["coordinates_"]["coordInfo"][dim]["folds"]["dim_prop_attr"]
            extent, core_fold = 1, 1
            for a in fa:
                extent *= a["factor_"]
                if a["label_"] == "core_fold":
                    core_fold = a["factor_"]
            return extent, core_fold
    return None, None


def classify(prod_path, cons_path):
    pdoc = json.load(open(prod_path))
    cdoc = json.load(open(cons_path))
    pb, pop = dl_op(pdoc)
    cb, cop = dl_op(cdoc)
    p_out = pop["primaryDsInfo_"]["OUTPUT"]
    c_in = cop["primaryDsInfo_"]["INPUT"]
    p_addr = hbm_base(pop, 2)
    c_addr = hbm_base(cop, 0)
    # split dim = the dim with numWkSlicesPerDim > 1
    p_shard = pb["numWkSlicesPerDim_"]
    c_shard = cb["numWkSlicesPerDim_"]
    p_split = [d for d, v in p_shard.items() if v > 1][0]
    c_split = [d for d, v in c_shard.items() if v > 1][0]
    p_stick = p_out["stickDimOrder_"][0]
    c_stick = c_in["stickDimOrder_"][0]
    p_ext, p_cf = folds(pop, 2, p_split)
    p_rows = p_ext // p_cf
    out = {
        "shared_hbm_base": (p_addr, c_addr, p_addr == c_addr),
        "producer_output": {
            "layout": p_out["layoutDimOrder_"],
            "stick": p_stick,
            "split_dim": p_split,
            "n_cores": p_shard[p_split],
            "rows_per_core": p_rows,
        },
        "consumer_input": {
            "layout": c_in["layoutDimOrder_"],
            "stick": c_stick,
            "split_dim": c_split,
            "n_cores": c_shard[c_split],
        },
        # same-stick: both stick on the hidden axis (producer 'out' == consumer
        # 'in' == the 2048-dim). Names differ; physical stick dim is identical.
        "same_stick": (p_stick in ("out", "in")) and (c_stick in ("out", "in")),
        "same_shard": p_split == c_split and p_shard[p_split] == c_shard[c_split],
        "stick_is_split": p_stick == p_split,
    }
    return out


if __name__ == "__main__":
    print(json.dumps(classify(sys.argv[1], sys.argv[2]), indent=2))

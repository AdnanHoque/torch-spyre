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

"""Offline producer->consumer HBM-handoff classifier for torch-spyre SDSC bundles.

Reads bundle.mlir + per-SDSC JSON. Traces edges by matching per-core HBM base
addresses (from scheduleTree_ allocate nodes) between a producer's OUTPUT tensor
and a consumer's INPUT tensor. Classifies each edge same-stick vs layout-changing.
"""

import json
import sys
from pathlib import Path

import regex as re


def load(p):
    return json.loads(Path(p).read_text())


def parse_bundle_order(d):
    txt = (Path(d) / "bundle.mlir").read_text()
    return re.findall(r'sdsc_filename="([^"]+)"', txt)


def core0_addr(node):
    d = node.get("startAddressCoreCorelet_", {}).get("data_", {})
    v = d.get("[0, 0, 0]")
    return int(v) if v is not None else None


def all_core_addrs(node):
    d = node.get("startAddressCoreCorelet_", {}).get("data_", {})
    out = {}
    for k, v in d.items():
        m = re.match(r"\[(\d+), 0, 0\]", k)
        if m:
            out[int(m.group(1))] = int(v)
    return out


def parse_sdsc(path):
    """Return dict: op name, sharding, and per-tensor info keyed by ldsIdx."""
    doc = load(path)
    top = doc[list(doc.keys())[0]]
    shard = top.get("numWkSlicesPerDim_")
    ncores = top.get("numCoresUsed_")
    dl_dsc = top["dscs_"][0]
    op_key = list(dl_dsc.keys())[0]
    dl = dl_dsc[op_key]
    co = dl["computeOp_"][0]
    opfn = co.get("opFuncName")
    ins = co.get("inputLabeledDs", [])
    outs = co.get("outputLabeledDs", [])
    # map ldsIdx -> dsType_ (INPUT/OUTPUT/KERNEL role)
    idx_role = {}
    for lds in dl["labeledDs_"]:
        idx_role[lds["ldsIdx_"]] = lds.get("dsType_")
    # primaryDsInfo_ keyed by role -> stick/layout
    pdi = dl.get("primaryDsInfo_", {})
    # scheduleTree allocate nodes -> ldsIdx -> hbm core0 addr + percore
    alloc = {}
    for n in dl.get("scheduleTree_", []):
        if n.get("nodeType_") == "allocate":
            li = n.get("ldsIdx_")
            comp = n.get("component_")
            # prefer hbm component for edge tracing
            if li not in alloc or comp == "hbm":
                alloc[li] = {
                    "component": comp,
                    "core0": core0_addr(n),
                    "percore": all_core_addrs(n),
                    "layout": n.get("layoutDimOrder_"),
                }

    def label_idx(lbl):
        return int(lbl.rsplit("-idx", 1)[1])

    # Build tensor records. dsType_ tells role but multiple inputs may share a
    # role; primaryDsInfo is per-ROLE not per-tensor, so we attach pdi by role.
    def stick_for(idx):
        role = idx_role.get(idx)
        info = pdi.get(role, {})
        return {
            "role": role,
            "stick": info.get("stickDimOrder_"),
            "layout": info.get("layoutDimOrder_"),
            "sticksize": info.get("stickSize_"),
            "dimToStickSize": info.get("dimToStickSize_"),
        }

    inputs = []
    for lbl in ins:
        i = label_idx(lbl)
        rec = {"label": lbl, "idx": i}
        rec.update(stick_for(i))
        a = alloc.get(i, {})
        rec["hbm_core0"] = a.get("core0")
        rec["component"] = a.get("component")
        rec["alloc_layout"] = a.get("layout")
        rec["percore"] = a.get("percore")
        inputs.append(rec)
    outputs = []
    for lbl in outs:
        i = label_idx(lbl)
        rec = {"label": lbl, "idx": i}
        rec.update(stick_for(i))
        a = alloc.get(i, {})
        rec["hbm_core0"] = a.get("core0")
        rec["component"] = a.get("component")
        rec["alloc_layout"] = a.get("layout")
        rec["percore"] = a.get("percore")
        outputs.append(rec)
    return {
        "op": opfn,
        "shard": shard,
        "ncores": ncores,
        "inputs": inputs,
        "outputs": outputs,
        "op_key": op_key,
    }


def analyze(bundle_dir):
    order = parse_bundle_order(bundle_dir)
    sdscs = []
    for fn in order:
        rec = parse_sdsc(Path(bundle_dir) / fn)
        rec["file"] = fn
        sdscs.append(rec)
    return order, sdscs


def build_edges(sdscs):
    """Match producer OUTPUT hbm base -> consumer INPUT hbm base.

    Returns list of edges. Address must match AND consumer must come after
    producer.
    """
    edges = []
    # index: for each sdsc, its output hbm bases
    for ci, cons in enumerate(sdscs):
        for inp in cons["inputs"]:
            a = inp.get("hbm_core0")
            if a is None:
                continue
            # find latest prior producer whose any output has same addr
            prod_i = None
            prod_out = None
            for pi in range(ci - 1, -1, -1):
                for outp in sdscs[pi]["outputs"]:
                    if outp.get("hbm_core0") == a:
                        prod_i = pi
                        prod_out = outp
                        break
                if prod_i is not None:
                    break
            if prod_i is not None:
                edges.append(
                    {
                        "prod_i": prod_i,
                        "prod": sdscs[prod_i],
                        "prod_out": prod_out,
                        "cons_i": ci,
                        "cons": cons,
                        "cons_in": inp,
                        "addr": a,
                    }
                )
    return edges


if __name__ == "__main__":
    d = sys.argv[1]
    order, sdscs = analyze(d)
    print("BUNDLE:", d)
    print("ORDER:", order)
    print()
    for i, s in enumerate(sdscs):
        print("[%d] %s  op=%s shard=%s" % (i, s["file"], s["op"], s["shard"]))
        for r in s["inputs"]:
            print(
                "      IN  idx%s role=%s stick=%s layout=%s hbm0=%s comp=%s"
                % (
                    r["idx"],
                    r["role"],
                    r["stick"],
                    r["layout"],
                    r["hbm_core0"],
                    r["component"],
                )
            )
        for r in s["outputs"]:
            print(
                "      OUT idx%s role=%s stick=%s layout=%s hbm0=%s comp=%s"
                % (
                    r["idx"],
                    r["role"],
                    r["stick"],
                    r["layout"],
                    r["hbm_core0"],
                    r["component"],
                )
            )

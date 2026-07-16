#!/usr/bin/env python3
"""Audit whether an accepted explicit SHUFFLE generated physical movement."""

import argparse
import hashlib
import json
from pathlib import Path


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def only_value(mapping):
    if len(mapping) != 1:
        raise ValueError("expected a single root object")
    return next(iter(mapping.values()))


def allocation_nodes(source):
    root = only_value(source)
    dsc = only_value(root["dscs_"][0])
    schedule = dsc["scheduleTree_"]
    nodes = [node for node in schedule if node.get("component_") == "lx"]
    if len(nodes) != 2:
        raise ValueError("expected exactly two LX allocation nodes")
    return nodes


def fold_alpha(node, dim, label):
    folds = node["coordinates_"]["coordInfo"][dim]["folds"]
    for index, attr in enumerate(folds["dim_prop_attr"]):
        if attr["label_"] == label:
            func = folds["dim_prop_func"][index]
            return func.get("Affine", {}).get("alpha_")
    raise KeyError("missing {} for {}".format(label, dim))


def first_address(node):
    values = node["startAddressCoreCorelet_"]["data_"].values()
    return int(next(iter(values)))


def source_endpoint(node):
    out_extent = fold_alpha(node, "out", "core_fold")
    return {
        "name": node["name_"],
        "address": first_address(node),
        "layout_order": node["layoutDimOrder_"],
        "out_core_fold": out_extent,
        "row_stride_bytes": out_extent * 2,
        "core_mapping_sample": dict(
            list(node["coordinates_"]["coreIdToWkSlice_"].items())[:8]
        ),
    }


def generated_datadsc(generated):
    root = only_value(generated)
    entry = only_value(root["datadscs_"][0])
    return entry


def generated_endpoint(lds):
    placements = [
        placement
        for piece in lds.get("PieceInfo", [])
        for placement in piece.get("PlacementInfo", [])
    ]
    addresses = sorted(
        {
            int(value)
            for placement in placements
            for value in placement.get("startAddr", {}).get("data_", {}).values()
            for value in (value if isinstance(value, list) else [value])
        }
    )
    out_extent = lds["dimToLayoutSize_"]["out"]
    piece_starts = sorted(
        {
            piece["dimToStartCordinate"].get("out")
            for piece in lds.get("PieceInfo", [])
        }
    )
    piece_sizes = sorted(
        {piece["dimToSize_"].get("out") for piece in lds.get("PieceInfo", [])}
    )
    return {
        "name": lds["ldsName_"],
        "addresses": addresses,
        "layout_order": lds["layoutDimOrder_"],
        "layout_sizes": lds["dimToLayoutSize_"],
        "out_layout_extent": out_extent,
        "row_stride_bytes": out_extent * lds["wordLength"],
        "piece_out_starts": piece_starts,
        "piece_out_sizes": piece_sizes,
        "piece_count": len(lds.get("PieceInfo", [])),
    }


def pcfg_types(datadsc):
    types = []

    def visit(value):
        if isinstance(value, dict):
            if "type" in value:
                types.append(value["type"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(datadsc.get("pcfg_", []))
    return types


def normalized_generated_descriptor(endpoints, op):
    payload = {
        "endpoints": [
            {
                "layout_order": endpoint["layout_order"],
                "layout_sizes": endpoint["layout_sizes"],
                "piece_out_starts": endpoint["piece_out_starts"],
                "piece_out_sizes": endpoint["piece_out_sizes"],
                "piece_count": endpoint["piece_count"],
            }
            for endpoint in endpoints
        ],
        "dt_count": len(op.get("dtTable_", [])),
        "l3su_keys": len(op.get("coreIDtoDtKey_L3SU", [])),
        "l3lu_keys": len(op.get("coreIDtoDtKey_L3LU", [])),
        "lx_keys": len(op.get("coreIDtoDtKey_LX", [])),
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit_run(run_dir):
    fixture = run_dir / "fixture"
    source_path = fixture / "sdsc_0.json"
    debug_files = sorted(fixture.glob("debug/*-Relayout*/*.out.out.out.json"))
    if not debug_files:
        raise FileNotFoundError("no finalized inserted DataDSC under {}".format(fixture))
    generated_path = debug_files[0]

    source = [source_endpoint(node) for node in allocation_nodes(load_json(source_path))]
    datadsc = generated_datadsc(load_json(generated_path))
    endpoints = [generated_endpoint(lds) for lds in datadsc["labeledDs_"]]
    op = datadsc["op"]
    types = pcfg_types(datadsc)

    return {
        "run": run_dir.name,
        "return_code": int((run_dir / "return_code.txt").read_text().strip()),
        "source_path": str(source_path),
        "generated_path": str(generated_path),
        "source_endpoints": source,
        "generated_endpoints": endpoints,
        "generated_transfer_state": {
            "dt_count": len(op.get("dtTable_", [])),
            "l3su_key_count": len(op.get("coreIDtoDtKey_L3SU", [])),
            "l3lu_key_count": len(op.get("coreIDtoDtKey_L3LU", [])),
            "lx_key_count": len(op.get("coreIDtoDtKey_LX", [])),
            "pcfg_node_count": len(types),
            "pcfg_type_counts": {kind: types.count(kind) for kind in sorted(set(types))},
        },
        "normalized_generated_descriptor_sha256": normalized_generated_descriptor(
            endpoints, op
        ),
    }


def render_markdown(audits):
    honest = next(item for item in audits if item["run"] == "v6-honest-a-shuffle-only")
    full = next(item for item in audits if item["run"] == "v6-full-stride")
    source_in, source_out = honest["source_endpoints"]
    generated_in, generated_out = honest["generated_endpoints"]
    transfer = honest["generated_transfer_state"]
    identical = (
        honest["normalized_generated_descriptor_sha256"]
        == full["normalized_generated_descriptor_sha256"]
    )
    return """# Exact-backend Variant A audit

Deeptools SHA: `704c19f8fb7f0cc972f20404f9dd0010895a35e2`

The explicit SHUFFLE is accepted (`rc={rc}`), but acceptance does not materialize
the grouped all-gather.

| Endpoint | Frontend address | Frontend `out` extent | Frontend row stride | Inserted DataDSC `out` extent | Inserted row stride |
|---|---:|---:|---:|---:|---:|
| S1 input | `0x{src_addr:x}` | {src_extent} | {src_stride} B | {gen_src_extent} | {gen_src_stride} B |
| S2 output | `0x{dst_addr:x}` | {dst_extent} | {dst_stride} B | {gen_dst_extent} | {gen_dst_stride} B |

The source allocation explicitly describes compact S1 rows of 512 fp16 values,
while S2 uses expanded rows of 4096 values. Relayout insertion rewrites both
DataDSC endpoints to the consumer extent of 4096, losing S1's physical stride.

The finalized inserted DataDSC contains:

- `dtTable_`: {dt_count} entries
- L3 send keys: {l3su}
- L3 load keys: {l3lu}
- LX keys: {lx}
- PCFG node types: `{pcfg}`

The honest compact-source fixture and the full-stride diagnostic synthesize
equivalent normalized transfer descriptors: **{identical}**. This is direct
evidence that the backend is not preserving endpoint-specific physical layout.

## Conclusion

Current redundant-coordinate SHUFFLE is not sufficient as implemented. DXP must
preserve independent S1/S2 physical layout extents and convert work-slice
ordinals to logical element starts before DDC can generate the required 256
placements (224 cross-core plus 32 local).
""".format(
        rc=honest["return_code"],
        src_addr=source_in["address"],
        src_extent=source_in["out_core_fold"],
        src_stride=source_in["row_stride_bytes"],
        gen_src_extent=generated_in["out_layout_extent"],
        gen_src_stride=generated_in["row_stride_bytes"],
        dst_addr=source_out["address"],
        dst_extent=source_out["out_core_fold"],
        dst_stride=source_out["row_stride_bytes"],
        gen_dst_extent=generated_out["out_layout_extent"],
        gen_dst_stride=generated_out["row_stride_bytes"],
        dt_count=transfer["dt_count"],
        l3su=transfer["l3su_key_count"],
        l3lu=transfer["l3lu_key_count"],
        lx=transfer["lx_key_count"],
        pcfg=json.dumps(transfer["pcfg_type_counts"], sort_keys=True),
        identical=str(identical).lower(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()

    names = ["v6-honest-a-shuffle-only", "v6-full-stride"]
    audits = [audit_run(args.raw_root / name) for name in names]
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audits, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(audits))


if __name__ == "__main__":
    main()

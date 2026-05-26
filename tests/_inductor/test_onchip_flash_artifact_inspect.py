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

"""Tests for the mixed flash-pipeline artifact inspection helper."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))


def _load_tool():
    path = os.path.join(_ROOT, "tools", "onchip_flash_artifact_inspect.py")
    spec = importlib.util.spec_from_file_location(
        "onchip_flash_artifact_inspect", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tool = _load_tool()


def _labeled_ds(start_addr):
    return {
        "ldsName_": "",
        "segment_": "output",
        "layoutDimOrder_": ["mb_", "x_", "out_"],
        "stickDimOrder_": ["out_"],
        "PieceInfo": [
            {
                "dimToStartCordinate": {"mb_": 0, "x_": 0, "out_": 0},
                "dimToSize_": {"mb_": 4, "x_": 2, "out_": 64},
                "PlacementInfo": [
                    {"type": "lx", "memId": [0], "startAddr": [start_addr]}
                ],
            }
        ],
    }


def _source_dataop(index, corelet_id):
    lane = "k" if index % 2 == 0 else "v"
    tile = index // 2
    return {
        f"{index}_STCDPOpLx_prefetch_{lane}_tile{tile}": {
            "op": {"name": "STCDPOpLx", "coreletId": corelet_id},
            "labeledDs_": [
                _labeled_ds(16000 + index * 1024),
                _labeled_ds(17000 + index * 1024),
            ],
        }
    }


def _debug_dataop(index, source_corelet_id, routed_corelet_id):
    name, body = next(iter(_source_dataop(index, source_corelet_id).items()))
    body = dict(body)
    body["pcfg_"] = [
        {
            f"lxlu{routed_corelet_id}": [
                {
                    "name": f"c0-lxlu{routed_corelet_id}-ringDT-lx-pe-0-{index}",
                    "type": "datatransfer",
                    "coreletId": -1,
                    "srcDest": ["lx", f"pe{routed_corelet_id}"],
                    "srcStartAddr": str(16000 + index * 1024),
                    "destStartAddr": "-1",
                    "dimLayoutOrder": ["out_x_mb_"],
                }
            ],
            f"lxsu{routed_corelet_id}": [
                {
                    "name": f"c0-lxsu{routed_corelet_id}-ringDT-pe-lx-0-{index}",
                    "type": "datatransfer",
                    "coreletId": -1,
                    "srcDest": [f"pe{routed_corelet_id}", "lx"],
                    "srcStartAddr": "-1",
                    "destStartAddr": str(17000 + index * 1024),
                    "dimLayoutOrder": ["out_x_mb_"],
                }
            ],
            f"pe{routed_corelet_id}": [
                {
                    "name": f"c0-pe{routed_corelet_id}-FIFO-lx-lx-{index}",
                    "type": "ptsfpdatatransfer",
                    "self": f"pe{routed_corelet_id}",
                    "coreletId": routed_corelet_id,
                    "srcDest": ["lx", "lx"],
                    "srcStartAddr": "-1",
                    "destStartAddr": "-1",
                    "dimLayoutOrder": [],
                }
            ],
        }
    ]
    return {name: body}


def _write_fixture(tmp, routed_corelet_id=1):
    cache = Path(tmp)
    graph = cache / "inductor-spyre" / "graph0"
    debug = graph / "debug" / "sdsc_mixed_flash_pipeline_tile_0"
    debug.mkdir(parents=True)
    source = {
        "mixed_flash_pipeline_tile_0": {
            "numCoresUsed_": 32,
            "coreIdToDscSchedule": {
                "0": [[0, -1, 0, 1], [1, -1, 1, 1], [2, 0, 1, 1], [3, -1, 1, 0]]
            },
            "datadscs_": [_source_dataop(index, 1) for index in range(4)],
            "dscs_": [
                {
                    "batchmatmul": {
                        "numCoreletsUsed_": 1,
                        "coreIdsUsed_": [0],
                        "scheduleTree_": [
                            {"nodeType_": "allocate", "component_": "hbm"}
                        ],
                    }
                }
            ],
            "opFuncsUsed_": ["STCDPOpLx"] * 4,
            "flashAttentionPipeline_": {
                "tile_count": 1,
                "dataop_count": 4,
                "tile_index": 0,
                "replaces_sdsc": "15_batchmatmul",
                "overlap_prefix": True,
                "overlap_candidate": True,
                "prefetch_corelet_id": 1,
            },
        }
    }
    final = {
        "mixed_flash_pipeline_tile_0": {
            "coreIdToDscSchedule": source["mixed_flash_pipeline_tile_0"][
                "coreIdToDscSchedule"
            ],
            "datadscs_": [
                _debug_dataop(index, 1, routed_corelet_id) for index in range(4)
            ],
        }
    }
    (graph / "sdsc_mixed_flash_pipeline_tile_0.json").write_text(
        json.dumps(source)
    )
    (debug / "sdsc_mixed_flash_pipeline_tile_0.out.out.out.json").write_text(
        json.dumps(final)
    )
    (debug / "senprog.txt").write_text("LXLU:0:1\nLXSU:0:1\n")
    return cache


def test_inspector_reports_source_schedule_addresses_and_debug_corelet_route():
    with tempfile.TemporaryDirectory() as tmp:
        cache = _write_fixture(tmp, routed_corelet_id=1)
        report = tool.inspect_inputs([cache], None)
        diagnostics = tool.validate_report(report, 1, True)

    tile = report["reports"][0]["tiles"][0]
    assert diagnostics == []
    assert tile["overlapRows"] == [{"core": "0", "rowIndex": 2, "row": [2, 0, 1, 1]}]
    assert tile["dataops"][0]["labeledDs"][1]["first"]["startAddr"] == [17000]
    assert tile["debug"]["componentTotals"] == {"lxlu1": 4, "lxsu1": 4, "pe1": 4}
    assert tile["debug"]["transferCoreletIds"] == {"-1": 8, "1": 4}


def test_inspector_flags_debug_route_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        cache = _write_fixture(tmp, routed_corelet_id=0)
        report = tool.inspect_inputs([cache], None)
        diagnostics = tool.validate_report(report, 1, True)

    assert diagnostics == [
        "inductor-spyre/graph0/sdsc_mixed_flash_pipeline_tile_0.json: "
        "missing routed debug components ['lxlu1', 'lxsu1', 'pe1']",
        "no overlap-prefix DXP debug JSON with routed components found",
    ]


def _run_all():
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    fails = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            fails.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - len(fails)}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())

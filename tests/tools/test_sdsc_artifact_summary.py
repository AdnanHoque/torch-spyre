from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_tool():
    root = Path(__file__).resolve().parents[2]
    path = root / "tools" / "sdsc_artifact_summary.py"
    spec = importlib.util.spec_from_file_location("sdsc_artifact_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def test_sdsc_artifact_summary_outputs_table_diff_and_trace(tmp_path: Path) -> None:
    tool = _load_tool()
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    out = tmp_path / "out"
    trace_dir = tmp_path / "trace"
    current.mkdir()
    baseline.mkdir()
    trace_dir.mkdir()

    _write(
        baseline / "sdsc_0.json",
        {
            "0_ReStickifyOpHBM": {
                "numCoresUsed_": 25,
                "coreIdToWkSlice_": {"0": {"mb": 0, "out": 0}},
                "coreIdToDscSchedule": {"0": [[-1, 0, 0, 0]]},
                "dscs_": [
                    {
                        "ReStickifyOpHBM": {
                            "numCoresUsed_": 25,
                            "N_": {"mb": 256, "out": 512},
                            "primaryDsInfo_": {
                                "INPUT": {"layoutDimOrder_": ["mb", "out"]},
                                "OUTPUT": {"layoutDimOrder_": ["out", "mb"]},
                            },
                            "labeledDs_": [
                                {
                                    "ldsIdx_": 0,
                                    "dsName_": "Tensor0",
                                    "dsType_": "INPUT",
                                    "memOrg_": {"hbm": {"isPresent": 1}},
                                }
                            ],
                            "scheduleTree_": [
                                {
                                    "ldsIdx_": 0,
                                    "startAddressCoreCorelet_": {
                                        "data_": {"[0, 0, 0]": "17179869184"}
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        },
    )
    _write(
        current / "sdsc_1.json",
        {
            "1_OnChipMoveCoordinateRemap": {
                "numCoresUsed_": 32,
                "coreIdToWkSlice_": {
                    "0": {"mb": 0, "out": 0},
                    "1": {"mb": 1, "out": 0},
                },
                "coreIdToDscSchedule": {
                    "0": [[0, -1, 0, 1], [-1, 0, 1, 0]],
                    "1": [[0, -1, 0, 1], [-1, 0, 1, 0]],
                },
                "datadscs_": [
                    {
                        "0_OnChipMoveLXCoordinateRemapOp_0": {
                            "op": {"name": "LXCoordinateRemapOp"},
                            "coverage": {"device_sizes": [2, 8, 64]},
                            "lowering": {"coalescedMovements": 1},
                            "coreIdsUsed_": [0, 1],
                            "movements": [
                                {
                                    "bytes": 128,
                                    "source": {"lxAddress": 0},
                                    "destination": {"lxAddress": 1048576},
                                }
                            ],
                        }
                    }
                ],
                "dscs_": [
                    {
                        "neg": {
                            "numCoresUsed_": 32,
                            "N_": {"mb": 256, "out": 512},
                            "primaryDsInfo_": {
                                "INPUT": {"layoutDimOrder_": ["mb", "out"]},
                                "OUTPUT": {"layoutDimOrder_": ["mb", "out"]},
                            },
                            "labeledDs_": [
                                {
                                    "ldsIdx_": 0,
                                    "dsName_": "Tensor0",
                                    "dsType_": "INPUT",
                                    "memOrg_": {"lx": {"isPresent": 1}},
                                }
                            ],
                            "scheduleTree_": [
                                {
                                    "ldsIdx_": 0,
                                    "startAddressCoreCorelet_": {
                                        "data_": {"[0, 0, 0]": 1048576}
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        },
    )
    _write(
        trace_dir / "sample.pt.trace.json",
        {
            "traceEvents": [
                {"cat": "kernel", "name": "sdsc_1", "dur": 3000},
                {"cat": "gpu_memcpy", "name": "copy", "dur": 1000},
            ]
        },
    )

    rc = tool.main(
        [
            "--sdsc-dir",
            str(current),
            "--baseline-sdsc-dir",
            str(baseline),
            "--trace-dir",
            str(trace_dir),
            "--active-iters",
            "3",
            "--output-dir",
            str(out),
        ]
    )

    assert rc == 0
    table = (out / "sdsc_table.md").read_text(encoding="utf-8")
    diff = (out / "sdsc_diff.md").read_text(encoding="utf-8")
    trace = json.loads((out / "trace_summary.json").read_text(encoding="utf-8"))
    assert "LXCoordinateRemapOp" in table
    assert "lx->lx" in table
    assert "mb=core_id" in table
    assert "ReStickifyOpHBM" in diff
    assert "| remap_bytes | 0 | 128 |" in diff
    assert trace["kernel_ms_per_iter"] == 1.0


def test_sdsc_artifact_summary_can_archive_legacy_senprog_summary(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    current = tmp_path / "current"
    out = tmp_path / "out"
    current.mkdir()
    _write(
        current / "sdsc_0.json",
        {
            "0_add": {
                "numCoresUsed_": 1,
                "coreIdToWkSlice_": {"0": {"mb": 0}},
                "coreIdToDscSchedule": {"0": [[-1, 0, 0, 0]]},
                "dscs_": [{"add": {"numCoresUsed_": 1}}],
            }
        },
    )
    legacy = tmp_path / "sdsc_senprog_summary.py"
    legacy.write_text(
        "\n".join(
            [
                "import argparse, json",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--output-dir', type=Path, required=True)",
                "parser.add_argument('--dcc')",
                "parser.add_argument('--sdsc', action='append', default=[])",
                "args = parser.parse_args()",
                "args.output_dir.mkdir(parents=True, exist_ok=True)",
                "summary = {'rows': [{'sdsc': args.sdsc[0], 'returncode': 0}]}",
                "(args.output_dir / 'summary.json').write_text(json.dumps(summary))",
                "print(json.dumps(summary))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = tool.main(
        [
            "--sdsc-dir",
            str(current),
            "--sdsc-senprog-summary",
            str(legacy),
            "--output-dir",
            str(out),
        ]
    )

    assert rc == 0
    archived = json.loads(
        (out / "sdsc_senprog_summary" / "summary.json").read_text(encoding="utf-8")
    )
    command = json.loads(
        (out / "sdsc_senprog_summary" / "command.json").read_text(encoding="utf-8")
    )
    assert archived["rows"][0]["returncode"] == 0
    assert command[0] == sys.executable

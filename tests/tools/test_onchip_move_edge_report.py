from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_tool():
    root = Path(__file__).resolve().parents[2]
    path = root / "tools" / "onchip_move_edge_report.py"
    spec = importlib.util.spec_from_file_location("onchip_move_edge_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_onchip_move_edge_report_classifies_planned_and_fanout(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    jsonl = tmp_path / "onchip_move.jsonl"
    rows = [
        {
            "status": "planned",
            "producer_op": "batchmatmul",
            "consumer_op": "neg",
            "producer": "buf0",
            "consumer": "buf1",
            "bytes_moved": 128,
            "cell_count": 1,
            "producer_view": {
                "work_slice_dims": [{"device_dim": 0, "split": 4}],
                "core_to_slot": [{"device_dim": 0, "slot_expr": "Mod(core_id,4)"}],
            },
            "consumer_view": {
                "work_slice_dims": [{"device_dim": 0, "split": 32}],
                "core_to_slot": [{"device_dim": 0, "slot_expr": "core_id"}],
            },
        },
        {
            "status": "skipped",
            "producer_op": "mul",
            "consumer_op": "batchmatmul",
            "producer": "buf5",
            "consumer": "buf6",
            "fallback_reason": "consumer-duplicate-owner",
        },
    ]
    jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    rc = tool.main(
        [
            "--jsonl",
            str(jsonl),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )

    assert rc == 0
    report = (tmp_path / "artifacts" / "onchip_move_edge_report.md").read_text(
        encoding="utf-8"
    )
    assert "`exact-reshard`" in report
    assert "`fanout-multicast-unsupported`" in report
    with (tmp_path / "artifacts" / "onchip_move_edge_report.csv").open(
        encoding="utf-8"
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert {row["communication_class"] for row in csv_rows} == {
        "exact-reshard",
        "fanout-multicast-unsupported",
    }

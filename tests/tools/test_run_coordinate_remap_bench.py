from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_tool():
    root = Path(__file__).resolve().parents[2]
    path = root / "tools" / "run_coordinate_remap_bench.py"
    spec = importlib.util.spec_from_file_location("run_coordinate_remap_bench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_coordinate_remap_bench_dry_run_writes_variant_commands(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    torch_root = Path(__file__).resolve().parents[2]
    deeptools_root = tmp_path / "deeptools"
    perf_root = tmp_path / "perf"
    deeptools_root.mkdir()
    perf_root.mkdir()
    (perf_root / "benchmark.py").write_text("# placeholder\n", encoding="utf-8")

    rc = tool.main(
        [
            "--output-root",
            str(tmp_path / "runs"),
            "--torch-root",
            str(torch_root),
            "--deeptools-root",
            str(deeptools_root),
            "--perf-suite-root",
            str(perf_root),
            "--variant",
            "coordinate-remap",
            "--shape",
            "1",
            "512",
            "4096",
            "--dry-run",
        ]
    )

    assert rc == 0
    command_file = tmp_path / "runs" / "coordinate-remap" / "commands.json"
    env_file = tmp_path / "runs" / "coordinate-remap" / "env.json"
    commands = json.loads(command_file.read_text(encoding="utf-8"))
    env = json.loads(env_file.read_text(encoding="utf-8"))
    assert commands["benchmark"][commands["benchmark"].index("--op") + 1] == "mlp"
    assert commands["artifact_summary"]
    assert env["env"]["SPYRE_ONCHIP_MOVE_REALIZE"] == "1"
    assert env["shape"] == [1, 512, 4096]


def test_run_coordinate_remap_bench_dry_run_can_label_upstream_main(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    torch_root = Path(__file__).resolve().parents[2]
    main_root = tmp_path / "torch-main"
    deeptools_root = tmp_path / "deeptools"
    perf_root = tmp_path / "perf"
    main_root.mkdir()
    deeptools_root.mkdir()
    perf_root.mkdir()
    (perf_root / "benchmark.py").write_text("# placeholder\n", encoding="utf-8")

    rc = tool.main(
        [
            "--output-root",
            str(tmp_path / "runs"),
            "--torch-root",
            str(torch_root),
            "--main-torch-root",
            str(main_root),
            "--deeptools-root",
            str(deeptools_root),
            "--perf-suite-root",
            str(perf_root),
            "--variant",
            "upstream-main",
            "--emit-sdsc-senprog-summary",
            "--sdsc-senprog-summary",
            str(tmp_path / "sdsc_senprog_summary.py"),
            "--dry-run",
        ]
    )

    assert rc == 0
    env = json.loads(
        (tmp_path / "runs" / "upstream-main" / "env.json").read_text(
            encoding="utf-8"
        )
    )
    commands = json.loads(
        (tmp_path / "runs" / "upstream-main" / "commands.json").read_text(
            encoding="utf-8"
        )
    )
    assert env["torch_root"] == str(main_root)
    assert env["artifact_tool_root"] == str(torch_root)
    assert env["env"]["SPYRE_ONCHIP_MOVE_PLANNER"] == "0"
    assert "--sdsc-senprog-summary" in commands["artifact_summary"]

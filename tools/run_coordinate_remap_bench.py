#!/usr/bin/env python3
"""Run coordinate-remap benchmark variants and archive required artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    name: str
    env: dict[str, str]


BRANCH_VARIANTS = [
    Variant(
        "upstream-main",
        {
            "SPYRE_ONCHIP_MOVE_PLANNER": "0",
            "SPYRE_ONCHIP_MOVE_REALIZE": "0",
        },
    ),
    Variant(
        "branch-baseline",
        {
            "SPYRE_ONCHIP_MOVE_PLANNER": "0",
            "SPYRE_ONCHIP_MOVE_REALIZE": "0",
        },
    ),
    Variant(
        "planned-only",
        {
            "SPYRE_ONCHIP_MOVE_PLANNER": "1",
            "SPYRE_ONCHIP_MOVE_REALIZE": "0",
            "SPYRE_ONCHIP_MOVE_CARRIER": "coordinate_remap",
        },
    ),
    Variant(
        "coordinate-remap",
        {
            "SPYRE_ONCHIP_MOVE_PLANNER": "1",
            "SPYRE_ONCHIP_MOVE_REALIZE": "1",
            "SPYRE_ONCHIP_MOVE_CARRIER": "coordinate_remap",
            "SPYRE_ONCHIP_MOVE_COORDINATE_REMAP_CHUNK_CELLS": "512",
            "SPYRE_ONCHIP_MOVE_MAX_CELLS": "65536",
        },
    ),
]


def _repo_sha(path: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _repo_branch(path: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(path), "branch", "--show-current"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _prepend(existing: str | None, values: list[Path]) -> str:
    parts = [str(value) for value in values if value]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _ld_library_path(deeptools_root: Path, existing: str | None) -> str:
    candidates = []
    for build_dir in sorted(deeptools_root.glob("build*")):
        for child in [
            build_dir / "dsm",
            build_dir / "dsm" / "translators" / "perfDscToSdsc",
            build_dir / "dvs",
            build_dir / "deeprt",
            build_dir / "lib",
            build_dir / "lib64",
            build_dir,
        ]:
            if child.exists():
                candidates.append(child)
    for path in [
        Path("/opt/ibm/spyre/runtime/lib"),
        Path("/opt/ibm/spyre/deeptools/lib"),
        Path("/opt/ibm/spyre/senlib/lib"),
        Path("/opt/ibm/spyre/sentinyexec/lib"),
    ]:
        if path.exists():
            candidates.append(path)
    return _prepend(existing, candidates)


def _base_env(args: argparse.Namespace, run_dir: Path, variant: Variant) -> dict[str, str]:
    env = os.environ.copy()
    cache_dir = run_dir / "inductor-cache"
    torch_root = _runtime_torch_root(args, variant)
    env.update(
        {
            "PYTHONPATH": _prepend(
                env.get("PYTHONPATH"), [torch_root, torch_root / "tests/inductor"]
            ),
            "DEEPTOOLS_PATH": str(args.deeptools_root),
            "PATH": _prepend(
                env.get("PATH"),
                [
                    args.deeptools_root / "build-swiglu-dxp-main-lean" / "dxp",
                    args.deeptools_root / "build" / "dxp",
                ],
            ),
            "LD_LIBRARY_PATH": _ld_library_path(
                args.deeptools_root, env.get("LD_LIBRARY_PATH")
            ),
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(cache_dir),
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
            "SPYRE_ONCHIP_MOVE_JSONL": str(run_dir / "onchip_move.jsonl"),
            "SPYRE_ONCHIP_MOVE_DEBUG_DIR": str(run_dir / "onchip_move_debug"),
        }
    )
    env.update(variant.env)
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--env must be KEY=VALUE, got {item!r}")
        env[key] = value
    return env


def _benchmark_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    if args.command:
        return [part.format(run_dir=str(run_dir)) for part in args.command]
    benchmark = args.perf_suite_root / "benchmark.py"
    shape_args = [str(value) for value in args.shape]
    return [
        sys.executable,
        str(benchmark),
        "--stack",
        "torch-spyre",
        "--op",
        args.op,
        "--shape",
        *shape_args,
        "--runs",
        str(args.runs),
        "--without-compilation",
        "--with-profiling",
        "--output",
        str(run_dir / "perf.txt"),
    ]


def _write_env_record(
    path: Path, args: argparse.Namespace, variant: Variant, env: dict[str, str]
) -> None:
    keys = [
        "PYTHONPATH",
        "DEEPTOOLS_PATH",
        "PATH",
        "LD_LIBRARY_PATH",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCHINDUCTOR_FX_GRAPH_CACHE",
        "SPYRE_ONCHIP_MOVE_PLANNER",
        "SPYRE_ONCHIP_MOVE_REALIZE",
        "SPYRE_ONCHIP_MOVE_CARRIER",
        "SPYRE_ONCHIP_MOVE_COORDINATE_REMAP_CHUNK_CELLS",
        "SPYRE_ONCHIP_MOVE_MAX_CELLS",
        "SPYRE_ONCHIP_MOVE_JSONL",
        "SPYRE_ONCHIP_MOVE_DEBUG_DIR",
        "SPYRE_SMALL_SWIGLU_MODE",
    ]
    torch_root = _runtime_torch_root(args, variant)
    payload = {
        "variant": variant.name,
        "torch_root": str(torch_root),
        "torch_branch": _repo_branch(torch_root),
        "torch_sha": _repo_sha(torch_root),
        "artifact_tool_root": str(args.torch_root),
        "artifact_tool_branch": _repo_branch(args.torch_root),
        "artifact_tool_sha": _repo_sha(args.torch_root),
        "deeptools_root": str(args.deeptools_root),
        "deeptools_branch": _repo_branch(args.deeptools_root),
        "deeptools_sha": _repo_sha(args.deeptools_root),
        "perf_suite_root": str(args.perf_suite_root),
        "perf_suite_branch": _repo_branch(args.perf_suite_root),
        "perf_suite_sha": _repo_sha(args.perf_suite_root),
        "op": args.op,
        "shape": args.shape,
        "runs": args.runs,
        "env": {key: env.get(key, "") for key in keys},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(args.torch_root / "tools" / "sdsc_artifact_summary.py"),
        "--sdsc-dir",
        str(run_dir / "inductor-cache" / "inductor-spyre"),
        "--trace-dir",
        str(run_dir / "logs"),
        "--active-iters",
        str(max(args.runs - 1, 1)),
        "--output-dir",
        str(run_dir / "artifacts"),
    ]
    if args.baseline_sdsc_dir:
        command.extend(["--baseline-sdsc-dir", str(args.baseline_sdsc_dir)])
    if args.emit_senprog:
        command.append("--emit-senprog")
        command.extend(["--dcc", args.dcc])
    if args.emit_sdsc_senprog_summary:
        if args.sdsc_senprog_summary is None:
            raise ValueError(
                "--emit-sdsc-senprog-summary requires --sdsc-senprog-summary"
            )
        command.extend(["--sdsc-senprog-summary", str(args.sdsc_senprog_summary)])
    return command


def _run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
        handle.flush()
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        handle.write(f"\nreturncode={proc.returncode}\n")
    return proc.returncode


def _variants(args: argparse.Namespace) -> list[Variant]:
    default_variants = [
        variant.name for variant in BRANCH_VARIANTS if variant.name != "upstream-main"
    ]
    selected = set(args.variant or default_variants)
    variants = [variant for variant in BRANCH_VARIANTS if variant.name in selected]
    unknown = selected - {variant.name for variant in BRANCH_VARIANTS}
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(sorted(unknown))}")
    if "upstream-main" in selected and args.main_torch_root is None:
        raise ValueError("--variant upstream-main requires --main-torch-root")
    return variants


def _runtime_torch_root(args: argparse.Namespace, variant: Variant) -> Path:
    if variant.name == "upstream-main":
        if args.main_torch_root is None:
            raise ValueError("--variant upstream-main requires --main-torch-root")
        return args.main_torch_root
    return args.torch_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--torch-root", type=Path, default=Path.cwd())
    parser.add_argument("--main-torch-root", type=Path)
    parser.add_argument(
        "--deeptools-root", type=Path, default=Path("/tmp/deeptools-coordinate-remap-mainport-lean")
    )
    parser.add_argument("--perf-suite-root", type=Path, default=Path("/home/adnan-cdx/spyre-perf-suite"))
    parser.add_argument("--baseline-sdsc-dir", type=Path)
    parser.add_argument("--variant", action="append", choices=[v.name for v in BRANCH_VARIANTS])
    parser.add_argument("--op", default="mlp")
    parser.add_argument("--shape", type=int, nargs="+", default=[1, 512, 4096])
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--env", action="append", default=[], help="Extra KEY=VALUE env")
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    parser.add_argument("--emit-senprog", action="store_true")
    parser.add_argument("--emit-sdsc-senprog-summary", action="store_true")
    parser.add_argument("--sdsc-senprog-summary", type=Path)
    parser.add_argument("--dcc", default="dcc_standalone")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for variant in _variants(args):
        run_dir = args.output_root / variant.name
        run_dir.mkdir(parents=True, exist_ok=True)
        env = _base_env(args, run_dir, variant)
        command = _benchmark_command(args, run_dir)
        artifact_command = _artifact_command(args, run_dir)
        _write_env_record(run_dir / "env.json", args, variant, env)
        (run_dir / "commands.json").write_text(
            json.dumps(
                {"benchmark": command, "artifact_summary": artifact_command},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest.append({"variant": variant.name, "run_dir": str(run_dir)})
        if args.dry_run:
            continue
        rc = _run(command, cwd=run_dir, env=env, log=run_dir / "benchmark.log")
        if rc != 0:
            return rc
        rc = _run(
            artifact_command,
            cwd=args.torch_root,
            env=env,
            log=run_dir / "artifact_summary.log",
        )
        if rc != 0:
            return rc
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

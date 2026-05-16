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

"""Run restickify probes with aiu-smi sampling only around the timed loop.

The hierarchy sweep is good for compiler telemetry and wall-clock timing, but
wrapping the whole process with aiu-smi also samples compile, allocation, and
warmup noise.  This script compiles once, warms up, starts aiu-smi, runs only the
timed iterations, then stops aiu-smi and summarizes the counter CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))

import restickify_hierarchy_sweep as hierarchy
import restickify_scenario_probe as probe


torch = None

AIUSMI_RATE_KEYS = (
    "rdmem GiB/s",
    "wrmem GiB/s",
    "rxpci GiB/s",
    "txpci GiB/s",
    "rdrdma GiB/s",
    "wrrdma GiB/s",
    "n_rdmem Mreq/s",
    "n_wrmem Mreq/s",
    "n_rxpci Mreq/s",
    "n_txpci Mreq/s",
    "busy %",
)


def _find_case(name: str) -> hierarchy.HierarchyCase:
    for case in hierarchy.CASES:
        if case.name == name:
            return case
    available = ", ".join(case.name for case in hierarchy.CASES)
    raise SystemExit(f"unknown case {name!r}; available cases: {available}")


@contextmanager
def _restickify_capture(telemetry_path: Path | None):
    import torch_spyre._inductor.insert_restickify as insert_restickify

    insert_restickify.restickify_plan = {}
    previous_capture = os.environ.get("SPYRE_CAPTURE_RESTICKIFY_PLAN")
    previous_ring = os.environ.get("SPYRE_RESTICKIFY_RING_TELEMETRY")
    previous_ring_jsonl = os.environ.get("SPYRE_RESTICKIFY_RING_TELEMETRY_JSONL")
    spyre_config = None
    previous_config_ring = None
    previous_config_ring_jsonl = None

    os.environ["SPYRE_CAPTURE_RESTICKIFY_PLAN"] = "1"
    if telemetry_path is not None:
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_path.unlink(missing_ok=True)
        os.environ["SPYRE_RESTICKIFY_RING_TELEMETRY"] = "1"
        os.environ["SPYRE_RESTICKIFY_RING_TELEMETRY_JSONL"] = str(telemetry_path)
        try:
            import torch_spyre._inductor.config as spyre_config_module

            spyre_config = spyre_config_module
            previous_config_ring = spyre_config.restickify_ring_telemetry
            previous_config_ring_jsonl = spyre_config.restickify_ring_telemetry_jsonl
            spyre_config.restickify_ring_telemetry = True
            spyre_config.restickify_ring_telemetry_jsonl = str(telemetry_path)
        except Exception:
            spyre_config = None

    try:
        yield insert_restickify
    finally:
        if previous_capture is None:
            os.environ.pop("SPYRE_CAPTURE_RESTICKIFY_PLAN", None)
        else:
            os.environ["SPYRE_CAPTURE_RESTICKIFY_PLAN"] = previous_capture
        if previous_ring is None:
            os.environ.pop("SPYRE_RESTICKIFY_RING_TELEMETRY", None)
        else:
            os.environ["SPYRE_RESTICKIFY_RING_TELEMETRY"] = previous_ring
        if previous_ring_jsonl is None:
            os.environ.pop("SPYRE_RESTICKIFY_RING_TELEMETRY_JSONL", None)
        else:
            os.environ["SPYRE_RESTICKIFY_RING_TELEMETRY_JSONL"] = previous_ring_jsonl
        if spyre_config is not None:
            spyre_config.restickify_ring_telemetry = previous_config_ring
            spyre_config.restickify_ring_telemetry_jsonl = previous_config_ring_jsonl


def _ensure_aiusmi_env(metric_file: Path, export_dir: Path) -> None:
    """Set defaults needed for the aiu-monitor wheel to emit metrics."""
    default_config = Path(sys.prefix) / "etc" / "senlib_config_aiusmi.json"
    if "SENLIB_DEVEL_CONFIG_FILE" not in os.environ and default_config.exists():
        os.environ["SENLIB_DEVEL_CONFIG_FILE"] = str(default_config)
    os.environ.setdefault("AIUPTI_ENABLE_METRICS", "1")
    os.environ.setdefault("AIUSMI_ENABLE_METRICS", "1")
    os.environ.setdefault("ENABLE_AIUPTI_ACTIVITY_KIND_EVENT", "1")
    os.environ.setdefault("ENABLE_AIUPTI_ACTIVITY_KIND_METRIC", "1")
    os.environ.setdefault("AIUPTI_SAMPLER_INTERVAL", "1")
    os.environ.setdefault("DTCOMPILER_EXPORT_DIR", str(export_dir))

    # The aiu-monitor config currently points at /tmp/metrics.%BUSID.  These
    # env vars are useful if the runtime honors them in a future build; today
    # the default metric path is still the one that matters.
    os.environ.setdefault("AIUPTI_METRIC_PATH", str(metric_file))
    os.environ.setdefault("SPYRE_METRIC_PATH", str(metric_file))


def _start_aiusmi(aiusmi: str, csv_path: Path, interval: float) -> subprocess.Popen:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            aiusmi,
            "-s",
            "-g",
            "A",
            "-d",
            str(interval),
            "-f",
            str(csv_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _stop_aiusmi(process: subprocess.Popen) -> str:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        _, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        _, stderr = process.communicate()
    return stderr or ""


def _read_aiusmi_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "aiusmi_csv_path": str(path),
            "aiusmi_sample_rows": 0,
            "aiusmi_traffic_samples": 0,
            "aiusmi_error": "csv missing",
        }

    rows = []
    with path.open(encoding="utf-8") as csv_file:
        for line in csv_file:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line)
    if not rows:
        return {
            "aiusmi_csv_path": str(path),
            "aiusmi_sample_rows": 0,
            "aiusmi_traffic_samples": 0,
            "aiusmi_error": "csv empty",
        }

    parsed = list(csv.DictReader(rows))
    traffic_rows = []
    for row in parsed:
        values = {}
        for key in AIUSMI_RATE_KEYS:
            if key in row and row[key] not in ("", None):
                values[key] = float(row[key])
        if any(
            values.get(key, 0.0)
            for key in (
                "rdmem GiB/s",
                "wrmem GiB/s",
                "n_rdmem Mreq/s",
                "n_wrmem Mreq/s",
            )
        ):
            traffic_rows.append((values, row))

    summary: dict[str, Any] = {
        "aiusmi_csv_path": str(path),
        "aiusmi_sample_rows": len(parsed),
        "aiusmi_traffic_samples": len(traffic_rows),
        "aiusmi_first_traffic_rows": [
            {
                "time": row.get("Time (HH:MM:SS)", ""),
                "values": {key: value for key, value in values.items() if value},
            }
            for values, row in traffic_rows[:8]
        ],
    }
    for key in AIUSMI_RATE_KEYS:
        values = [values[key] for values, _ in traffic_rows if key in values]
        if values:
            stem = key.replace(" ", "_").replace("/", "_per_").replace("%", "pct")
            summary[f"aiusmi_peak_{stem}"] = max(values)
            summary[f"aiusmi_avg_nonzero_{stem}"] = sum(values) / len(values)
    return summary


def _time_loop(compiled: Any, dev_args: tuple[Any, ...], iters: int) -> dict[str, Any]:
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        compiled(*dev_args)
        probe._sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p10_ms": samples[max(0, int(0.10 * (len(samples) - 1)))],
        "p90_ms": samples[min(len(samples) - 1, int(0.90 * (len(samples) - 1)))],
    }


def _run_mode(args: argparse.Namespace, case: hierarchy.HierarchyCase, mode: str, size: int) -> dict[str, Any]:
    dtype = probe._dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir) / mode / f"{case.name}_{size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = output_dir / "ring_telemetry.jsonl"
    aiusmi_csv = output_dir / "aiusmi.csv"
    primer_csv = output_dir / "aiusmi_compile_warmup.csv"
    metric_file = Path(args.metric_file)
    metric_copy = output_dir / "metrics.0000:aa:00.0"
    export_dir = output_dir / "dtcompiler-export"
    export_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING": "1" if mode == "stage3b" else "0",
        "SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION": "1" if mode == "stage3b" else "0",
        "SPYRE_RESTICKIFY_LOCALITY_ASSERT": "1" if mode == "stage3b" and args.locality_assert else "0",
        "DTCOMPILER_EXPORT_DIR": str(export_dir),
    }
    enable_alignment = mode == "stage3b"
    with hierarchy._temporary_env(env), hierarchy._temporary_spyre_alignment_config(
        core_mapping=enable_alignment,
        work_distribution=enable_alignment,
        locality_assert=mode == "stage3b" and args.locality_assert,
    ):
        _ensure_aiusmi_env(metric_file, export_dir)
        probe._reset_compile_caches()

        args_tuple, shape_label = case.builder(size, dtype)
        dev_args = tuple(arg.to(args.device) if hasattr(arg, "to") else arg for arg in args_tuple)

        primer = None
        primer_stderr = ""
        if args.prime_aiusmi:
            primer = _start_aiusmi(args.aiusmi, primer_csv, args.sample_interval)
            time.sleep(args.prime_delay)

        with _restickify_capture(telemetry_path) as insert_restickify:
            compiled = torch.compile(case.fn, backend=args.backend, dynamic=False)
            start = time.perf_counter()
            compiled(*dev_args)
            probe._sync()
            compile_run_ms = (time.perf_counter() - start) * 1000.0

            plan = dict(insert_restickify.restickify_plan)
            entries, total_elements, total_bytes = probe._summarize_plan(plan)
            ring_summary = probe._read_ring_telemetry(telemetry_path)

        for _ in range(args.warmup):
            compiled(*dev_args)
        probe._sync()

        if primer is not None:
            primer_stderr = _stop_aiusmi(primer).strip()

        aiusmi = _start_aiusmi(args.aiusmi, aiusmi_csv, args.sample_interval)
        time.sleep(args.start_delay)
        timing = _time_loop(compiled, dev_args, args.iters)
        time.sleep(args.stop_delay)
        aiusmi_stderr = _stop_aiusmi(aiusmi)

    if metric_file.exists():
        shutil.copyfile(metric_file, metric_copy)

    aiusmi_summary = _read_aiusmi_csv(aiusmi_csv)
    row = {
        "status": "ok",
        "mode": mode,
        "case": case.name,
        "scenario": case.family,
        "source_hint": case.boundary_model,
        "description": case.description,
        "shape": shape_label,
        "size": size,
        "dtype": str(dtype).replace("torch.", ""),
        "compile_run_ms": compile_run_ms,
        "warmup": args.warmup,
        "iters": args.iters,
        "sample_interval": args.sample_interval,
        "restickify_count": len(entries),
        "total_elements": total_elements,
        "total_bytes": total_bytes,
        "entries": entries,
        "metric_file_path": str(metric_file),
        "metric_file_copied_path": str(metric_copy) if metric_copy.exists() else "",
        "metric_file_size": metric_copy.stat().st_size if metric_copy.exists() else 0,
        "aiusmi_stderr": aiusmi_stderr.strip(),
        "aiusmi_primer_csv_path": str(primer_csv) if args.prime_aiusmi else "",
        "aiusmi_primer_stderr": primer_stderr,
        **ring_summary,
        **timing,
        **aiusmi_summary,
    }
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="adds_then_matmul_x", help="Hierarchy sweep case to run.")
    parser.add_argument("--size", type=int, action="append", default=[], help="Problem size. May be repeated.")
    parser.add_argument("--mode", action="append", choices=("baseline", "stage3b"), default=[], help="Mode to run. Defaults to both.")
    parser.add_argument("--dtype", default="float16", help="Input dtype.")
    parser.add_argument("--device", default="spyre", help="Execution device.")
    parser.add_argument("--backend", default="inductor", help="torch.compile backend.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations before aiu-smi starts.")
    parser.add_argument("--iters", type=int, default=1000, help="Timed iterations while aiu-smi is running.")
    parser.add_argument("--sample-interval", type=float, default=0.1, help="aiu-smi sample interval in seconds.")
    parser.add_argument("--start-delay", type=float, default=0.5, help="Delay after starting aiu-smi before timed loop.")
    parser.add_argument("--stop-delay", type=float, default=0.5, help="Delay after timed loop before stopping aiu-smi.")
    parser.add_argument("--prime-delay", type=float, default=0.2, help="Delay after starting primer aiu-smi before compile.")
    parser.add_argument(
        "--no-prime-aiusmi",
        action="store_false",
        dest="prime_aiusmi",
        help="Do not run a throwaway aiu-smi during compile/warmup.",
    )
    parser.add_argument("--aiusmi", default="aiu-smi", help="aiu-smi executable.")
    parser.add_argument("--metric-file", default="/tmp/metrics.0000:aa:00.0", help="Runtime metric file to archive.")
    parser.add_argument("--output-dir", default="/tmp/restickify-aiusmi-marker", help="Output directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--locality-assert", action="store_true", help="Enable locality assert for stage3b.")
    parser.add_argument("--fail-on-error", action="store_true", help="Return nonzero on errors.")
    return parser.parse_args()


def main() -> int:
    global torch

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    early_export_dir = Path(args.output_dir) / "dtcompiler-export"
    early_export_dir.mkdir(parents=True, exist_ok=True)
    _ensure_aiusmi_env(Path(args.metric_file), early_export_dir)
    Path(args.metric_file).unlink(missing_ok=True)

    import_primer = None
    import_primer_stderr = ""
    if args.prime_aiusmi:
        import_primer = _start_aiusmi(
            args.aiusmi, output_dir / "aiusmi_import_primer.csv", args.sample_interval
        )
        time.sleep(args.prime_delay)

    rows: list[dict[str, Any]] = []
    try:
        import torch as torch_module

        torch = torch_module
        hierarchy.torch = torch_module
        probe.torch = torch_module
        torch.manual_seed(args.seed)

        case = _find_case(args.case)
        sizes = args.size or [2048]
        modes = args.mode or ["baseline", "stage3b"]

        for size in sizes:
            for mode in modes:
                try:
                    row = _run_mode(args, case, mode, size)
                except Exception as exc:
                    row = {
                        "status": "error",
                        "mode": mode,
                        "case": case.name,
                        "size": size,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                row["aiusmi_import_primer_csv_path"] = (
                    str(output_dir / "aiusmi_import_primer.csv") if import_primer is not None else ""
                )
                rows.append(row)
                print(
                    f"{row['status']:5} mode={mode:<8} size={size:<5} case={case.name:<28} "
                    f"median_ms={row.get('median_ms', '')} "
                    f"byte_hops={row.get('ring_total_byte_hops', 0)} "
                    f"rd_peak={row.get('aiusmi_peak_rdmem_GiB_per_s', 0)} "
                    f"wr_peak={row.get('aiusmi_peak_wrmem_GiB_per_s', 0)} "
                    f"traffic_samples={row.get('aiusmi_traffic_samples', 0)}"
                )
    finally:
        if import_primer is not None:
            import_primer_stderr = _stop_aiusmi(import_primer).strip()

    for row in rows:
        row["aiusmi_import_primer_stderr"] = import_primer_stderr

    jsonl_path = output_dir / "aiusmi_marker_rows.jsonl"
    csv_path = output_dir / "aiusmi_marker_rows.csv"
    jsonl_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    fieldnames = sorted(
        {key for row in rows for key in row if key != "entries" and key != "ring_entries"}
    )
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    errors = [row for row in rows if row["status"] != "ok"]
    print(f"Completed {len(rows)} rows with {len(errors)} errors")
    return 1 if errors and args.fail_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())

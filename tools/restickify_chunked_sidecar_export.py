#!/usr/bin/env python3
# Copyright 2025 The Torch-Spyre Authors.
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

"""Export chunked PT-LX restickify sidecars with retries and a manifest.

The native PT-LX restickify prototype emits one standalone SDSC JSON per
schedule-aware chunk:

    restickify_lx_neighbor_streaming_bridge_edge_<edge>_chunk<N>.json

The DeeRT export path is currently flaky for these diagnostic data-op SDSCs:
some attempts produce usable artifacts and then crash in post-export metadata
cleanup. This tool keeps that instability contained by exporting each chunk in
an isolated output directory, retrying independently, validating the generated
``senprog.txt`` token mix, and writing a manifest that later packaging/runtime
experiments can consume.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


_TOKENS = ("HBM", "L3LU", "L3SU", "LXLU", "LXSU", "SFP", "PT")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _default_exporter(explicit: str | None) -> str:
    candidates = [
        explicit,
        os.environ.get("SPYRE_RESTICKIFY_DEEPRT_DATAOP_EXPORTER"),
        "/tmp/stage65-deeprt-dataop-probe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError(
        "could not find DeeRT data-op exporter; pass --exporter or set "
        "SPYRE_RESTICKIFY_DEEPRT_DATAOP_EXPORTER"
    )


def _chunk_index(path: Path) -> int:
    match = re.search(r"_chunk(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"could not parse chunk index from {path.name!r}")
    return int(match.group(1))


def _edge_index(path: Path) -> int | None:
    match = re.search(r"_edge_(\d+)(?:_chunk\d+)?\.json$", path.name)
    return int(match.group(1)) if match else None


def _find_sidecars(
    sidecar_dir: Path,
    *,
    requested_chunks: set[int] | None,
) -> list[Path]:
    candidates = sorted(
        sidecar_dir.glob("restickify_lx_neighbor_streaming_bridge_edge_*_chunk*.json"),
        key=lambda path: (_edge_index(path) or -1, _chunk_index(path), path.name),
    )
    if requested_chunks is not None:
        candidates = [
            path for path in candidates if _chunk_index(path) in requested_chunks
        ]
    return candidates


def _first_export_file(export_dir: Path, name: str) -> Path | None:
    candidates = sorted((export_dir / "execute").glob(f"*/{name}"))
    return candidates[0] if candidates else None


def _token_counts(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {token: 0 for token in _TOKENS}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {token: text.count(token) for token in _TOKENS}


def _sidecar_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    root_name, root = next(iter(payload.items()))
    datadscs = root.get("datadscs_", []) or []
    names = [str(next(iter(dataop))) for dataop in datadscs]
    meta = root.get("streamingPTLXFull_", {}) or {}
    return {
        "root_name": root_name,
        "edge_index": _edge_index(path),
        "chunk_index": _chunk_index(path),
        "dataop_count": len(datadscs),
        "gather_dataops": sum("STCDPOpLx_gather" in name for name in names),
        "restickify_dataops": sum("ReStickifyOpWithPTLx" in name for name in names),
        "scatter_dataops": sum("STCDPOpLx_scatter" in name for name in names),
        "meta": meta,
    }


def _export_once(
    *,
    exporter: str,
    sidecar: Path,
    attempt_dir: Path,
    backend: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    timed_out = False
    try:
        proc = subprocess.run(
            [exporter, str(sidecar), str(attempt_dir), backend],
            cwd=attempt_dir,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr += f"\nexport timed out after {timeout_seconds} seconds\n"
    stdout_path = attempt_dir.with_suffix(".stdout.txt")
    stderr_path = attempt_dir.with_suffix(".stderr.txt")
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    init_path = _first_export_file(attempt_dir, "init.txt")
    senprog_path = _first_export_file(attempt_dir, "senprog.txt")
    smc_path = _first_export_file(attempt_dir, "smc.txt")
    sdsc_path = _first_export_file(attempt_dir, "sdsc.json")
    token_counts = _token_counts(senprog_path)
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "output_dir": str(attempt_dir),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "init": str(init_path) if init_path is not None else "",
        "senprog": str(senprog_path) if senprog_path is not None else "",
        "smc": str(smc_path) if smc_path is not None else "",
        "sdsc": str(sdsc_path) if sdsc_path is not None else "",
        "tokens": token_counts,
        "has_init": init_path is not None,
        "has_senprog": senprog_path is not None,
    }


def _attempt_is_success(
    attempt: dict[str, Any],
    *,
    require_clean_returncode: bool,
    require_no_hbm: bool,
) -> bool:
    if require_clean_returncode and int(attempt["returncode"]) != 0:
        return False
    if not attempt["has_init"] or not attempt["has_senprog"]:
        return False
    if require_no_hbm and int(attempt["tokens"].get("HBM", 0)) != 0:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exporter")
    parser.add_argument("--backend", default="sentient")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="per-attempt DeeRT exporter timeout",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        action="append",
        default=[],
        help="export only the selected chunk index; may be repeated",
    )
    parser.add_argument(
        "--require-clean-returncode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="treat post-export crashes as failures even if artifacts exist",
    )
    parser.add_argument(
        "--require-no-hbm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fail a chunk if its senprog.txt contains any HBM token",
    )
    parser.add_argument(
        "--fail-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--manifest",
        default="manifest.json",
        help="manifest filename relative to --output-dir",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sidecar_dir = args.sidecar_dir.resolve()
    output_dir = args.output_dir.resolve()
    exporter = _default_exporter(args.exporter)
    requested_chunks = set(args.chunk_index) if args.chunk_index else None
    sidecars = _find_sidecars(sidecar_dir, requested_chunks=requested_chunks)
    if not sidecars:
        print(f"no chunked sidecars found under {sidecar_dir}", file=sys.stderr)
        return 2

    chunks: list[dict[str, Any]] = []
    for ordinal, sidecar in enumerate(sidecars):
        summary = _sidecar_summary(sidecar)
        chunk_index = summary["chunk_index"]
        chunk_record: dict[str, Any] = {
            **summary,
            "ordinal": ordinal,
            "sidecar": str(sidecar),
            "attempts": [],
            "success": False,
            "selected_attempt": None,
            "failure_reason": "",
        }
        for attempt_index in range(1, int(args.retries) + 1):
            attempt_dir = (
                output_dir
                / "exports"
                / f"chunk{int(chunk_index):04d}_try{attempt_index}"
            )
            attempt = _export_once(
                exporter=exporter,
                sidecar=sidecar,
                attempt_dir=attempt_dir,
                backend=args.backend,
                timeout_seconds=args.timeout_seconds,
            )
            attempt["attempt_index"] = attempt_index
            chunk_record["attempts"].append(attempt)
            if _attempt_is_success(
                attempt,
                require_clean_returncode=args.require_clean_returncode,
                require_no_hbm=args.require_no_hbm,
            ):
                chunk_record["success"] = True
                chunk_record["selected_attempt"] = attempt_index
                chunk_record["selected"] = attempt
                break

        if not chunk_record["success"]:
            last = chunk_record["attempts"][-1] if chunk_record["attempts"] else {}
            if not last.get("has_init") or not last.get("has_senprog"):
                chunk_record["failure_reason"] = "missing-export-artifacts"
            elif args.require_clean_returncode and int(last.get("returncode", -1)) != 0:
                chunk_record["failure_reason"] = "nonzero-returncode"
            elif args.require_no_hbm and int(last.get("tokens", {}).get("HBM", 0)) != 0:
                chunk_record["failure_reason"] = "hbm-token-present"
            else:
                chunk_record["failure_reason"] = "unknown"
        chunks.append(chunk_record)

        selected = chunk_record.get("selected") or (
            chunk_record["attempts"][-1] if chunk_record["attempts"] else {}
        )
        print(
            f"chunk {chunk_index}: "
            f"success={chunk_record['success']} "
            f"attempt={chunk_record.get('selected_attempt') or 'none'} "
            f"rc={selected.get('returncode', 'NA')} "
            f"HBM={selected.get('tokens', {}).get('HBM', 'NA')} "
            f"LXLU={selected.get('tokens', {}).get('LXLU', 'NA')} "
            f"LXSU={selected.get('tokens', {}).get('LXSU', 'NA')}",
            flush=True,
        )

    successful = [chunk for chunk in chunks if chunk["success"]]
    failed = [chunk for chunk in chunks if not chunk["success"]]
    selected_tokens = {
        token: sum(
            int((chunk.get("selected") or {}).get("tokens", {}).get(token, 0))
            for chunk in successful
        )
        for token in _TOKENS
    }
    manifest = {
        "sidecar_dir": str(sidecar_dir),
        "output_dir": str(output_dir),
        "exporter": exporter,
        "backend": args.backend,
        "retries": args.retries,
        "timeout_seconds": args.timeout_seconds,
        "require_clean_returncode": args.require_clean_returncode,
        "require_no_hbm": args.require_no_hbm,
        "chunk_count": len(chunks),
        "successful_chunks": len(successful),
        "failed_chunks": len(failed),
        "selected_token_totals": selected_tokens,
        "chunks": chunks,
    }
    manifest_path = output_dir / args.manifest
    _write_json(manifest_path, manifest)
    print(
        "summary "
        f"ok={len(successful)} fail={len(failed)} "
        f"manifest={manifest_path}"
    )
    if failed and args.fail_on_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

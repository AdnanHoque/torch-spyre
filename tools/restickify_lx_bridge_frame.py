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

"""Generate a compile-only LX-to-LX restickify bridge frame.

This tool packages the already-proven schema-v4 materialization path into a
reusable frame artifact without launching hardware:

    producer LX -> ReStickifyOpLx/STCDPOpLx data-op -> consumer LX

The input is a Torch-Spyre generated code directory containing the normal
producer -> ReStickifyOpHBM -> consumer SDSCs plus
``restickify_lx_neighbor_edges.json``.  The output is a standalone bridge frame
directory with the patched data-op SDSC, DeeRT export files, ``init.txt``,
``init_binary.bin``, optional sentinel-cleared frame bytes, ``senprog.txt``, and
``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


_TOKENS = ("HBM", "L3LU", "L3SU", "LXLU", "LXSU", "SFP", "PT")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    stdout: Path,
    stderr: Path,
    env: dict[str, str],
) -> int:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stdout.write_text(proc.stdout, encoding="utf-8")
    stderr.write_text(proc.stderr, encoding="utf-8")
    return proc.returncode


def _read_hex_init(path: Path) -> bytes:
    out = bytearray()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if len(line) != 256:
            raise ValueError(f"{path} has non-256-char init line")
        out.extend(bytes.fromhex(line)[::-1])
    return bytes(out)


def _write_hex_init(path: Path, data: bytes) -> None:
    if len(data) % 128 != 0:
        raise ValueError(f"program frame size must be 128-byte aligned: {len(data)}")
    lines = []
    for offset in range(0, len(data), 128):
        lines.append(data[offset : offset + 128][::-1].hex())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clear_sentinel(frame: bytes) -> bytes:
    if len(frame) < 4:
        raise ValueError("program frame is too small to contain a header word")
    word = struct.unpack_from("<I", frame, 0)[0]
    out = bytearray(frame)
    struct.pack_into("<I", out, 0, word & 0xFFBFFFFF)
    return bytes(out)


def _term_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {token: text.count(token) for token in _TOKENS}


def _first_export_file(export_dir: Path, name: str) -> Path | None:
    candidates = sorted((export_dir / "execute").glob(f"*/{name}"))
    return candidates[0] if candidates else None


def _default_address_probe() -> Path:
    return Path(__file__).with_name("restickify_address_preserving_dataop_probe.py")


def _default_exporter(explicit: str | None) -> str:
    candidates = [
        explicit,
        os.environ.get("SPYRE_RESTICKIFY_DEEPRT_DATAOP_EXPORTER"),
        "/tmp/stage65-deeprt-dataop-probe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "could not find DeeRT data-op exporter; pass --exporter or set "
        "SPYRE_RESTICKIFY_DEEPRT_DATAOP_EXPORTER"
    )


def _generate_address_preserving_sdsc(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    work_dir = output_dir / "address_preserving_dataop"
    probe = Path(args.address_probe).resolve()
    cmd = [
        args.python,
        str(probe),
        "--code-dir",
        str(Path(args.code_dir).resolve()),
        "--output-dir",
        str(work_dir),
        "--mode",
        args.mode,
        "--no-run-dataop-standalone",
    ]
    if args.descriptor:
        cmd.extend(["--descriptor", str(Path(args.descriptor).resolve())])
    if args.size is not None:
        cmd.extend(["--size", str(args.size)])
    if args.num_cores is not None:
        cmd.extend(["--num-cores", str(args.num_cores)])

    env = {
        **os.environ,
        "SPYRE_RESTICKIFY_LX_DATAOP": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }
    rc = _run(
        cmd,
        cwd=Path.cwd(),
        stdout=output_dir / "address_preserving_dataop.stdout.txt",
        stderr=output_dir / "address_preserving_dataop.stderr.txt",
        env=env,
    )
    summary_path = work_dir / "summary.json"
    summary: dict[str, Any] = {
        "command": cmd,
        "returncode": rc,
        "summary": str(summary_path) if summary_path.exists() else "",
    }
    if rc != 0:
        summary["status"] = "failed"
        return summary
    if not summary_path.exists():
        summary["status"] = "missing-summary"
        return summary
    payload = _load_json(summary_path)
    summary.update(
        {
            "status": "ok",
            "payload": payload,
            "patched_sdsc": payload.get("patched", {}).get("path", ""),
            "contract_source": (
                payload.get("address_summary", {})
                .get("endpoint_contract", {})
                .get("source", "")
            ),
            "materialization_kind": (
                payload.get("address_summary", {})
                .get("endpoint_contract", {})
                .get("materialization_kind", "")
            ),
            "intended_sequence": (
                payload.get("address_summary", {})
                .get("endpoint_contract", {})
                .get("intended_deeptools_sequence", [])
            ),
            "producer_pieces_patched": payload.get("patched", {}).get(
                "producer_pieces_patched", 0
            ),
            "consumer_pieces_patched": payload.get("patched", {}).get(
                "consumer_pieces_patched", 0
            ),
        }
    )
    return summary


def _export_frame(args: argparse.Namespace, patched_sdsc: Path, output_dir: Path) -> dict[str, Any]:
    exporter = _default_exporter(args.exporter)
    export_dir = output_dir / "deeprt_export"
    rc = _run(
        [exporter, str(patched_sdsc), str(export_dir), args.target],
        cwd=output_dir,
        stdout=output_dir / "deeprt_export.stdout.txt",
        stderr=output_dir / "deeprt_export.stderr.txt",
        env=os.environ.copy(),
    )
    init = _first_export_file(export_dir, "init.txt")
    senprog = _first_export_file(export_dir, "senprog.txt")
    return {
        "command": [exporter, str(patched_sdsc), str(export_dir), args.target],
        "returncode": rc,
        "export_dir": str(export_dir),
        "init": str(init) if init else "",
        "senprog": str(senprog) if senprog else "",
    }


def _materialize_frame_files(export_summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    init = Path(export_summary.get("init") or "")
    if not init.exists():
        return {"status": "missing-init"}

    frame = _read_hex_init(init)
    copied_init = output_dir / "init.txt"
    shutil.copy2(init, copied_init)
    init_binary = output_dir / "init_binary.bin"
    init_binary.write_bytes(frame)

    cleared = _clear_sentinel(frame)
    cleared_binary = output_dir / "init_binary_sentinel_cleared.bin"
    cleared_binary.write_bytes(cleared)
    cleared_init = output_dir / "init_sentinel_cleared.txt"
    _write_hex_init(cleared_init, cleared)

    senprog = Path(export_summary.get("senprog") or "")
    copied_senprog = output_dir / "senprog.txt"
    token_counts: dict[str, int] = {}
    if senprog.exists():
        shutil.copy2(senprog, copied_senprog)
        token_counts = _term_counts(copied_senprog)

    return {
        "status": "ok",
        "init_txt": str(copied_init),
        "init_binary": str(init_binary),
        "init_binary_sentinel_cleared": str(cleared_binary),
        "init_sentinel_cleared_txt": str(cleared_init),
        "senprog_txt": str(copied_senprog) if copied_senprog.exists() else "",
        "frame_bytes": len(frame),
        "frame_flits_128b": len(frame) // 128,
        "frame_128b_aligned": len(frame) % 128 == 0,
        "first_header_word": struct.unpack_from("<I", frame, 0)[0] if len(frame) >= 4 else None,
        "sentinel_cleared_header_word": (
            struct.unpack_from("<I", cleared, 0)[0] if len(cleared) >= 4 else None
        ),
        "senprog_token_counts": token_counts,
        "hbm_free": token_counts.get("HBM", 0) == 0 if token_counts else None,
        "has_l3_ring_facing_tokens": (
            token_counts.get("L3LU", 0) + token_counts.get("L3SU", 0) > 0
            if token_counts
            else None
        ),
        "has_lx_endpoint_tokens": (
            token_counts.get("LXLU", 0) + token_counts.get("LXSU", 0) > 0
            if token_counts
            else None
        ),
    }


def _copy_patched_sdsc(address_summary: dict[str, Any], output_dir: Path) -> str:
    patched = Path(address_summary.get("patched_sdsc") or "")
    if not patched.exists():
        return ""
    target = output_dir / "sdsc_lx_bridge_dataop.json"
    shutil.copy2(patched, target)
    return str(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--descriptor", default="")
    parser.add_argument("--output-dir", default="/tmp/restickify-lx-bridge-frame")
    parser.add_argument("--mode", choices=("baseline", "stage3b"), default="stage3b")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--num-cores", type=int, default=None)
    parser.add_argument("--target", default="sentient")
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--address-probe", default=str(_default_address_probe()))
    parser.add_argument("--exporter", default=None)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-materialization-contract",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fail-on-hbm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail-on-missing-senprog", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and args.clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    address_summary = _generate_address_preserving_sdsc(args, output_dir)
    summary: dict[str, Any] = {
        "status": "failed",
        "code_dir": str(Path(args.code_dir).resolve()),
        "output_dir": str(output_dir),
        "address_preserving_dataop": {
            key: value for key, value in address_summary.items() if key != "payload"
        },
    }

    if address_summary.get("status") != "ok":
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return int(address_summary.get("returncode") or 1)

    contract_source = address_summary.get("contract_source", "")
    if (
        args.require_materialization_contract
        and contract_source != "schema-v4-lx-materialization-contract"
    ):
        summary["failure_reason"] = (
            "expected schema-v4-lx-materialization-contract, "
            f"got {contract_source or '<missing>'}"
        )
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    patched_sdsc = Path(address_summary["patched_sdsc"])
    copied_sdsc = _copy_patched_sdsc(address_summary, output_dir)
    export_summary = _export_frame(args, patched_sdsc, output_dir)
    summary["deeprt_export"] = export_summary
    if export_summary.get("returncode") != 0:
        summary["failure_reason"] = "deeprt export failed"
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return int(export_summary["returncode"] or 1)

    frame_summary = _materialize_frame_files(export_summary, output_dir)
    summary["frame"] = frame_summary
    summary["sdsc_lx_bridge_dataop"] = copied_sdsc

    if frame_summary.get("status") != "ok":
        summary["failure_reason"] = frame_summary["status"]
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 3
    if not frame_summary.get("frame_128b_aligned"):
        summary["failure_reason"] = "frame is not 128-byte aligned"
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 4
    if args.fail_on_missing_senprog and not frame_summary.get("senprog_txt"):
        summary["failure_reason"] = "missing senprog.txt"
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 5
    if args.fail_on_hbm and frame_summary.get("hbm_free") is False:
        summary["failure_reason"] = "senprog contains HBM tokens"
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 6

    summary["status"] = "ok"
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

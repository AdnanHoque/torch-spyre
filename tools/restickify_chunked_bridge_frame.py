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

"""Materialize a chunked PT-LX sidecar manifest as one bridge frame.

``restickify_chunked_sidecar_export.py`` proves that every row chunk can export
to a clean no-HBM DeeRT artifact. This tool consumes that manifest and produces
a single concatenated program-frame artifact that the existing same-artifact
splice probe can use as a replacement for the stock ``ReStickifyOpHBM`` frame.

This is still compile/package-only. It does not prove that a concatenated chunk
sequence is a valid runtime replacement; that is the next hardware validation
step.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Any


_TOKENS = ("HBM", "L3LU", "L3SU", "LXLU", "LXSU", "SFP", "PT")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _header_word(frame: bytes) -> int | None:
    if len(frame) < 4:
        return None
    return struct.unpack_from("<I", frame, 0)[0]


def _token_counts(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {token: 0 for token in _TOKENS}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {token: text.count(token) for token in _TOKENS}


def _selected_chunks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = sorted(
        manifest.get("chunks", []) or [],
        key=lambda chunk: int(chunk.get("chunk_index", -1)),
    )
    failed = [chunk for chunk in chunks if not chunk.get("success")]
    if failed:
        indices = [chunk.get("chunk_index") for chunk in failed]
        raise ValueError(f"manifest contains failed chunks: {indices}")
    return chunks


def _copy_optional(path_text: str, target: Path) -> str:
    if not path_text:
        return ""
    source = Path(path_text)
    if not source.exists():
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-no-hbm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    manifest = _read_json(manifest_path)
    chunks = _selected_chunks(manifest)

    raw_frames: list[bytes] = []
    cleared_frames: list[bytes] = []
    chunk_records: list[dict[str, Any]] = []
    token_totals = {token: 0 for token in _TOKENS}

    for ordinal, chunk in enumerate(chunks):
        selected = chunk.get("selected") or {}
        init = Path(selected.get("init") or "")
        if not init.exists():
            raise FileNotFoundError(
                f"chunk {chunk.get('chunk_index')} selected init is missing: {init}"
            )
        frame = _read_hex_init(init)
        if len(frame) % 128 != 0:
            raise ValueError(
                f"chunk {chunk.get('chunk_index')} frame is not aligned: {len(frame)}"
            )
        tokens = selected.get("tokens") or _token_counts(
            Path(selected.get("senprog") or "")
        )
        if args.require_no_hbm and int(tokens.get("HBM", 0)) != 0:
            raise ValueError(
                f"chunk {chunk.get('chunk_index')} has HBM token count "
                f"{tokens.get('HBM')}"
            )
        for token in _TOKENS:
            token_totals[token] += int(tokens.get(token, 0))

        raw_frame = frame if ordinal == 0 else _clear_sentinel(frame)
        cleared_frame = _clear_sentinel(frame)
        raw_frames.append(raw_frame)
        cleared_frames.append(cleared_frame)

        copied_init = _copy_optional(
            str(init),
            output_dir / "chunks" / f"chunk{int(chunk['chunk_index']):04d}" / "init.txt",
        )
        copied_senprog = _copy_optional(
            selected.get("senprog", ""),
            output_dir
            / "chunks"
            / f"chunk{int(chunk['chunk_index']):04d}"
            / "senprog.txt",
        )
        chunk_records.append(
            {
                "chunk_index": chunk.get("chunk_index"),
                "source_manifest_attempt": chunk.get("selected_attempt"),
                "init": selected.get("init", ""),
                "senprog": selected.get("senprog", ""),
                "copied_init": copied_init,
                "copied_senprog": copied_senprog,
                "frame_bytes": len(frame),
                "frame_flits_128b": len(frame) // 128,
                "header_word": _header_word(frame),
                "cleared_header_word": _header_word(cleared_frame),
                "tokens": tokens,
            }
        )

    raw = b"".join(raw_frames)
    cleared = b"".join(cleared_frames)
    _write_hex_init(output_dir / "init.txt", raw)
    _write_hex_init(output_dir / "init_sentinel_cleared.txt", cleared)
    (output_dir / "init_binary.bin").write_bytes(raw)
    (output_dir / "init_binary_sentinel_cleared.bin").write_bytes(cleared)

    combined_senprog = output_dir / "senprog.txt"
    with combined_senprog.open("w", encoding="utf-8") as out:
        for record in chunk_records:
            path = Path(record["copied_senprog"])
            out.write(f"# chunk {record['chunk_index']}\n")
            if path.exists():
                out.write(path.read_text(encoding="utf-8", errors="replace"))
                out.write("\n")

    summary = {
        "status": "ok",
        "manifest": str(manifest_path),
        "chunk_count": len(chunk_records),
        "chunks": chunk_records,
        "frame": {
            "init_txt": str(output_dir / "init.txt"),
            "init_binary": str(output_dir / "init_binary.bin"),
            "init_sentinel_cleared_txt": str(output_dir / "init_sentinel_cleared.txt"),
            "init_binary_sentinel_cleared": str(
                output_dir / "init_binary_sentinel_cleared.bin"
            ),
            "senprog_txt": str(combined_senprog),
            "frame_bytes": len(raw),
            "frame_flits_128b": len(raw) // 128,
            "frame_128b_aligned": len(raw) % 128 == 0,
            "sentinel_cleared_frame_bytes": len(cleared),
            "senprog_token_counts": token_totals,
            "hbm_free": token_totals.get("HBM", 0) == 0,
            "has_lx_endpoint_tokens": (
                token_totals.get("LXLU", 0) + token_totals.get("LXSU", 0) > 0
            ),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

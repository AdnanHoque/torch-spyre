#!/usr/bin/env python3
"""Verify immutable closure bytes and prove the loader resolves only inside it."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


ARROW = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("closure", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    closure = args.closure.resolve()
    manifest_path = closure / "closure_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    observed_files: dict[str, dict[str, object]] = {}
    for relative, expected in manifest["files"].items():
        path = closure / relative
        if not path.is_file():
            errors.append(f"missing file: {relative}")
            continue
        observed = {"sha256": sha256(path), "size": path.stat().st_size}
        observed_files[relative] = observed
        if observed != expected:
            errors.append(f"file drift: {relative} expected={expected} observed={observed}")

    loader = closure / "lib" / "ld-linux-x86-64.so.2"
    binary = closure / "dxp_standalone"
    listed = subprocess.run(
        [
            str(loader),
            "--library-path",
            str(closure / "lib"),
            "--list",
            str(binary),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if listed.returncode != 0:
        errors.append(f"loader --list failed rc={listed.returncode}: {listed.stderr}")
    escaped_root = str(closure / "lib") + os.sep
    for line in listed.stdout.splitlines():
        arrow = ARROW.match(line)
        if not arrow:
            continue
        name, raw_path = arrow.groups()
        if name == "linux-vdso.so.1":
            continue
        if not os.path.realpath(raw_path).startswith(escaped_root):
            errors.append(f"resolution escaped closure: {name} -> {raw_path}")

    report = {
        "pass": not errors,
        "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "closure": str(closure),
        "manifest_sha256": sha256(manifest_path),
        "files": observed_files,
        "loader_returncode": listed.returncode,
        "loader_list": listed.stdout,
        "errors": errors,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()

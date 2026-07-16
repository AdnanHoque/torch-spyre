#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: analyze_replay.py RUN_ROOT VARIANT...")

    run_root = Path(sys.argv[1])
    for variant in sys.argv[2:]:
        path = run_root / variant
        print(f"\n{variant}")
        print("rc:", (path / "return_code.txt").read_text().strip())
        print("stdout:", (path / "stdout.log").read_text().strip())
        stderr = (path / "stderr.log").read_text().splitlines()
        print("stderr_tail:", "\n".join(stderr[-3:]))

        relayout_path = path / "relayout_debug_out0.json"
        print("sha256:", hashlib.sha256(relayout_path.read_bytes()).hexdigest())
        document = json.loads(relayout_path.read_text())
        superdsc = next(iter(document.values()))
        data_op_entry = superdsc["datadscs_"][0]
        data_op = next(iter(data_op_entry.values()))
        print("data_op:", data_op["op"])
        for index, labeled_ds in enumerate(data_op["labeledDs_"]):
            pieces = labeled_ds.get("PieceInfo", [])
            first_piece = pieces[0] if pieces else None
            addresses = []
            for piece in pieces:
                placement = piece["PlacementInfo"][0]
                addresses.extend(placement["startAddr"]["data_"].values())
            print(
                "labeled_ds:",
                index,
                labeled_ds.get("ldsName_"),
                "piece_count=",
                len(pieces),
                "addresses=",
                sorted(set(address for group in addresses for address in group)),
                "first_piece=",
                json.dumps(first_piece, sort_keys=True),
            )


if __name__ == "__main__":
    main()

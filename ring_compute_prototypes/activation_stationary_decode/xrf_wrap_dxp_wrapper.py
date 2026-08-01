#!/usr/bin/env python3
"""DXP wrapper for the fixed-grid Design A XRF-wrap ablation.

The exact M512/K12800/N4096 activation-stationary program loads 16 XRF
entries per PT unit.  Its generic program resets the XRF read pointer at the
end of a 64-iteration inner loop.  This experiment sets the architectural XRF
wrap size to 16 once and moves that inner-loop end bit to the final FMA.  The
existing reset remains in its original packet slot and executes once after the
loop, preserving InitPacket size and every runtime patch location.

This is experimental tooling, not a production compiler transform.  It fails
closed unless the descriptor and exact instruction patterns match.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any


TARGET_BMM = {"mb_": 4096, "out_": 512, "in_": 12800}
TARGET_SLICES = {"mb": 8, "out": 4, "in": 1}
TARGET_CORES = 32
EXPECTED_PT_PROGRAMS = 8

XRF_SET_WRITE_ZERO = 0x00100028
XRF_SET_READ_ZERO = 0x00001028
OUTER_LOOP_200 = 0x00003221
XRF_SET_SIZE_16_AND_WRITE_ZERO = 0x00103428
INNER_FINAL_FMA = 0x12FE8383
INNER_RESET_READ_ZERO_BE = 0x80001028
INNER_RESET_READ_ZERO = 0x00001028


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(data: bytes | bytearray) -> str:
    return hashlib.sha256(data).hexdigest()


def find_all(data: bytes | bytearray, pattern: bytes) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        offset = data.find(pattern, start)
        if offset < 0:
            return offsets
        offsets.append(offset)
        start = offset + 1


def pack_word(word: int) -> bytes:
    return struct.pack("<I", word)


def descriptor(bundle: Path) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for path in sorted(bundle.glob("sdsc_*.json")):
        document = json.loads(path.read_text())
        if len(document) != 1:
            continue
        name, sdsc = next(iter(document.items()))
        if not name.endswith("batchmatmul"):
            continue
        dscs = sdsc.get("dscs_", [])
        if len(dscs) != 1 or "batchmatmul" not in dscs[0]:
            continue
        dsc = dscs[0]["batchmatmul"]
        matches.append(
            {
                "path": str(path),
                "shape": dsc.get("N_"),
                "slices": sdsc.get("numWkSlicesPerDim_"),
                "cores": sdsc.get("numCoresUsed_"),
            }
        )
    require(len(matches) <= 1, f"multiple BMM descriptors below {bundle}")
    return matches[0] if matches else None


def patch_init_binary(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    desc = descriptor(bundle)
    report: dict[str, Any] = {
        "schema": "design_a_xrf_wrap_ablation_v1",
        "bundle": str(bundle),
        "descriptor": desc,
        "applied": False,
    }
    if desc is None:
        report["reason"] = "not_a_single_bmm_bundle"
        return report
    shape = desc["shape"] or {}
    slices = desc["slices"] or {}
    if (
        any(shape.get(key) != value for key, value in TARGET_BMM.items())
        or any(slices.get(key) != value for key, value in TARGET_SLICES.items())
        or desc["cores"] != TARGET_CORES
    ):
        report["reason"] = "descriptor_not_exact_target"
        return report

    binary_path = bundle / "spyreCodeDir/init_binary.bin"
    spyrecode_path = bundle / "spyreCodeDir/spyrecode.json"
    require(binary_path.is_file(), f"missing InitPacket binary: {binary_path}")
    require(spyrecode_path.is_file(), f"missing SpyreCode plan: {spyrecode_path}")
    original = binary_path.read_bytes()
    data = bytearray(original)

    init_pattern = b"".join(
        pack_word(word)
        for word in (XRF_SET_WRITE_ZERO, XRF_SET_READ_ZERO, OUTER_LOOP_200)
    )
    init_offsets = find_all(data, init_pattern)
    reset_offsets = find_all(data, pack_word(INNER_RESET_READ_ZERO_BE))
    require(
        len(init_offsets) == EXPECTED_PT_PROGRAMS,
        f"expected {EXPECTED_PT_PROGRAMS} XRF initializers, got {init_offsets}",
    )
    require(
        len(reset_offsets) == EXPECTED_PT_PROGRAMS,
        f"expected {EXPECTED_PT_PROGRAMS} inner resets, got {reset_offsets}",
    )
    for offset in reset_offsets:
        require(offset >= 4, f"reset has no preceding instruction: {offset}")
        preceding = struct.unpack_from("<I", data, offset - 4)[0]
        require(
            preceding == INNER_FINAL_FMA,
            f"unexpected inner-loop predecessor at {offset}: 0x{preceding:08x}",
        )

    for offset in init_offsets:
        struct.pack_into("<I", data, offset, XRF_SET_SIZE_16_AND_WRITE_ZERO)
    for offset in reset_offsets:
        struct.pack_into("<I", data, offset - 4, INNER_FINAL_FMA | 0x80000000)
        struct.pack_into("<I", data, offset, INNER_RESET_READ_ZERO)

    require(len(data) == len(original), "InitPacket size changed")
    require(
        not find_all(data, init_pattern),
        "unpatched XRF initializer remains",
    )
    require(
        not find_all(data, pack_word(INNER_RESET_READ_ZERO_BE)),
        "inner-loop reset still carries loop-end bit",
    )
    require(
        len(find_all(data, pack_word(XRF_SET_SIZE_16_AND_WRITE_ZERO)))
        == EXPECTED_PT_PROGRAMS,
        "wrong number of XRF-size instructions after patch",
    )
    require(
        len(find_all(data, pack_word(INNER_FINAL_FMA | 0x80000000)))
        == EXPECTED_PT_PROGRAMS,
        "wrong number of FMA loop ends after patch",
    )

    # Program-correction patch locations are a separate prefix packet in this
    # plan.  Fail if any location names an instruction flit modified here.
    patched_flits = {
        offset // 128
        for offset in init_offsets
        + reset_offsets
        + [offset - 4 for offset in reset_offsets]
    }
    spyrecode = json.loads(spyrecode_path.read_text())
    correction_flits: set[int] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if {"flitId", "sliceId", "stPosn", "endPosn"} <= value.keys():
                correction_flits.add(int(value["flitId"]))
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(spyrecode)
    require(
        patched_flits.isdisjoint(correction_flits),
        f"runtime correction overlaps patched flits: {patched_flits & correction_flits}",
    )

    binary_path.write_bytes(data)
    report.update(
        {
            "applied": True,
            "reason": "exact_target_patched",
            "binary_size": len(data),
            "before_sha256": sha256_bytes(original),
            "after_sha256": sha256_bytes(data),
            "xrf_size": 16,
            "initializer_offsets": init_offsets,
            "inner_reset_offsets": reset_offsets,
            "patched_flits": sorted(patched_flits),
            "runtime_correction_overlap": [],
            "expected_dynamic_effect": (
                "inner XRF read-pointer reset executes once after the loop "
                "instead of once per loop iteration"
            ),
        }
    )
    return report


def write_report(bundle: Path, report: dict[str, Any]) -> None:
    path = bundle / "xrf_wrap_ablation.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def wrapper(argv: list[str]) -> int:
    real_dxp = os.environ.get("REAL_DXP_STANDALONE")
    require(real_dxp, "REAL_DXP_STANDALONE is required")
    require(Path(real_dxp).is_file(), f"real DXP does not exist: {real_dxp}")
    completed = subprocess.run([real_dxp, *argv], check=False)
    if completed.returncode:
        return completed.returncode
    require("-d" in argv, f"DXP invocation has no -d bundle: {argv}")
    bundle = Path(argv[argv.index("-d") + 1]).resolve()
    report = patch_init_binary(bundle)
    write_report(bundle, report)
    print(
        "[xrf-wrap-ablation] "
        + json.dumps(
            {
                "bundle": str(bundle),
                "applied": report["applied"],
                "reason": report["reason"],
                "after_sha256": report.get("after_sha256"),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--patch-only":
        bundle = Path(sys.argv[2])
        report = patch_init_binary(bundle)
        write_report(bundle, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    raise SystemExit(wrapper(sys.argv[1:]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Decode a final DeepTools InitPacket into loop-aware program accounting.

DeepTools' ``initpacket.json`` is intentionally human-readable rather than
strict JSON: instruction words are emitted as unquoted hexadecimal values.
This parser streams only header and IBUFF records.  It reports the delivered
base-program slots as well as a loop-expanded issue proxy for each execution
unit.

The loop-expanded counts are program accounting, not a hardware instruction
trace or cycle count.  Patch records may specialize addresses and GTR values
per core; the base-program opcode and loop structure are what is decoded here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import median
from typing import Any


STICK_BYTES = 128

L3_COMMON = {
    0x00: "LARIMM",
    0x01: "ADDLARIMM",
    0x02: "SUBLARIMM",
    0x04: "EARIMM",
    0x05: "ADDEARIMM",
    0x08: "GTRIMM",
    0x0C: "SYNC",
    0x10: "LARREGCOPY",
    0x11: "MODLARREG",
    0x14: "EARREGCOPY",
    0x15: "MODEARREG",
    0x30: "MVLOOPCNT",
    0x31: "JIMMCOPY",
    0x32: "JCRSWAP",
    0x33: "JADD",
    0x34: "JCMP",
    0x35: "JSUB",
    0x36: "NOP",
    0x37: "RETURN",
    0x3C: "SJCMP",
}
L3_LOAD = {
    0x20: "LD",
    0x21: "LDM",
    0x22: "LDU",
    0x23: "LDMU",
    0x24: "LDZ",
    0x25: "LDAM",
    0x26: "LDZU",
    0x27: "LDAMU",
    0x28: "LDG",
    0x29: "LDGM",
    0x2A: "LDGU",
    0x2B: "LDGMU",
    0x2C: "LDIGM",
    0x2D: "LDIM",
    0x2E: "LDIGMU",
    0x2F: "LDIMU",
}
L3_STORE = {
    0x20: "ST",
    0x21: "STM",
    0x22: "STU",
    0x23: "STMU",
    0x24: "STZ",
    0x25: "STAM",
    0x27: "STAMU",
    0x28: "STG",
    0x2A: "STGU",
    0x2D: "STIM",
    0x2F: "STIMU",
}
LX = {
    0x00: "MODLRFREG",
    0x04: "LRFREGCOPY",
    0x05: "LRFCOPY",
    0x08: "MODLRFIMM",
    0x0C: "IMMCOPY",
    0x0D: "SUBLRFIMM",
    0x11: "LDSTU",
    0x19: "LDSTIU",
    0x1B: "LDCVTIU",
    0x21: "MVLOOPCNT",
    0x22: "JCMP",
    0x23: "JCRSWAP",
    0x24: "JADD",
    0x25: "JSUB",
    0x26: "SYNC",
    0x27: "JIMMCOPY",
    0x29: "SPMV",
    0x2A: "SAMV",
    0x2B: "SETDSTMASK",
    0x30: "NOP",
    0x31: "LDST",
    0x32: "SJCMP",
    0x39: "LDSTI",
    0x3B: "LDCVTI",
    0x3F: "RETURN",
}
L0 = {
    0x00: "MODLRFREG",
    0x04: "LRFREGCOPY",
    0x08: "MODLRFIMM",
    0x0C: "IMMCOPY",
    0x11: "LDSTU",
    0x19: "LDSTIU",
    0x21: "MVLOOPCNT",
    0x22: "JCMP",
    0x23: "JCRSWAP",
    0x24: "JADD",
    0x25: "JSUB",
    0x26: "SYNC",
    0x27: "JIMMCOPY",
    0x28: "TILEADV",
    0x30: "NOP",
    0x31: "LDST",
    0x32: "SJCMP",
    0x39: "LDSTI",
    0x3F: "RETURN",
}
PE_SFP = {
    0x00: "FMUL",
    0x04: "FMA",
    0x06: "FNMS",
    0x08: "ICVT",
    0x09: "IME",
    0x0A: "FCMP",
    0x0B: "FMINMAX",
    0x0D: "EE",
    0x0E: "SELECT",
    0x0F: "SHR",
    0x10: "LOGICAL",
    0x11: "FCVT",
    0x12: "MERGE",
    0x13: "COMPRESS",
    0x14: "FEST",
    0x15: "SPLAT",
    0x16: "GCVT",
    0x17: "PACK",
    0x18: "PERMUTE",
    0x19: "LOAD_LFSR",
    0x1A: "COPY_LFSR",
    0x21: "MVLOOPCNT",
    0x25: "IMMCOPY",
    0x26: "SYNC",
    0x27: "JCRSWAP",
    0x2A: "REDUCE",
    0x30: "NOP",
    0x31: "JADD",
    0x32: "JSUB",
    0x33: "JIMMCOPY",
    0x34: "JCMP",
    0x35: "SJCMP",
    0x36: "SETDEST",
    0x3F: "RETURN",
}
PT = {
    0x03: "FMA",
    0x04: "IMA8",
    0x05: "FMA8",
    0x08: "IMA4",
    0x09: "XMA4",
    0x0A: "FMA4",
    0x20: "NOP",
    0x21: "MVLOOPCNT",
    0x22: "INCRMASK",
    0x23: "SETMASK",
    0x26: "SYNC",
    0x27: "JCRSWAP",
    0x28: "XRFACCESS",
    0x31: "JADD",
    0x32: "JSUB",
    0x33: "JIMMCOPY",
    0x34: "JCMP",
    0x35: "SJCMP",
    0x3F: "RETURN",
}

TYPE_RE = re.compile(r'"type"\s*:\s*"(header|ibuff)"')
NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
HEADER_RE = re.compile(r'"myHeader"\s*:\s*"([^"]+)"')
TARGET_RE = re.compile(r'"targetUnits"\s*:\s*\[\s*(.*?)\s*\]')
CORE_RE = re.compile(r'"targetCores"\s*:\s*\[\s*(.*?)\s*\]')
VALUE_RE = re.compile(r'"value"\s*:\s*\[\s*(.*?)\s*\]')
QUOTED_RE = re.compile(r'"([^"]+)"')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def root_unit(target: str) -> str:
    if target.startswith("ptrow"):
        return "pt"
    for prefix in (
        "l3lu",
        "l3su",
        "lxlu",
        "lxsu",
        "l0lu",
        "l0su",
        "sfp",
        "pe",
    ):
        if target.startswith(prefix):
            return prefix
    return target


def opcode_name(unit: str, word: int) -> str:
    opcode = word & 0x3F
    if unit == "l3lu":
        table = L3_COMMON | L3_LOAD
    elif unit == "l3su":
        table = L3_COMMON | L3_STORE
    elif unit.startswith("lx"):
        table = LX
    elif unit.startswith("l0"):
        table = L0
    elif unit in ("pe", "sfp"):
        table = PE_SFP
    elif unit == "pt":
        table = PT
    else:
        table = {}
    return table.get(opcode, f"UNKNOWN_0x{opcode:02x}")


def parse_programs(path: Path) -> list[dict[str, Any]]:
    headers: dict[str, dict[str, Any]] = {}
    program_words: dict[str, list[int]] = defaultdict(list)
    header_order: list[str] = []
    state: str | None = None
    record: dict[str, Any] = {}

    def finish() -> None:
        nonlocal state, record
        if state == "header":
            name = record.get("name")
            targets = record.get("targets")
            cores = record.get("cores")
            if isinstance(name, str) and isinstance(targets, list):
                require(
                    isinstance(cores, list) and cores, f"header has no cores: {name}"
                )
                headers[name] = {"targets": targets, "cores": cores}
                header_order.append(name)
        elif state == "ibuff":
            header = record.get("header")
            words = record.get("words")
            if isinstance(header, str) and isinstance(words, list):
                require(header in headers, f"unknown header: {header}")
                program_words[header].extend(words)
        state = None
        record = {}

    with path.open(errors="replace") as stream:
        for line in stream:
            type_match = TYPE_RE.search(line)
            if type_match:
                if state is not None:
                    finish()
                state = type_match.group(1)
                record = {}
            if state is None:
                continue
            if match := NAME_RE.search(line):
                record["name"] = match.group(1)
            if match := TARGET_RE.search(line):
                record["targets"] = QUOTED_RE.findall(match.group(1))
            if match := CORE_RE.search(line):
                text = match.group(1).strip()
                record["cores"] = (
                    [int(token.strip()) for token in text.split(",")] if text else []
                )
            if match := VALUE_RE.search(line):
                record["words"] = [
                    int(token.strip(), 16) for token in match.group(1).split(",")
                ]
            if match := HEADER_RE.search(line):
                record["header"] = match.group(1)
            if line.strip() in {"}", "},"}:
                finish()
    if state is not None:
        finish()

    programs = []
    for name in header_order:
        words = program_words.get(name, [])
        if not words:
            continue
        targets = headers[name]["targets"]
        units = {root_unit(target) for target in targets}
        require(len(units) == 1, f"mixed-unit header: {name}: {targets}")
        unit = next(iter(units))
        if unit in {"pe", "sfp"}:
            require(len(words) % 4 == 0, f"misaligned 64-bit program: {name}")
            instructions = []
            for index in range(0, len(words), 4):
                instructions.extend(
                    (
                        words[index] | (words[index + 1] << 32),
                        words[index + 2] | (words[index + 3] << 32),
                    )
                )
        else:
            instructions = words
        programs.append(
            {
                "header": name,
                "unit": unit,
                "targets": targets,
                "cores": headers[name]["cores"],
                "instructions": instructions,
            }
        )
    require(programs, f"no executable programs in {path}")
    return programs


def loop_count(unit: str, word: int) -> int:
    if unit == "pt":
        return (word >> 6) & 0xFFFF
    if unit in {"pe", "sfp"}:
        return (word >> 11) & 0xFFFF
    return (word >> 10) & 0xFFFF


def branch_end(unit: str, word: int) -> bool:
    shift = 35 if unit in {"pe", "sfp"} else 31
    return bool((word >> shift) & 1)


def build_loop_tree(unit: str, instructions: list[int]) -> list[dict[str, Any]]:
    root: list[dict[str, Any]] = []
    stack = [root]
    for word in instructions:
        opcode = opcode_name(unit, word)
        instruction = {"kind": "instruction", "opcode": opcode, "word": word}
        if opcode == "MVLOOPCNT":
            count = loop_count(unit, word)
            require(count > 0, f"zero-count static loop in {unit}: 0x{word:x}")
            node = {"kind": "loop", "count": count, "setup": instruction, "body": []}
            stack[-1].append(node)
            stack.append(node["body"])
        else:
            stack[-1].append(instruction)
        if branch_end(unit, word):
            require(len(stack) > 1, f"loop end without loop in {unit}: 0x{word:x}")
            stack.pop()
    require(len(stack) == 1, f"unterminated loop in {unit}")
    return root


def add_issue_counts(
    unit: str,
    nodes: list[dict[str, Any]],
    counts: Counter[str],
    *,
    multiplier: int = 1,
) -> None:
    for node in nodes:
        if node["kind"] == "loop":
            counts["MVLOOPCNT"] += multiplier
            add_issue_counts(
                unit,
                node["body"],
                counts,
                multiplier=multiplier * node["count"],
            )
            continue
        opcode = node["opcode"]
        word = node["word"]
        counts[opcode] += multiplier
        if unit == "pt" and opcode == "FMA":
            unroll = 1 << ((word >> 27) & 0x3)
            src0 = (word >> 6) & 0xF
            target_register = (word >> 20) & 0xF
            if src0 == 14:
                category = "PT_COMPUTE_FMA_UNROLLED"
            elif target_register == 14:
                category = "PT_XRF_LOAD_FMA_UNROLLED"
            else:
                category = "PT_OTHER_FMA_UNROLLED"
            counts[category] += multiplier * unroll


def execute_l3(
    nodes: list[dict[str, Any]],
    report: Counter[str],
    gtr: dict[int, int],
    *,
    multiplier: int = 1,
) -> None:
    for node in nodes:
        if node["kind"] == "loop":
            report["MVLOOPCNT"] += multiplier
            execute_l3(
                node["body"],
                report,
                gtr,
                multiplier=multiplier * node["count"],
            )
            continue
        opcode = node["opcode"]
        word = node["word"]
        report[opcode] += multiplier
        if opcode == "GTRIMM":
            index = (word >> 6) & 0xF
            immediate = (word >> 10) & 0x3FFF
            gtr[index] = immediate & 0x3F
            report[f"GTR_SHARERS_{gtr[index]}"] += multiplier
            continue
        if opcode not in {"LDM", "LDMU", "LDGM", "LDGMU"}:
            continue
        encoded_burst = (word >> 22) & 0x1F
        sticks = 32 if encoded_burst == 0 else encoded_burst
        report[f"{opcode}_REQUESTS"] += multiplier
        report[f"{opcode}_DELIVERED_STICKS"] += multiplier * sticks
        if opcode in {"LDGM", "LDGMU"}:
            group = (word >> 27) & 0x7
            require(group in gtr, f"load uses unset GTR {group}")
            sharers = gtr[group]
            require(sharers > 0 and sticks % sharers == 0, (sticks, sharers))
            report[f"LDGM_SHARERS_{sharers}_REQUESTS"] += multiplier
            report[f"LDGM_SHARERS_{sharers}_DELIVERED_STICKS"] += multiplier * sticks
            report["ESTIMATED_UNIQUE_HBM_STICKS"] += multiplier * sticks // sharers
        else:
            report["ESTIMATED_UNIQUE_HBM_STICKS"] += multiplier * sticks


def summarize(path: Path) -> dict[str, Any]:
    programs = parse_programs(path)
    static_counts: dict[str, Counter[str]] = defaultdict(Counter)
    issue_counts: dict[str, Counter[str]] = defaultdict(Counter)
    issue_pressure: dict[str, list[int]] = defaultdict(list)
    l3_report: Counter[str] = Counter()
    program_rows = []

    for program in programs:
        unit = program["unit"]
        weight = len(program["targets"]) * len(program["cores"])
        # LX IBUFF records contain an embedded function directory between live
        # instruction regions.  The directory is delimited by an all-ones word,
        # but its remaining words are not tagged separately in InitPacket JSON.
        # Treating those words as LX opcodes fabricates instructions and loop
        # endings.  Preserve their slot cost, but do not claim an instruction
        # decode or loop expansion for that mixed record.
        has_embedded_lx_data = unit.startswith("lx") and any(
            word == 0xFFFFFFFF for word in program["instructions"]
        )
        if has_embedded_lx_data:
            static = Counter(
                {"MIXED_CODE_AND_PACKET_DATA": len(program["instructions"])}
            )
        else:
            static = Counter(
                opcode_name(unit, word) for word in program["instructions"]
            )
        static_counts[unit].update(
            {key: value * weight for key, value in static.items()}
        )

        tree = None
        per_entry: Counter[str] = Counter()
        raw_issues = None
        if not has_embedded_lx_data:
            tree = build_loop_tree(unit, program["instructions"])
            add_issue_counts(unit, tree, per_entry)
            issue_counts[unit].update(
                {key: value * weight for key, value in per_entry.items()}
            )
            raw_issues = sum(
                value for key, value in per_entry.items() if not key.startswith("PT_")
            )
            issue_pressure[unit].extend([raw_issues] * len(program["targets"]))
        program_rows.append(
            {
                "header": program["header"],
                "unit": unit,
                "target_count": len(program["targets"]),
                "core_count": len(program["cores"]),
                "static_slots_per_target": len(program["instructions"]),
                "loop_expanded_issues_per_target": raw_issues,
                "loop_expansion_status": (
                    "skipped_mixed_lx_code_and_packet_data"
                    if has_embedded_lx_data
                    else "decoded"
                ),
            }
        )
        if unit == "l3lu":
            require(tree is not None, "L3 program unexpectedly skipped")
            per_entry_l3: Counter[str] = Counter()
            execute_l3(tree, per_entry_l3, {})
            l3_report.update(
                {key: value * weight for key, value in per_entry_l3.items()}
            )

    pt = issue_counts.get("pt", Counter())
    delivered_sticks = sum(
        value
        for key, value in l3_report.items()
        if key.endswith("_DELIVERED_STICKS")
        and key.startswith("LD")
        and not key.startswith("LDGM_SHARERS_")
    )
    unique_sticks = l3_report["ESTIMATED_UNIQUE_HBM_STICKS"]
    require(unique_sticks > 0, "no L3 HBM load traffic")
    return {
        "schema": "final_initpacket_program_accounting_v1",
        "source": str(path),
        "source_sha256": sha256(path),
        "method": {
            "static_scope": "delivered base IBUFF slots weighted by target units and cores",
            "dynamic_scope": "decoded base-program loops expanded; each unit entry invoked once",
            "dynamic_exclusions": (
                "LX IBUFF records containing an embedded function directory are not "
                "loop-expanded because InitPacket JSON does not tag code versus data"
            ),
            "patch_scope": "patch records are not applied; addresses and per-core GTR specialization may differ",
            "not_claimed": [
                "hardware executed-instruction trace",
                "hardware active or stall cycles",
                "ring-link utilization",
            ],
        },
        "program_count": len(programs),
        "programs": program_rows,
        "static_delivered_base": {
            "slots": sum(sum(values.values()) for values in static_counts.values()),
            "opcode_hist_by_unit": {
                unit: dict(sorted(values.items()))
                for unit, values in sorted(static_counts.items())
            },
        },
        "loop_expanded": {
            "decoded_instruction_issues": sum(
                sum(value for key, value in values.items() if not key.startswith("PT_"))
                for values in issue_counts.values()
            ),
            "opcode_hist_by_unit": {
                unit: dict(sorted(values.items()))
                for unit, values in sorted(issue_counts.items())
            },
            "issue_pressure_per_unit_entry": {
                unit: {
                    "entries": len(values),
                    "minimum": min(values),
                    "median": median(values),
                    "maximum": max(values),
                }
                for unit, values in sorted(issue_pressure.items())
            },
        },
        "pt": {
            "compute_fma_unrolled": pt["PT_COMPUTE_FMA_UNROLLED"],
            "xrf_load_fma_unrolled": pt["PT_XRF_LOAD_FMA_UNROLLED"],
            "other_fma_unrolled": pt["PT_OTHER_FMA_UNROLLED"],
            "fma_instruction_issues": pt["FMA"],
            "xrfaccess_issues": pt["XRFACCESS"],
            "loop_setup_issues": pt["MVLOOPCNT"],
            "nop_issues": pt["NOP"],
        },
        "l3_loads": {
            "recipient_delivered_bytes": delivered_sticks * STICK_BYTES,
            "estimated_unique_hbm_response_bytes": unique_sticks * STICK_BYTES,
            "recipient_bytes_per_unique_hbm_byte": delivered_sticks / unique_sticks,
            "loop_expanded": dict(sorted(l3_report.items())),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initpacket", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    result = summarize(args.initpacket)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

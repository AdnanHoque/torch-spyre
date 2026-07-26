#!/usr/bin/env python3
"""Stream an InitPacket JSON dump and summarize ISA opcode slots.

DeepTools' InitPacket exporter is intentionally human-readable but is not
strict JSON (hex words are unquoted).  This parser only consumes header and
IBUFF records, which is sufficient for a static base-program instruction mix.
Patch traffic is summarized separately by initpacketstat.json.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


L3_COMMON = {
    0x00: "LARIMM", 0x01: "ADDLARIMM", 0x02: "SUBLARIMM",
    0x04: "EARIMM", 0x05: "ADDEARIMM", 0x08: "GTRIMM",
    0x0C: "SYNC", 0x10: "LARREGCOPY", 0x11: "MODLARREG",
    0x14: "EARREGCOPY", 0x15: "MODEARREG", 0x30: "MVLOOPCNT",
    0x31: "JIMMCOPY", 0x32: "JCRSWAP", 0x33: "JADD",
    0x34: "JCMP", 0x35: "JSUB", 0x36: "NOP", 0x37: "RETURN",
    0x3C: "SJCMP",
}
L3_LOAD = {
    0x20: "LD", 0x21: "LDM", 0x22: "LDU", 0x23: "LDMU",
    0x24: "LDZ", 0x25: "LDAM", 0x26: "LDZU", 0x27: "LDAMU",
    0x28: "LDG", 0x29: "LDGM", 0x2A: "LDGU", 0x2B: "LDGMU",
    0x2C: "LDIGM", 0x2D: "LDIM", 0x2E: "LDIGMU", 0x2F: "LDIMU",
}
L3_STORE = {
    0x20: "ST", 0x21: "STM", 0x22: "STU", 0x23: "STMU",
    0x24: "STZ", 0x25: "STAM", 0x27: "STAMU", 0x28: "STG",
    0x2A: "STGU", 0x2D: "STIM", 0x2F: "STIMU",
}
LX = {
    0x00: "MODLRFREG", 0x04: "LRFREGCOPY", 0x05: "LRFCOPY",
    0x08: "MODLRFIMM", 0x0C: "IMMCOPY", 0x0D: "SUBLRFIMM",
    0x11: "LDSTU", 0x19: "LDSTIU", 0x1B: "LDCVTIU",
    0x21: "MVLOOPCNT", 0x22: "JCMP", 0x23: "JCRSWAP",
    0x24: "JADD", 0x25: "JSUB", 0x26: "SYNC", 0x27: "JIMMCOPY",
    0x29: "SPMV", 0x2A: "SAMV", 0x2B: "SETDSTMASK", 0x30: "NOP",
    0x31: "LDST", 0x32: "SJCMP", 0x39: "LDSTI", 0x3B: "LDCVTI",
    0x3F: "RETURN",
}
L0 = {
    0x00: "MODLRFREG", 0x04: "LRFREGCOPY", 0x08: "MODLRFIMM",
    0x0C: "IMMCOPY", 0x11: "LDSTU", 0x19: "LDSTIU",
    0x21: "MVLOOPCNT", 0x22: "JCMP", 0x23: "JCRSWAP",
    0x24: "JADD", 0x25: "JSUB", 0x26: "SYNC", 0x27: "JIMMCOPY",
    0x28: "TILEADV", 0x30: "NOP", 0x31: "LDST", 0x32: "SJCMP",
    0x39: "LDSTI", 0x3F: "RETURN",
}
PE_SFP = {
    0x00: "FMUL", 0x04: "FMA", 0x06: "FNMS", 0x08: "ICVT",
    0x09: "IME", 0x0A: "FCMP", 0x0B: "FMINMAX", 0x0D: "EE",
    0x0E: "SELECT", 0x0F: "SHR", 0x10: "LOGICAL", 0x11: "FCVT",
    0x12: "MERGE", 0x13: "COMPRESS", 0x14: "FEST", 0x15: "SPLAT",
    0x16: "GCVT", 0x17: "PACK", 0x18: "PERMUTE", 0x19: "LOAD_LFSR",
    0x1A: "COPY_LFSR", 0x21: "MVLOOPCNT", 0x25: "IMMCOPY",
    0x26: "SYNC", 0x27: "JCRSWAP", 0x2A: "REDUCE", 0x30: "NOP",
    0x31: "JADD", 0x32: "JSUB", 0x33: "JIMMCOPY", 0x34: "JCMP",
    0x35: "SJCMP", 0x36: "SETDEST", 0x3F: "RETURN",
}
PT = {
    0x03: "FMA", 0x04: "IMA8", 0x05: "FMA8", 0x08: "IMA4",
    0x09: "XMA4", 0x0A: "FMA4", 0x20: "NOP", 0x21: "MVLOOPCNT",
    0x22: "INCRMASK", 0x23: "SETMASK", 0x26: "SYNC",
    0x27: "JCRSWAP", 0x28: "XRFACCESS", 0x31: "JADD", 0x32: "JSUB",
    0x33: "JIMMCOPY", 0x34: "JCMP", 0x35: "SJCMP", 0x3F: "RETURN",
}

TYPE_RE = re.compile(r'"type"\s*:\s*"(header|ibuff)"')
NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
HEADER_RE = re.compile(r'"myHeader"\s*:\s*"([^"]+)"')
TARGET_RE = re.compile(r'"targetUnits"\s*:\s*\[\s*(.*?)\s*\]')
VALUE_RE = re.compile(r'"value"\s*:\s*\[\s*(.*?)\s*\]')
QUOTED_RE = re.compile(r'"([^"]+)"')


def root_unit(target: str) -> str:
    if target.startswith("ptrow"):
        return "pt"
    for prefix in ("l3lu", "l3su", "lxlu", "lxsu", "l0lu", "l0su", "sfp", "pe"):
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


def instruction_class(unit: str, name: str) -> str:
    if name == "SYNC":
        return "sync"
    if name in {"RETURN", "NOP"}:
        return name.lower()
    if name in {"MVLOOPCNT", "JADD", "JSUB", "JIMMCOPY", "JCMP", "SJCMP", "JCRSWAP"}:
        return "loop_control"
    if unit.startswith("l3") and (name.startswith("LD") or name.startswith("ST")):
        return "hbm_or_intercore_transfer"
    if unit.startswith("lx") and name in {"LDST", "LDSTI", "LDSTU", "LDSTIU", "LDCVTI", "LDCVTIU", "SAMV", "SPMV"}:
        return "lx_transfer"
    if unit.startswith("l0") and name in {"LDST", "LDSTI", "LDSTU", "LDSTIU", "TILEADV"}:
        return "l0_transfer"
    if unit == "pt" and name in {"FMA", "FMA4", "FMA8", "IMA4", "IMA8", "XMA4"}:
        return "pt_compute"
    if unit in {"pe", "sfp"} and name not in {"UNKNOWN"}:
        return "vector_compute_or_dataflow"
    if name.startswith("UNKNOWN"):
        return "unknown"
    return "register_or_address_control"


def parse(path: Path) -> dict[str, object]:
    headers: dict[str, list[str]] = {}
    ibuff_flits = Counter()
    opcode_hist: dict[str, Counter] = defaultdict(Counter)
    class_hist: dict[str, Counter] = defaultdict(Counter)
    state: str | None = None
    record: dict[str, object] = {}

    def finish() -> None:
        nonlocal state, record
        if state == "header":
            name = record.get("name")
            targets = record.get("targets")
            if isinstance(name, str) and isinstance(targets, list):
                headers[name] = targets
        elif state == "ibuff":
            header = record.get("header")
            words = record.get("words")
            if isinstance(header, str) and isinstance(words, list):
                targets = headers.get(header, ["unknown"])
                unit = root_unit(targets[0])
                ibuff_flits[unit] += 1
                if unit in {"pe", "sfp"}:
                    slots = [words[0] | (words[1] << 32), words[2] | (words[3] << 32)]
                else:
                    slots = words
                for word in slots:
                    opname = opcode_name(unit, word)
                    opcode_hist[unit][opname] += 1
                    class_hist[unit][instruction_class(unit, opname)] += 1
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
            if (match := NAME_RE.search(line)):
                record["name"] = match.group(1)
            if (match := TARGET_RE.search(line)):
                record["targets"] = QUOTED_RE.findall(match.group(1))
            if (match := VALUE_RE.search(line)):
                record["words"] = [int(token.strip(), 16) for token in match.group(1).split(",")]
            if (match := HEADER_RE.search(line)):
                record["header"] = match.group(1)
            if line.strip() in {"}", "},"}:
                finish()
    if state is not None:
        finish()

    return {
        "source": str(path),
        "header_count": len(headers),
        "ibuff_flits_by_unit": dict(sorted(ibuff_flits.items())),
        "ibuff_slots_by_unit": {
            unit: sum(hist.values()) for unit, hist in sorted(opcode_hist.items())
        },
        "opcode_hist_by_unit": {
            unit: dict(hist.most_common()) for unit, hist in sorted(opcode_hist.items())
        },
        "class_hist_by_unit": {
            unit: dict(hist.most_common()) for unit, hist in sorted(class_hist.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("initpacket", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(parse(args.initpacket), indent=2) + "\n")


if __name__ == "__main__":
    main()

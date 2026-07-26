#!/usr/bin/env python3
"""Small dependency-free CBOR-to-JSON helper for SenDNN graph artifacts."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


BREAK = object()


class Decoder:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def take(self, size: int) -> bytes:
        value = self.payload[self.offset : self.offset + size]
        if len(value) != size:
            raise ValueError(f"truncated CBOR at byte {self.offset}")
        self.offset += size
        return value

    def uint(self, additional: int) -> int | None:
        if additional < 24:
            return additional
        if additional == 24:
            return self.take(1)[0]
        if additional == 25:
            return struct.unpack(">H", self.take(2))[0]
        if additional == 26:
            return struct.unpack(">I", self.take(4))[0]
        if additional == 27:
            return struct.unpack(">Q", self.take(8))[0]
        if additional == 31:
            return None
        raise ValueError(f"reserved additional info {additional} at byte {self.offset - 1}")

    def value(self):
        initial = self.take(1)[0]
        major, additional = initial >> 5, initial & 0x1F
        if major == 7 and additional == 31:
            return BREAK
        argument = self.uint(additional)
        if major == 0:
            return argument
        if major == 1:
            return -1 - argument
        if major in (2, 3):
            if argument is None:
                chunks = []
                while True:
                    item = self.value()
                    if item is BREAK:
                        break
                    chunks.append(item)
                return b"".join(chunks) if major == 2 else "".join(chunks)
            raw = self.take(argument)
            return raw if major == 2 else raw.decode("utf-8")
        if major == 4:
            if argument is None:
                result = []
                while True:
                    item = self.value()
                    if item is BREAK:
                        return result
                    result.append(item)
            return [self.value() for _ in range(argument)]
        if major == 5:
            result = {}
            if argument is None:
                while True:
                    key = self.value()
                    if key is BREAK:
                        return result
                    result[key] = self.value()
            else:
                for _ in range(argument):
                    key = self.value()
                    result[key] = self.value()
            return result
        if major == 6:
            return self.value()  # Tags are not needed for these JSON-origin graphs.
        if major == 7:
            if additional == 20:
                return False
            if additional == 21:
                return True
            if additional in (22, 23):
                return None
            if additional == 24:
                return argument
            if additional == 25:
                return struct.unpack(">e", struct.pack(">H", argument))[0]
            if additional == 26:
                return struct.unpack(">f", struct.pack(">I", argument))[0]
            if additional == 27:
                return struct.unpack(">d", struct.pack(">Q", argument))[0]
        raise ValueError(f"unsupported CBOR major={major} additional={additional}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    decoder = Decoder(args.input.read_bytes())
    value = decoder.value()
    if decoder.offset != len(decoder.payload):
        raise ValueError(f"{len(decoder.payload) - decoder.offset} trailing bytes")
    json.dump(value, fp=__import__("sys").stdout, separators=(",", ":"))
    print()


if __name__ == "__main__":
    main()

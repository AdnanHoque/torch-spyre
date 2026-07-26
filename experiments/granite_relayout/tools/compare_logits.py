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

"""Compare two directories of saved Torch logits bit-for-bit."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    reference = sorted(args.reference.glob("*.pt"))
    candidate = sorted(args.candidate.glob("*.pt"))
    if not reference or [p.name for p in reference] != [p.name for p in candidate]:
        print("FAIL: logit file sets differ")
        return 1

    ok = True
    for ref_path, got_path in zip(reference, candidate):
        ref = torch.load(ref_path, map_location="cpu", weights_only=True)
        got = torch.load(got_path, map_location="cpu", weights_only=True)
        equal = ref.shape == got.shape and torch.equal(ref, got)
        max_abs = float((ref - got).abs().max()) if ref.shape == got.shape else float("inf")
        print(f"{ref_path.name}: bit_exact={equal} max_abs={max_abs}")
        ok &= equal
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


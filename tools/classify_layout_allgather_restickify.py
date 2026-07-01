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

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch_spyre._inductor.layout_allgather_restickify import (
    classify_layout_allgather_restickify_sdsc_triplet,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a flash mul -> ReStickifyOpHBM -> batchmatmul SDSC "
            "snippet as layout_allgather_restickify."
        )
    )
    parser.add_argument("snippet", type=Path, help="Path to sdsc_triplet_snippets.json")
    args = parser.parse_args()

    triplet = json.loads(args.snippet.read_text())
    classification = classify_layout_allgather_restickify_sdsc_triplet(triplet)
    if classification is None:
        print(json.dumps({"classification": None}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(classification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

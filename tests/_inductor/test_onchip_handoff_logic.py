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

"""Standalone tests for the Tier 1 on-chip handoff planning math.

The planner's IR traversal needs torch / the compiled extension and cannot run
in isolation, so this exercises only the pure transfer-cost decisions the
planner relies on. Like ``test_restickify_cost.py`` it loads
``restickify_cost.py`` by file path via ``importlib`` so it never imports the
``torch_spyre`` package. Also pytest-compatible: collect ``test_*`` normally.
"""

from __future__ import annotations

import importlib.util
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.normpath(
    os.path.join(
        _HERE, "..", "..", "torch_spyre", "_inductor", "restickify_cost.py"
    )
)

_spec = importlib.util.spec_from_file_location("restickify_cost", _MODULE_PATH)
rc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rc)


# --------------------------------------------------------------------------- #
# Identity ownership: the planner would SKIP this edge.
# --------------------------------------------------------------------------- #
def test_identity_ownership_has_no_remote_transfers():
    # Producer and consumer split the same dim the same way with an identity
    # symbol map => every owned tile already lives on its consuming core, so
    # the planner sees remote_elements == 0 and skips the edge.
    sizes = {"d0": 2048, "d1": 256}
    splits = {"d0": 8, "d1": 1}
    mapping = rc.materialize_default_core_mapping(["d0", "d1"], splits)
    symbol_map = {"d0": "d0", "d1": "d1"}

    transfers, summary = rc.build_transfer_plan(
        sizes, sizes, splits, splits, mapping, mapping, symbol_map, 32
    )

    assert summary["remote_elements"] == 0
    assert summary["total_byte_hops"] == 0
    assert summary["max_hops"] == 0
    assert all(t["hops"] == 0 for t in transfers)
    assert all(t["src_core"] == t["dst_core"] for t in transfers)
    # All elements are conserved and stay local.
    assert summary["local_elements"] == math.prod(sizes.values())


# --------------------------------------------------------------------------- #
# Divergent ownership, same sizes: the planner would RECORD this edge.
# --------------------------------------------------------------------------- #
def test_divergent_ownership_same_size_is_remote_and_conserved():
    # Producer splits d1 8-way; consumer splits d0 8-way. Same total tensor and
    # same core count, but orthogonal partitions => the activation must shuffle
    # cross-core. This is the canonical Tier 1 same-layout handoff.
    sizes = {"d0": 1024, "d1": 512}
    producer_splits = {"d0": 1, "d1": 8}
    consumer_splits = {"d0": 8, "d1": 1}
    producer_mapping = rc.materialize_default_core_mapping(
        ["d0", "d1"], producer_splits
    )
    consumer_mapping = rc.materialize_default_core_mapping(
        ["d0", "d1"], consumer_splits
    )
    symbol_map = {"d0": "d0", "d1": "d1"}

    transfers, summary = rc.build_transfer_plan(
        sizes,
        sizes,
        producer_splits,
        consumer_splits,
        producer_mapping,
        consumer_mapping,
        symbol_map,
        32,
    )

    # Orthogonal 8x8 grids => every producer core overlaps every consumer core.
    assert summary["total_transfers"] == 8 * 8
    assert summary["remote_elements"] > 0
    assert summary["max_hops"] > 0

    # Conservation: every element is accounted for exactly once across the
    # local + remote split, independent of the product of sizes.
    total = summary["local_elements"] + summary["remote_elements"]
    assert total == math.prod(sizes.values())

    # Closed-form cross-check on the byte-hops (elem_size factored out: the
    # cost core reports total_byte_hops at elem_size == 1).
    overlap = (1024 // 8) * (512 // 8)
    sum_dist = sum(
        rc.ring_distance(p, c, 32) for p in range(8) for c in range(8)
    )
    assert summary["total_byte_hops"] == sum_dist * overlap


def test_divergent_ownership_reverse_map_max_hops_positive():
    # A reversed consumer ownership of the same split dim also forces remote
    # movement: the producer's slice k is consumed by core (n-1-k).
    sizes = {"d0": 1024, "d1": 64}
    splits = {"d0": 4, "d1": 1}
    producer_mapping = rc.materialize_default_core_mapping(["d0", "d1"], splits)
    consumer_mapping = {str(c): {"d0": 3 - c, "d1": 0} for c in range(4)}
    symbol_map = {"d0": "d0", "d1": "d1"}

    _transfers, summary = rc.build_transfer_plan(
        sizes,
        sizes,
        splits,
        splits,
        producer_mapping,
        consumer_mapping,
        symbol_map,
        32,
    )

    assert summary["remote_elements"] > 0
    assert summary["max_hops"] > 0
    total = summary["local_elements"] + summary["remote_elements"]
    assert total == math.prod(sizes.values())


# --------------------------------------------------------------------------- #
# standalone runner
# --------------------------------------------------------------------------- #
def _run_all():
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print()
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())

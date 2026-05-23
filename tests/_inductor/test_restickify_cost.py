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

"""Standalone tests for the pure restickify cost-model core.

Loads ``restickify_cost.py`` by file path via ``importlib`` so it never
imports the ``torch_spyre`` package (no torch / compiled extension needed).
Also pytest-compatible: collect ``test_*`` functions normally.
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
# ring_distance
# --------------------------------------------------------------------------- #
def test_ring_distance_adjacent():
    assert rc.ring_distance(0, 1, 32) == 1
    assert rc.ring_distance(5, 6, 32) == 1


def test_ring_distance_opposite():
    assert rc.ring_distance(0, 16, 32) == 16


def test_ring_distance_wraparound():
    # 31 and 0 are adjacent across the wrap.
    assert rc.ring_distance(31, 0, 32) == 1
    # 30 -> 1 is 3 hops the short way (30->31->0->1).
    assert rc.ring_distance(30, 1, 32) == 3


def test_ring_distance_invalid_ring_size_raises():
    for bad in (0, -1, -32):
        try:
            rc.ring_distance(0, 1, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for ring_size={bad}")


# --------------------------------------------------------------------------- #
# materialize_default_core_mapping
# --------------------------------------------------------------------------- #
def test_materialize_default_1d_split():
    mapping = rc.materialize_default_core_mapping(["d0", "d1"], {"d0": 4})
    # 4 cores, each owns one d0 slice, d1 unsplit (always slice 0).
    assert mapping == {
        "0": {"d0": 0, "d1": 0},
        "1": {"d0": 1, "d1": 0},
        "2": {"d0": 2, "d1": 0},
        "3": {"d0": 3, "d1": 0},
    }


def test_materialize_default_2d_split():
    mapping = rc.materialize_default_core_mapping(["d0", "d1"], {"d0": 2, "d1": 2})
    # d0 is the inner (fastest) dim, d1 the outer dim.
    assert mapping == {
        "0": {"d0": 0, "d1": 0},
        "1": {"d0": 1, "d1": 0},
        "2": {"d0": 0, "d1": 1},
        "3": {"d0": 1, "d1": 1},
    }


# --------------------------------------------------------------------------- #
# estimate_byte_hops_from_mappings
# --------------------------------------------------------------------------- #
def test_estimate_byte_hops_aligned_is_zero():
    # Producer and restickify both split the SAME dim (d0) the same way,
    # with an identity symbol map => every owned tile already lives on the
    # right core => zero byte-hops.
    sizes = {"d0": 2048, "d1": 2048}
    splits = {"d0": 32, "d1": 1}
    mapping = rc.materialize_default_core_mapping(["d0", "d1"], splits)
    symbol_map = {"d0": "d0", "d1": "d1"}
    bytes_moved, byte_hops, max_hops = rc.estimate_byte_hops_from_mappings(
        sizes, sizes, splits, splits, mapping, mapping, symbol_map, 2, 32
    )
    assert bytes_moved == 2048 * 2048 * 2
    assert byte_hops == 0
    assert max_hops == 0


def test_estimate_byte_hops_orthogonal():
    # Producer splits d1 32-way; restickify splits d0 32-way. Identity symbol
    # map. Every producer core overlaps every restickify core with a
    # 64(d0) x 64(d1) = 4096-element tile. Hand-computed against the closed
    # form below.
    sizes = {"d0": 2048, "d1": 2048}
    producer_splits = {"d0": 1, "d1": 32}
    restickify_splits = {"d0": 32, "d1": 1}
    producer_mapping = rc.materialize_default_core_mapping(
        ["d0", "d1"], producer_splits
    )
    restickify_mapping = rc.materialize_default_core_mapping(
        ["d0", "d1"], restickify_splits
    )
    symbol_map = {"d0": "d0", "d1": "d1"}
    elem = 2
    bytes_moved, byte_hops, max_hops = rc.estimate_byte_hops_from_mappings(
        sizes,
        sizes,
        producer_splits,
        restickify_splits,
        producer_mapping,
        restickify_mapping,
        symbol_map,
        elem,
        32,
    )

    # Producer core p owns d1 slice p; restickify core r owns d0 slice r.
    # ring_distance is between core ids p and r.
    overlap = 64 * 64
    sum_dist = sum(
        rc.ring_distance(p, r, 32) for p in range(32) for r in range(32)
    )
    expected_byte_hops = sum_dist * overlap * elem

    assert bytes_moved == 2048 * 2048 * elem
    assert byte_hops > 0
    assert byte_hops == expected_byte_hops == 67108864
    assert max_hops == 16


# --------------------------------------------------------------------------- #
# producer_aligned_dim_order
# --------------------------------------------------------------------------- #
def test_producer_aligned_dim_order_dominant_first():
    # Restickify dims a, b, c. Producer splits p_b dominantly (16) over p_a (2);
    # symbol map sends restickify "b" -> producer "p_b". So "b" must lead.
    restickify_dims = ["a", "b", "c"]
    producer_splits = {"p_a": 2, "p_b": 16}
    symbol_map = {"a": "p_a", "b": "p_b", "c": "p_c"}
    order, reason = rc.producer_aligned_dim_order(
        restickify_dims, producer_splits, symbol_map
    )
    assert reason is None
    assert order == ["b", "a", "c"]


def test_producer_aligned_dim_order_no_split():
    order, reason = rc.producer_aligned_dim_order(
        ["a", "b"], {"p_a": 1, "p_b": 1}, {"a": "p_a", "b": "p_b"}
    )
    assert order is None
    assert reason == "producer-has-no-mapped-split"


def test_producer_aligned_dim_order_ambiguous():
    # Two dims share the same dominant split factor => ambiguous.
    order, reason = rc.producer_aligned_dim_order(
        ["a", "b"], {"p_a": 4, "p_b": 4}, {"a": "p_a", "b": "p_b"}
    )
    assert order is None
    assert reason == "ambiguous-producer-split"


# --------------------------------------------------------------------------- #
# build_transfer_plan
# --------------------------------------------------------------------------- #
def test_build_transfer_plan_identity_is_all_local():
    # Same split, same mapping, identity symbol map => every transfer is
    # producer_core == consumer_core (hops 0), no remote movement.
    sizes = {"d0": 1024, "d1": 256}
    splits = {"d0": 4, "d1": 1}
    mapping = rc.materialize_default_core_mapping(["d0", "d1"], splits)
    symbol_map = {"d0": "d0", "d1": "d1"}
    transfers, summary = rc.build_transfer_plan(
        sizes, sizes, splits, splits, mapping, mapping, symbol_map, 32
    )

    assert summary["total_transfers"] == 4
    assert all(t["hops"] == 0 for t in transfers)
    assert all(t["src_core"] == t["dst_core"] for t in transfers)
    assert summary["remote_elements"] == 0
    assert summary["total_byte_hops"] == 0
    assert summary["max_hops"] == 0
    # All elements conserved and local.
    assert summary["local_elements"] == 1024 * 256


def test_build_transfer_plan_reversed_ownership_is_remote():
    # Consumer owns the d0 slices in reverse order relative to the producer.
    # That forces cross-core movement: nonzero remote elements and max_hops.
    sizes = {"d0": 1024, "d1": 256}
    splits = {"d0": 4, "d1": 1}
    producer_mapping = rc.materialize_default_core_mapping(["d0", "d1"], splits)
    # Reverse: core c owns d0 slice (3 - c).
    consumer_mapping = {
        str(c): {"d0": 3 - c, "d1": 0} for c in range(4)
    }
    symbol_map = {"d0": "d0", "d1": "d1"}
    transfers, summary = rc.build_transfer_plan(
        sizes,
        sizes,
        splits,
        splits,
        producer_mapping,
        consumer_mapping,
        symbol_map,
        32,
    )

    # Each producer core's d0 slice is owned by exactly one (different, except
    # the self-mapping middle when symmetric) consumer core.
    assert summary["total_transfers"] == 4
    assert summary["remote_elements"] > 0
    assert summary["max_hops"] > 0

    # Total elements conserved: every element of the tensor moves exactly once.
    total = summary["local_elements"] + summary["remote_elements"]
    assert total == 1024 * 256
    # Cross-check conservation against an independent product.
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

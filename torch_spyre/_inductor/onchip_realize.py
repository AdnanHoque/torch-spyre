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

"""Pure realization logic for the on-chip same-layout handoff (Tier 1).

This module is torch-free so it unit-tests in a bare worktree (mirrors the
restickify_cost.py split). Given an eligible same-stick edge -- the consumer's
sharding, the shared stick dim, iteration sizes, num_cores -- it produces a
structured ``OnChipRealization``: LX bases (non-overlapping, in-capacity), the
synthesized datadscs_/schedule/opFuncsUsed_ (via codegen.onchip_bridge), and
the producer/consumer LX-flip descriptors. Over-capacity or layout-changing
edges return None (fail-closed). The realize FIRST CUT targets the simplest
same-core same-shard case: a single STCDPOpLx, 2 LX regions, no ring.
"""

from __future__ import annotations

import dataclasses

from .codegen.onchip_bridge import (
    LX_CAPACITY_BYTES,
    allocate_lx_bases,
    build_same_layout_bridge,
    per_core_slice_bytes,
)

# Per-core LX byte span declared inside each data-op labeledDs (2 MB, matches
# splice_2048_stcdp DATAOP_LX_SIZE). DL-DSC LX size sentinel.
DATAOP_LX_SIZE = 2 << 20
DL_LX_SENTINEL = 2147483647


@dataclasses.dataclass(frozen=True)
class LxFlip:
    """Flip one DL labeledDs to LX-resident: ``base`` per core, sentinel size."""

    ldsidx: int
    lx_base: int
    role: str  # "producer-output" | "consumer-input"


@dataclasses.dataclass(frozen=True)
class OnChipRealization:
    """A realized same-layout handoff: LX bases + synthesized mixed-DSC parts."""

    producer_base: int
    consumer_base: int
    slice_bytes: int
    producer_flip: LxFlip
    consumer_flip: LxFlip
    datadscs: list
    opfuncs: list[str]
    schedule: dict
    realizable: bool = True


def realize_same_core_handoff(
    iter_sizes: dict[str, int],
    layout: list[str],
    stick_dim: str,
    split_dim: str,
    stick_size: int,
    num_cores: int,
    producer_ldsidx: int,
    consumer_ldsidx: int,
    capacity: int = LX_CAPACITY_BYTES,
) -> OnChipRealization | None:
    """Build a 2-region same-core single-STCDP realization, or None (fail-closed).

    Both producer-out and consumer-in split the SAME dim (same-shard), so a
    single STCDPOpLx copies each core's slice LX->LX with no ring. Bases are
    derived per-size and packed non-overlapping; if the 2 regions exceed
    per-core LX, return None. The fold is the simplest 18/40 case.
    """
    if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
        return None
    slice_bytes = per_core_slice_bytes(iter_sizes, split_dim, stick_size, num_cores)
    try:
        bases = allocate_lx_bases(2, slice_bytes, capacity=capacity)
    except ValueError:
        return None
    producer_base, consumer_base = bases
    datadscs, opfuncs, sched = build_same_layout_bridge(
        dim_pool=layout, iter_sizes=iter_sizes, stick_size=stick_size,
        num_cores=num_cores, lx_size=DATAOP_LX_SIZE,
        src_base=producer_base, dst_base=consumer_base, layout=layout,
        stick_dim=stick_dim, src_split_dim=split_dim, dst_split_dim=split_dim,
    )
    return OnChipRealization(
        producer_base=producer_base,
        consumer_base=consumer_base,
        slice_bytes=slice_bytes,
        producer_flip=LxFlip(producer_ldsidx, producer_base, "producer-output"),
        consumer_flip=LxFlip(consumer_ldsidx, consumer_base, "consumer-input"),
        datadscs=datadscs,
        opfuncs=opfuncs,
        schedule=sched,
    )


def is_same_shard(
    producer_splits: dict[str, int],
    consumer_splits: dict[str, int],
    symbol_map: dict[str, str],
) -> bool:
    """True when both sides split the same dim the same way (no ring needed)."""
    import math

    if math.prod(producer_splits.values()) != math.prod(consumer_splits.values()):
        return False
    for cons_sym, prod_sym in symbol_map.items():
        if consumer_splits.get(cons_sym, 1) != producer_splits.get(prod_sym, 1):
            return False
    return True

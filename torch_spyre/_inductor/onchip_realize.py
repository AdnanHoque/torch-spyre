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
import copy

from .codegen.onchip_bridge import (
    DATA_FORMAT,
    LX_CAPACITY_BYTES,
    STICK_BYTES,
    STREAM_TILE_BYTES,
    WORD_LENGTH,
    allocate_flash_attention_pipeline_bases,
    allocate_lx_bases,
    allocate_stream_bases,
    build_flash_attention_pipeline_bridge,
    build_flash_attention_pipeline_mixed_sdsc,
    flash_pipeline_overlap_prefix_schedule,
    mixed_schedule,
    build_roundtrip_bridge,
    build_same_layout_bridge,
    build_streamed_bridge,
    Endpoint,
    make_datadsc,
    num_stream_tiles,
    per_core_slice_bytes,
    per_core_same_stick_slice_bytes,
    _align_up,
    _stcdp_op,
)

# Per-core LX byte span declared inside each data-op labeledDs (2 MB, matches
# splice_2048_stcdp DATAOP_LX_SIZE). DL-DSC LX size sentinel.
DATAOP_LX_SIZE = 2 << 20
DL_LX_SENTINEL = 2147483647

# Device-proven LX address contract for the 2048 same-core add->add bridge
# (splice_2048_stcdp): producer output @16384, consumer input @8192.
PRODUCER_LX_BASE = 16384
CONSUMER_LX_BASE = 8192
STICK_SIZE = 64

# Tier select: stream once the single 2-region move would consume more than half
# the 2 MB LX, leaving no headroom for the DL op. Below it the single move fits.
STREAM_THRESHOLD = LX_CAPACITY_BYTES // 2
# Device-proven 512x512 add and seq64 attention splices reserve at least 256 KiB
# per LX bridge region. Tighter packing can overlap DL-op private LX scratch even
# when the logical tensor slice is smaller.
MIN_BRIDGE_REGION_BYTES = 256 << 10


def layout_xform_compose_pointwise_lx_base(layout_slice_bytes: int) -> int:
    # Keep composed pointwise handoffs above the layout pair's bridge footprint.
    return PRODUCER_LX_BASE + 2 * max(layout_slice_bytes, MIN_BRIDGE_REGION_BYTES)


LAYOUT_XFORM_COMPOSE_POINTWISE_LX_BASE = layout_xform_compose_pointwise_lx_base(
    MIN_BRIDGE_REGION_BYTES
)
# Stage022 device sweep: score-scale PT->SFP handoff is value-correct through
# 128-wide score blocks, but 256-wide score blocks corrupt values. Keep larger
# blocks fail-closed to the HBM score-scale path while retaining later SFP
# pointwise handoffs.
FLASH_SCORE_SCALE_MAX_STICK_ELEMS = 128
INPUT_FETCH_NEIGHBOR_DISALLOWED_PINS = {"hbm", None}
INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX = 0
LAYOUT_XFORM_PAIR_AUTO_TILE = -2
LEGACY_DATA_STRUCT_DIM_KEYS = (
    "in_",
    "out_",
    "mb_",
    "i_",
    "j_",
    "ki_",
    "kj_",
    "x_",
    "x1_",
    "y_",
    "r_",
    "c_",
    "ij_",
    "rc_",
    "kij_",
    "sij_",
    "zij_",
    "si_",
    "sj_",
    "zi_",
    "zj_",
)
PINNED_COMPONENT_ORDER = (
    "hbm",
    "ring",
    "sfpring",
    "lx",
    "pt",
    "ptxrf",
    "ptarf",
    "sfplrf",
    "pelrf",
    "l0",
    "ptirf",
)


def _reserve_bridge_region_bytes(slice_bytes: int) -> int:
    return max(slice_bytes, MIN_BRIDGE_REGION_BYTES)


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
    region0: int = 0,
) -> OnChipRealization | None:
    """Build a 2-region same-core single-STCDP realization, or None (fail-closed).

    Both producer-out and consumer-in split the SAME dim (same-shard), so a
    single STCDPOpLx copies each core's slice LX->LX with no ring. Bases are
    derived per-size and packed non-overlapping; if the 2 regions exceed
    per-core LX, return None. The fold is the simplest 18/40 case.
    """
    if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
        return None
    slice_bytes = _reserve_bridge_region_bytes(
        per_core_slice_bytes(iter_sizes, split_dim, stick_size, num_cores)
    )
    try:
        bases = allocate_lx_bases(2, slice_bytes, capacity=capacity, region0=region0)
    except ValueError:
        return None
    producer_base, consumer_base = bases
    datadscs, opfuncs, sched = build_same_layout_bridge(
        dim_pool=layout,
        iter_sizes=iter_sizes,
        stick_size=stick_size,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
        src_base=producer_base,
        dst_base=consumer_base,
        layout=layout,
        stick_dim=stick_dim,
        src_split_dim=split_dim,
        dst_split_dim=split_dim,
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


def realize_same_layout_handoff(
    iter_sizes: dict[str, int],
    layout: list[str],
    stick_dim: str,
    split_dim: str,
    stick_size: int,
    num_cores: int,
    producer_ldsidx: int,
    consumer_ldsidx: int,
    capacity: int = LX_CAPACITY_BYTES,
    region0: int = 0,
) -> OnChipRealization | None:
    """Build a same-layout STCDP handoff when split and stick can differ."""
    if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
        return None
    slice_bytes = _reserve_bridge_region_bytes(
        per_core_same_stick_slice_bytes(
            iter_sizes,
            split_dim,
            stick_dim,
            stick_size,
            num_cores,
        )
    )
    try:
        bases = allocate_lx_bases(2, slice_bytes, capacity=capacity, region0=region0)
    except ValueError:
        return None
    producer_base, consumer_base = bases
    datadscs, opfuncs, sched = build_same_layout_bridge(
        dim_pool=layout,
        iter_sizes=iter_sizes,
        stick_size=stick_size,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
        src_base=producer_base,
        dst_base=consumer_base,
        layout=layout,
        stick_dim=stick_dim,
        src_split_dim=split_dim,
        dst_split_dim=split_dim,
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


@dataclasses.dataclass(frozen=True)
class StreamedRealization:
    """A realized streamed handoff: 2 fixed tile bases + K-tile mixed-DSC parts."""

    producer_base: int
    consumer_base: int
    slice_bytes: int
    num_tiles: int
    tile_bytes: int
    producer_flip: LxFlip
    consumer_flip: LxFlip
    datadscs: list
    opfuncs: list[str]
    schedule: dict
    realizable: bool = True


def realize_streamed_handoff(
    iter_sizes: dict[str, int],
    layout: list[str],
    stick_dim: str,
    split_dim: str,
    stick_size: int,
    num_cores: int,
    producer_ldsidx: int,
    consumer_ldsidx: int,
    tile_bytes: int = STREAM_TILE_BYTES,
    capacity: int = LX_CAPACITY_BYTES,
    region0: int = 0,
) -> StreamedRealization | None:
    """Stream a >LX/2 slice through 2 fixed tile buffers, or None (fail-closed).

    The single move stays the same-shard same-stick cross-core copy; streaming
    just tiles it along the non-split row dim so only 2*tile_bytes of LX live at
    once (vs 2*slice). K = ceil(slice/tile). Returns None if even the 2 fixed
    tiles don't fit -- the DL op gets the rest of the 2 MB. Single-buffer reuse:
    device-validate (fallback = double-buffer).
    """
    if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
        return None
    row_dims = [d for d in layout if d != split_dim]
    if len(row_dims) != 1:
        return None
    row_dim = row_dims[0]
    slice_bytes = _reserve_bridge_region_bytes(
        per_core_slice_bytes(iter_sizes, split_dim, stick_size, num_cores)
    )
    try:
        bases = allocate_stream_bases(
            tile_bytes,
            capacity=capacity,
            region0=region0,
        )
    except ValueError:
        return None
    producer_base, consumer_base = bases
    k_tiles = num_stream_tiles(slice_bytes, tile_bytes)
    datadscs, opfuncs, sched = build_streamed_bridge(
        dim_pool=layout,
        iter_sizes=iter_sizes,
        stick_size=stick_size,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
        src_base=producer_base,
        dst_base=consumer_base,
        layout=layout,
        stick_dim=stick_dim,
        src_split_dim=split_dim,
        dst_split_dim=split_dim,
        row_dim=row_dim,
        slice_bytes=slice_bytes,
        tile_bytes=tile_bytes,
    )
    return StreamedRealization(
        producer_base=producer_base,
        consumer_base=consumer_base,
        slice_bytes=slice_bytes,
        num_tiles=k_tiles,
        tile_bytes=tile_bytes,
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


@dataclasses.dataclass(frozen=True)
class ReductionReshardRealization:
    """A realized 2-D core-to-core reduction reshard (the genuine ring move).

    Unlike :class:`OnChipRealization` (same-shard, src_split_dim==dst_split_dim),
    this carries a producer that co-splits two dims (``{mb, out}``) feeding a
    consumer that reduces over the producer's stick/split dim (``K`` not split):
    the SwiGLU ``mul -> down_proj`` edge. The datadscs/schedule come from the
    :mod:`~torch_spyre._inductor.reshard` substrate (the CPU-proven 2-D
    ``STCDPOpLx`` authoring); ``standalone_sdsc`` is the pure-data-op SDSC the
    splice writes as its own ``sdsc_execute`` step (avoiding the mixed-fold
    ``SdscTree.cpp:152`` reject).
    """

    producer_base: int
    consumer_base: int
    slice_bytes: int
    producer_flip: LxFlip
    consumer_flip: LxFlip
    datadscs: list
    opfuncs: list[str]
    schedule: dict
    standalone_sdsc: dict
    perband: bool
    realizable: bool = True


def realize_reduction_reshard(
    iter_sizes: dict[str, int],
    layout: list[str],
    row_dim: str,
    stick_dim: str,
    producer_splits: dict[str, int],
    consumer_splits: dict[str, int],
    stick_size: int,
    num_cores: int,
    producer_ldsidx: int,
    consumer_ldsidx: int,
    perband: bool = False,
    capacity: int = LX_CAPACITY_BYTES,
    region0: int = 0,
) -> ReductionReshardRealization | None:
    """Realize the 2-D ``mul -> down_proj`` reduction reshard, or None (fail-closed).

    The genuine non-co-assignable edge flash-ws fail-closes on: the producer
    co-splits ``{mb:m_split, out:n_split}`` (its stick/``out`` dim IS the
    consumer's reduction ``K``), and the consumer mb-bands ``{mb:num_cores}`` and
    reduces over the full ``K`` (not split). The move is LX -> RIU ring -> LX.

    Delegates the device-program synthesis to the ``reshard`` substrate (the
    CPU-proven 2-D ``STCDPOpLx`` authoring) rather than the same-shard
    ``build_same_layout_bridge`` (which hardcodes ``src_split_dim==dst_split_dim``
    and cannot express the orthogonal reshard). Fail-closes (returns ``None``)
    on any shape it cannot map exactly: a non-2-D producer co-split, a consumer
    that is not a single ``mb``-banded reduction, an uneven band split, or a
    per-core LX footprint that overflows ``capacity``.
    """
    from .reshard import (
        Band,
        Piece,
        allocate_lx_bases,
        build_asymmetric_reshard_bridge,
        build_consumer_pieces,
        build_perband_reshard_bridge,
        build_producer_pieces,
        build_standalone_dataop_sdsc,
    )

    row_sym = row_dim
    stick_sym = stick_dim
    if row_sym not in iter_sizes or stick_sym not in iter_sizes:
        return None
    m_split = int(producer_splits.get(row_sym, 1))
    n_split = int(producer_splits.get(stick_sym, 1))
    cons_m_split = int(consumer_splits.get(row_sym, 1))
    cons_n_split = int(consumer_splits.get(stick_sym, 1))
    if m_split < 1 or n_split < 2:
        # Not a co-split producer over the reduction (stick) dim.
        return None
    if cons_n_split != 1:
        # The consumer must NOT split the dim it reduces over (K resident whole).
        return None
    if m_split * n_split != num_cores or cons_m_split != num_cores:
        return None

    m_rows = int(iter_sizes[row_sym])
    k_extent = int(iter_sizes[stick_sym])
    if m_rows % m_split or m_rows % cons_m_split or k_extent % n_split:
        return None

    # Producer owner = mb + m_split*out (the {mb:4,out:8} co-split, pinned).
    def _producer_owner(mb_band: int, out_band: int) -> int:
        return mb_band + m_split * out_band

    def _consumer_owner(mb_band: int, out_band: int) -> int:
        return mb_band

    # Per-core LX footprint: producer tile + consumer band, two regions.
    producer_tile_bytes = (m_rows // m_split) * (k_extent // n_split) * WORD_LENGTH
    consumer_band_bytes = (m_rows // cons_m_split) * k_extent * WORD_LENGTH
    slice_bytes = max(producer_tile_bytes, consumer_band_bytes)
    try:
        producer_base, consumer_base = allocate_lx_bases(
            2, slice_bytes, capacity=capacity, region0=region0
        )
    except ValueError:
        return None

    bridge_iter = {row_sym: m_rows, stick_sym: k_extent}
    try:
        if perband:
            col_step = k_extent // n_split
            row_step = m_rows // m_split
            cons_row_step = m_rows // cons_m_split
            edges: list[tuple[list[Piece], list[Piece]]] = []
            for b in range(n_split):
                producer = [
                    Piece(
                        key=f"p{mb + 1}",
                        owner=_producer_owner(mb, b),
                        rows=Band(mb * row_step, row_step),
                        cols=Band(b * col_step, col_step),
                    )
                    for mb in range(m_split)
                ]
                consumer = [
                    Piece(
                        key=f"p{c + 1}",
                        owner=_consumer_owner(c, 0),
                        rows=Band(c * cons_row_step, cons_row_step),
                        cols=Band(b * col_step, col_step),
                    )
                    for c in range(cons_m_split)
                ]
                edges.append((producer, consumer))
            datadscs, opfuncs, schedule = build_perband_reshard_bridge(
                edges,
                dim_pool=layout,
                iter_sizes=bridge_iter,
                stick_size=stick_size,
                num_cores=num_cores,
                lx_size=DATAOP_LX_SIZE,
                src_base=producer_base,
                dst_base=consumer_base,
                layout=layout,
                row_dim=row_sym,
                stick_dim=stick_sym,
            )
        else:
            producer_pieces = build_producer_pieces(
                m_rows, k_extent, m_split, n_split, _producer_owner
            )
            consumer_pieces = build_consumer_pieces(
                m_rows, k_extent, cons_m_split, 1, _consumer_owner
            )
            datadscs, opfuncs, schedule = build_asymmetric_reshard_bridge(
                dim_pool=layout,
                iter_sizes=bridge_iter,
                stick_size=stick_size,
                num_cores=num_cores,
                lx_size=DATAOP_LX_SIZE,
                src_base=producer_base,
                dst_base=consumer_base,
                layout=layout,
                row_dim=row_sym,
                stick_dim=stick_sym,
                producer_pieces=producer_pieces,
                consumer_pieces=consumer_pieces,
            )
    except ValueError:
        # Owner-out-of-range / uneven band -- fail closed.
        return None

    standalone_sdsc = build_standalone_dataop_sdsc(
        "1b_STCDP_reshard", datadscs, opfuncs, num_cores
    )
    return ReductionReshardRealization(
        producer_base=producer_base,
        consumer_base=consumer_base,
        slice_bytes=slice_bytes,
        producer_flip=LxFlip(producer_ldsidx, producer_base, "producer-output"),
        consumer_flip=LxFlip(consumer_ldsidx, consumer_base, "consumer-input"),
        datadscs=datadscs,
        opfuncs=opfuncs,
        schedule=schedule,
        standalone_sdsc=standalone_sdsc,
        perband=perband,
    )


# --- In-memory SDSC transform: emit the mixed bundle DURING compilation. The
# functions below port splice_2048_stcdp into the codegen path; they are pure
# dict surgery (torch-free) so generate_bundle and the offline gate share them.


_POINTWISE_HANDOFF_OPS = {
    "add",
    "exp",
    "identity",
    "maximum",
    "mul",
    "realdiv",
    "sub",
}


def _body(sdsc_json: dict) -> dict:
    return sdsc_json[next(iter(sdsc_json))]


def _dl_op(sdsc_json: dict) -> dict:
    """Return the single DL op dict of an SDSC body's first dsc."""
    dsc = _body(sdsc_json)["dscs_"][0]
    return dsc[next(iter(dsc))]


def _op_name(sdsc_json: dict) -> str:
    dsc = _body(sdsc_json)["dscs_"][0]
    return next(iter(dsc))


def _symbol_dim(dim: str) -> str:
    return dim if dim.endswith("_") else f"{dim}_"


def _lds_by_idx(dl: dict, lds_idx: int) -> dict | None:
    for lds in dl.get("labeledDs_", []):
        if lds.get("ldsIdx_") == lds_idx:
            return lds
    return None


def _primary_ds_info(dl: dict, lds_idx: int) -> dict:
    lds = _lds_by_idx(dl, lds_idx)
    if lds is None:
        return {}
    role = lds.get("dsType_")
    return dl.get("primaryDsInfo_", {}).get(role, {})


def _stick_dim_for_lds(dl: dict, lds_idx: int) -> str | None:
    sticks = _primary_ds_info(dl, lds_idx).get("stickDimOrder_", [])
    if len(sticks) != 1:
        return None
    return _symbol_dim(sticks[0])


def _layout_for_lds(dl: dict, lds_idx: int) -> list[str] | None:
    layout = _primary_ds_info(dl, lds_idx).get("layoutDimOrder_", [])
    if not layout:
        return None
    return [_symbol_dim(d) for d in layout]


def _single_split_dim(shard: dict[str, int]) -> str | None:
    split = [d for d, v in shard.items() if v > 1]
    if len(split) != 1:
        return None
    return _symbol_dim(split[0])


def _split_dims(shard: dict[str, int]) -> list[str]:
    return [_symbol_dim(dim) for dim, factor in shard.items() if int(factor) > 1]


def _iter_sizes_for_layout(dl: dict, layout: list[str]) -> dict[str, int] | None:
    sizes = dl.get("N_", {})
    out: dict[str, int] = {}
    for dim in layout:
        if dim not in sizes:
            return None
        out[dim] = int(sizes[dim])
    return out


def _local_iter_sizes_for_layout(
    dl: dict,
    layout: list[str],
) -> dict[str, int] | None:
    """Per-core staged tensor sizes for a DL operand layout."""
    out = _iter_sizes_for_layout(dl, layout)
    if out is None:
        return None
    stage = next(iter(dl.get("dataStageParam_", {}).values()), {})
    core_sizes = stage.get("ss_", {})
    for dim in layout:
        if dim in core_sizes:
            out[dim] = int(core_sizes[dim])
    return out


def _dim_size(dl: dict, dim: str) -> int | None:
    value = dl.get("N_", {}).get(dim)
    return int(value) if value is not None else None


def _split_factor(shard: dict[str, int], split_dim: str) -> int:
    return int(shard.get(split_dim.removesuffix("_"), 1))


def _same_shard_on_layout(
    producer_shard: dict[str, int],
    consumer_shard: dict[str, int],
    layout: list[str],
) -> bool:
    layout_dims = {dim.removesuffix("_") for dim in layout}
    for dim in layout_dims:
        if int(producer_shard.get(dim, 1)) != int(consumer_shard.get(dim, 1)):
            return False
    for dim, factor in producer_shard.items():
        if dim not in layout_dims and int(factor) != 1:
            return False
    for dim, factor in consumer_shard.items():
        if dim not in layout_dims and int(factor) != 1:
            return False
    return True


def _same_physical_stick_layout(
    prod_dl: dict,
    prod_layout: list[str],
    prod_stick: str,
    cons_dl: dict,
    cons_layout: list[str],
    cons_stick: str,
) -> bool:
    """True when producer/consumer layouts differ only by stick dim naming.

    Matmul producer outputs name the hidden stick axis ``out_`` while a following
    matmul consumer names that same physical axis ``in_``.  Treat that as
    same-stick only when the stick appears in the same layout position, all
    non-stick dims have the same names, and paired extents match.
    """
    if len(prod_layout) != len(cons_layout):
        return False
    for p_dim, c_dim in zip(prod_layout, cons_layout):
        if p_dim == prod_stick and c_dim == cons_stick:
            pass
        elif p_dim != c_dim:
            return False
        p_size = _dim_size(prod_dl, p_dim)
        c_size = _dim_size(cons_dl, c_dim)
        if p_size is None or c_size is None or p_size != c_size:
            return False
    return True


def _physical_layout_signature(
    dl: dict,
    layout: list[str],
    stick_dim: str,
) -> list[tuple[str, int]] | None:
    signature: list[tuple[str, int]] = []
    for dim in layout:
        key = "<stick>" if dim == stick_dim else dim
        size = _dim_size(dl, dim)
        if size is None:
            return None
        signature.append((key, size))
    return sorted(signature)


def _layout_mismatch_reason(
    prefix: str,
    prod_dl: dict,
    prod_layout: list[str],
    prod_stick: str,
    cons_dl: dict,
    cons_layout: list[str],
    cons_stick: str,
) -> str | None:
    if _same_physical_stick_layout(
        prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
    ):
        return None
    prod_sig = _physical_layout_signature(prod_dl, prod_layout, prod_stick)
    cons_sig = _physical_layout_signature(cons_dl, cons_layout, cons_stick)
    reason = "physical_layout_mismatch"
    if prod_sig is not None and prod_sig == cons_sig:
        reason = "layout_transform_required"
    elif _layout_transform_dim_map(
        prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
    ) is not None:
        reason = "layout_transform_required"
    return (
        f"{prefix}:{reason}:"
        f"producer={prod_layout}/{prod_stick}:"
        f"consumer={cons_layout}/{cons_stick}"
    )


def _stick_aliased_layout(
    layout: list[str],
    old_stick: str,
    new_stick: str,
) -> list[str]:
    return [new_stick if dim == old_stick else dim for dim in layout]


def _layout_transform_dim_map(
    prod_dl: dict,
    prod_layout: list[str],
    prod_stick: str,
    cons_dl: dict,
    cons_layout: list[str],
    cons_stick: str,
) -> dict[str, str] | None:
    source_layout = _stick_aliased_layout(prod_layout, prod_stick, cons_stick)
    if set(source_layout) == set(cons_layout):
        out: dict[str, str] = {}
        for p_dim, c_dim in zip(prod_layout, source_layout):
            p_size = _dim_size(prod_dl, p_dim)
            c_size = _dim_size(cons_dl, c_dim)
            if p_size is None or c_size is None or p_size != c_size:
                break
            out[p_dim] = c_dim
        else:
            return out

    if len(prod_layout) != len(cons_layout):
        return None
    out = {}
    seen = set()
    for p_dim, c_dim in zip(prod_layout, cons_layout):
        if c_dim in seen:
            return None
        p_size = _dim_size(prod_dl, p_dim)
        c_size = _dim_size(cons_dl, c_dim)
        if p_size is None or c_size is None or p_size != c_size:
            return None
        out[p_dim] = c_dim
        seen.add(c_dim)
    return out


def _mapped_work_slice_piece_info(
    body: dict,
    producer_layout: list[str],
    dim_map: dict[str, str],
    iter_sizes: dict[str, int],
    base: int,
) -> list[dict] | None:
    shard = body.get("numWkSlicesPerDim_", {})
    core_slices = body.get("coreIdToWkSlice_", {})
    if not isinstance(core_slices, dict) or not core_slices:
        return None
    pieces = []
    for ordinal, core_id_str in enumerate(sorted(core_slices, key=lambda k: int(k))):
        wk_slice = core_slices[core_id_str]
        start = {target_dim: 0 for target_dim in dim_map.values()}
        size = {target_dim: iter_sizes[target_dim] for target_dim in dim_map.values()}
        for prod_dim in producer_layout:
            target_dim = dim_map.get(prod_dim)
            if target_dim is None or target_dim not in iter_sizes:
                return None
            factor = int(shard.get(prod_dim.removesuffix("_"), 1))
            if factor <= 0 or iter_sizes[target_dim] % factor != 0:
                return None
            coord = int(wk_slice.get(prod_dim.removesuffix("_"), 0))
            if coord < 0 or coord >= factor:
                return None
            chunk = iter_sizes[target_dim] // factor
            start[target_dim] = coord * chunk
            size[target_dim] = chunk
        pieces.append(
            {
                "key_": f"p{ordinal + 1}",
                "dimToStartCordinate": start,
                "dimToSize_": size,
                "validGap_": {dim: [[size[dim], 0]] for dim in size},
                "PlacementInfo": [
                    {
                        "type": "lx",
                        "memId": [int(core_id_str)],
                        "startAddr": [base],
                    }
                ],
            }
        )
    return pieces


def _handoff_bytes(iter_sizes: dict[str, int], word_length: int) -> int:
    size = word_length
    for n in iter_sizes.values():
        size *= n
    return size


def _memorg_is_present(mem_org: dict, component: str) -> bool:
    entry = mem_org.get(component) or mem_org.get(component.upper())
    if not isinstance(entry, dict):
        return False
    return bool(entry.get("isPresent", 0))


def _pinned_component(lds: dict) -> str | None:
    """Return Foundation's first present pinned component for a labeledDs."""
    mem_org = lds.get("memOrg_", {})
    if not isinstance(mem_org, dict):
        return None
    for component in PINNED_COMPONENT_ORDER:
        if _memorg_is_present(mem_org, component):
            return component
    return None


def _input_fetch_neighbor_rejection_reasons(compute_dsc: dict) -> list[str]:
    """Return DXP InputFetchNeighbor contract gaps for a compute DSC."""
    reasons: list[str] = []
    if not isinstance(compute_dsc, dict) or not compute_dsc:
        return ["invalid_compute_dsc"]
    dl = next(iter(compute_dsc.values()))
    if not isinstance(dl, dict):
        return ["invalid_compute_dl"]
    labeled_ds = dl.get("labeledDs_", [])
    if not labeled_ds:
        return ["missing_labeled_ds"]
    for lds in labeled_ds:
        pinned = _pinned_component(lds)
        if pinned in INPUT_FETCH_NEIGHBOR_DISALLOWED_PINS:
            idx = lds.get("ldsIdx_", "?")
            reasons.append(f"lds{idx}_pinned_{pinned or 'none'}")

    compute_ops = dl.get("computeOp_", [])
    if not compute_ops:
        reasons.append("missing_compute_op")
        return reasons
    input_labels = compute_ops[0].get("inputLabeledDs", [])
    if not input_labels:
        reasons.append("missing_input_labeled_ds")
        return reasons
    first_input_idx = int(input_labels[0].rsplit("-idx", 1)[1])
    if first_input_idx != INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX:
        reasons.append(f"first_input_not_lds{INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX}")
    input_lds = next(
        (lds for lds in labeled_ds if lds.get("ldsIdx_") == first_input_idx),
        None,
    )
    if input_lds is None:
        reasons.append(f"missing_input_lds{first_input_idx}")
    elif _pinned_component(input_lds) != "lx":
        reasons.append(
            f"input_lds{first_input_idx}_pinned_"
            f"{_pinned_component(input_lds) or 'none'}"
        )
    if not _input_fetch_neighbor_ij_order_supported(dl, first_input_idx):
        reasons.append("input_layout_missing_i_j")
    if not _has_input_fetch_neighbor_transfer(dl, first_input_idx):
        reasons.append(
            f"missing_no_component_to_lx_transfer_lds{first_input_idx}"
        )
    return reasons


def _input_fetch_neighbor_compute_eligible(compute_dsc: dict) -> bool:
    """True when a compute DSC satisfies DXP's InputFetchNeighbor contract."""
    return not _input_fetch_neighbor_rejection_reasons(compute_dsc)


def _input_fetch_neighbor_ij_order_supported(dl: dict, lds_idx: int) -> bool:
    """Current Foundation InputFetchNeighbor ordering assumes i/j coordinates."""
    layout = _layout_for_lds(dl, lds_idx)
    if layout is None:
        return False
    return {"i_", "j_"}.issubset(set(layout))


def _is_input_fetch_neighbor_transfer_node(node: dict, lds_idx: int) -> bool:
    """DXP later expects a NO_COMPONENT -> LX transfer node for the neighbor."""
    if node.get("nodeType_") != "transfer":
        return False
    src = node.get("src_", {})
    if src.get("unit_") != "no_component":
        return False
    if src.get("storage_") != "no_component":
        return False
    if not any(
        (via.get("loc_", {}) or {}).get("unit_") == "no_component"
        and (via.get("loc_", {}) or {}).get("storage_") == "lx"
        for via in node.get("dstVias_", [])
    ):
        return False
    return any(
        dst.get("myLdsIdx_") == lds_idx
        for dst in node.get("dstLdsAndLoopOffsets_", [])
    )


def _has_input_fetch_neighbor_transfer(dl: dict, lds_idx: int) -> bool:
    """True when the DSC has the NO_COMPONENT -> LX neighbor marker."""
    return any(
        _is_input_fetch_neighbor_transfer_node(node, lds_idx)
        for node in dl.get("scheduleTree_", [])
    )


def _input_fetch_neighbor_transfer_offsets(lds_idx: int) -> dict:
    return {
        "myLdsIdx_": lds_idx,
        "startAddr_": "0",
        "isStartAddrSymbolic_": 0,
        "latchDataId_": -1,
        "constantId_": -1,
        "constEleOffsets_": {},
        "loopEleOffsets_": {},
        "bufferAddrOffset_": {},
        "bufferSwitchPosition_": "",
        "dataConnect_": "",
    }


def _normalize_input_fetch_neighbor_transfer(node: dict, lds_idx: int) -> None:
    node.setdefault("name_", f"input_fetch_neighbor_transfer_lds{lds_idx}")
    node["prev_"] = node.get("prev_") if isinstance(node.get("prev_"), str) else ""
    node.setdefault("relevantComps_", {})
    node.setdefault(
        "src_",
        {
            "unit_": "no_component",
            "storage_": "no_component",
        },
    )
    node.setdefault("srcLdsAndLoopOffsets_", _input_fetch_neighbor_transfer_offsets(-1))
    node.setdefault("dstVias_", [])
    if not node["dstVias_"]:
        node["dstVias_"].append(
            {
                "loc_": {
                    "unit_": "no_component",
                    "storage_": "lx",
                },
                "via_": [],
            }
        )
    node.setdefault("dstLdsAndLoopOffsets_", [])
    for dst in node["dstLdsAndLoopOffsets_"]:
        if dst.get("myLdsIdx_") == lds_idx:
            dst.update(
                {
                    key: value
                    for key, value in _input_fetch_neighbor_transfer_offsets(
                        lds_idx
                    ).items()
                    if key not in dst
                }
            )
            break
    else:
        node["dstLdsAndLoopOffsets_"].append(
            _input_fetch_neighbor_transfer_offsets(lds_idx)
        )
    node.setdefault("lastFusableParentLoopSrc_", "")
    node.setdefault("lastFusableParentLoopDst_", [])
    node.setdefault("replicationFactor_", 1)
    node.setdefault("unitTimeTransferChunkSize_", [])
    node.setdefault("unitTimeTransferNumChunks_", 1)
    node.setdefault("unitTimeTransferChunkStride_", [])
    node.setdefault("rotateNumElements_", 0)
    node.setdefault("coreIdToGTRInfo_", {})
    node.setdefault("transferSize_", {})
    node.setdefault("coreletViews_", {})


def _link_input_fetch_neighbor_allocate(dl: dict, lds_idx: int, transfer_name: str) -> None:
    for node in dl.get("scheduleTree_", []):
        if (
            node.get("nodeType_") == "allocate"
            and node.get("ldsIdx_") == lds_idx
            and node.get("component_") == "lx"
        ):
            node.setdefault("allocUsers_", {})[transfer_name] = 1
            return


def _legacy_data_struct_dims(name: str, values: dict[str, int]) -> dict:
    dims = {
        "name_": name,
        **{key: -1 for key in LEGACY_DATA_STRUCT_DIM_KEYS},
        "symbolicDimInfo_": {},
        "maxSymbolicVolume_": {},
        "coreletSplit_": {},
        "rowSplit_": {},
        "peSfpSplit_": {},
        "paddingSizes_": {},
    }
    for key, value in values.items():
        if key in LEGACY_DATA_STRUCT_DIM_KEYS:
            dims[key] = int(value)
    return dims


def _core_data_stage_dims(dl: dict, shard: dict | None = None) -> dict[str, int]:
    core_stage = (
        dl.get("dataStageParam_", {})
        .get("0", {})
        .get("ss_", {})
    )
    dims = {
        key: int(value)
        for key, value in core_stage.items()
        if key in LEGACY_DATA_STRUCT_DIM_KEYS
    }
    if dims:
        return dims
    sizes = dl.get("N_", {})
    shard = shard or {}
    for dim, size in sizes.items():
        if dim not in LEGACY_DATA_STRUCT_DIM_KEYS:
            continue
        split_factor = int(shard.get(dim.rstrip("_"), 1) or 1)
        dims[dim] = int(size) // split_factor
    return dims


def _add_input_fetch_neighbor_legacy_dims(dl: dict, shard: dict | None = None) -> None:
    dims = _core_data_stage_dims(dl, shard)
    if not dims:
        return
    dl.setdefault("CoreD_", _legacy_data_struct_dims("d", dims))
    dl.setdefault("CoreletD_", _legacy_data_struct_dims("coreletd", dims))
    dl.setdefault("B_", _legacy_data_struct_dims("b", dims))


def _add_input_fetch_neighbor_transfer(dl: dict, lds_idx: int) -> None:
    """Add the minimal NO_COMPONENT -> LX transfer marker for IFN lowering."""
    schedule_tree = dl.setdefault("scheduleTree_", [])
    for node in schedule_tree:
        if _is_input_fetch_neighbor_transfer_node(node, lds_idx):
            _normalize_input_fetch_neighbor_transfer(node, lds_idx)
            _link_input_fetch_neighbor_allocate(dl, lds_idx, node["name_"])
            return
    transfer_name = f"input_fetch_neighbor_transfer_lds{lds_idx}"
    transfer = {
        "nodeType_": "transfer",
        "name_": transfer_name,
        "src_": {
            "unit_": "no_component",
            "storage_": "no_component",
        },
        "dstVias_": [
            {
                "loc_": {
                    "unit_": "no_component",
                    "storage_": "lx",
                },
                "via_": [],
            }
        ],
        "dstLdsAndLoopOffsets_": [
            _input_fetch_neighbor_transfer_offsets(lds_idx)
        ],
    }
    _normalize_input_fetch_neighbor_transfer(transfer, lds_idx)
    schedule_tree.append(
        transfer
    )
    _link_input_fetch_neighbor_allocate(dl, lds_idx, transfer_name)


def _core_state_init_entry(lx_base: int) -> dict:
    return {
        "ebrInit_": -1,
        "gtr_": {
            "type": "multicast",
            "id": 18446744073709551615,
            "count": 0,
            "sharers": 0,
            "groupInfo_": {},
        },
        "condGtr_": [],
        "lbrInit_": [lx_base],
        "gapPerDim_": {},
        "lxSizeWithGaps_": DL_LX_SENTINEL,
        "lbrInitForwardGap_": 0,
    }


def apply_lx_flip(
    sdsc_json: dict,
    flip: LxFlip,
    *,
    core_state_init: bool = True,
    num_corelets: int | None = None,
) -> None:
    """Flip the DL labeledDs at ``flip.ldsidx`` to LX-resident @ ``flip.lx_base``.

    Mirrors splice_2048_stcdp._flip_tensor_to_lx exactly: rewrites the labeledDs
    (memOrg_ -> lx, hbm addr/size cleared, lx size sentinel, per-core
    coreStateInit_) and its scheduleTree allocate node (name/component/per-core
    LX base). In place.
    """
    dl = _dl_op(sdsc_json)
    lds = next(e for e in dl["labeledDs_"] if e["ldsIdx_"] == flip.ldsidx)
    alloc_node = f"allocate-{lds['dsName_']}_lx"
    num_cores = dl["numCoresUsed_"]
    if num_corelets is not None:
        dl["numCoreletsUsed_"] = num_corelets
        dl["numCoreletsUsed_DSC2_"] = num_corelets
    node = next(
        n
        for n in dl["scheduleTree_"]
        if n.get("nodeType_") == "allocate" and n.get("ldsIdx_") == flip.ldsidx
    )
    node["name_"] = alloc_node
    node["component_"] = "lx"
    node["startAddressCoreCorelet_"]["data_"] = {
        f"[{c}, 0, 0]": str(flip.lx_base) for c in range(num_cores)
    }
    # HBM allocate nodes may carry global back-gap metadata for strided tile
    # views.  Deeptools only accepts those gaps on HBM allocations; after an
    # explicit prefill into LX the DL consumer should see a packed LX tile.
    node.pop("backGapCore_", None)
    node.pop("gapStickSpread_", None)
    lds["memOrg_"] = {"lx": {"isPresent": 1, "allocateNode_": alloc_node}}
    lds["hbmStartAddress_"] = -1
    lds["hbmSize_"] = 0
    lds["lxSize_"] = DL_LX_SENTINEL
    lds["lxBufferSize_"] = DL_LX_SENTINEL
    if core_state_init:
        lds["coreStateInit_"] = [
            _core_state_init_entry(flip.lx_base) for _ in range(num_cores)
        ]
    else:
        lds.pop("coreStateInit_", None)


def _retarget_lx_allocate_style(
    sdsc_json: dict,
    lds_idx: int,
    style: str,
) -> None:
    if not style:
        return
    if style not in {"canonical_name", "canonical_loop"}:
        raise ValueError(f"unsupported LX allocation style: {style}")

    dl = _dl_op(sdsc_json)
    lds = _lds_by_idx(dl, lds_idx)
    if lds is None:
        return

    old_name = lds.get("memOrg_", {}).get("lx", {}).get("allocateNode_")
    new_name = f"allocate_lds{lds_idx}_lx"
    lds.setdefault("memOrg_", {}).setdefault("lx", {})["allocateNode_"] = new_name

    alloc_node = None
    for node in dl.get("scheduleTree_", []):
        if old_name is not None and node.get("name_") == old_name:
            alloc_node = node
            break
    if alloc_node is None:
        alloc_node = next(
            (
                node
                for node in dl.get("scheduleTree_", [])
                if node.get("nodeType_") == "allocate"
                and node.get("ldsIdx_") == lds_idx
                and node.get("component_") == "lx"
            ),
            None,
        )
    if alloc_node is None:
        return

    alloc_node["name_"] = new_name
    for node in dl.get("scheduleTree_", []):
        if node.get("prev_") == old_name:
            node["prev_"] = new_name
        if isinstance(node.get("next_"), list):
            node["next_"] = [
                new_name if child == old_name else child for child in node["next_"]
            ]

    if style != "canonical_loop":
        return

    loop_node = next(
        (
            node
            for node in dl.get("scheduleTree_", [])
            if node.get("nodeType_") == "loop" and node.get("name_") == "loop_ds0_ds1_in"
        ),
        None,
    )
    if loop_node is None:
        return

    alloc_node["prev_"] = loop_node["name_"]
    alloc_node["numBuffers_"] = 2
    next_nodes = list(loop_node.get("next_", []))
    if new_name not in next_nodes:
        try:
            insert_at = next_nodes.index("lx_below_schedule")
        except ValueError:
            insert_at = len(next_nodes)
        next_nodes.insert(insert_at, new_name)
    loop_node["next_"] = next_nodes


def _hbm_base(dl: dict, lds_idx: int) -> str | None:
    """Per-core[0] HBM base for the labeledDs allocate node, else None."""
    for node in dl["scheduleTree_"]:
        if node.get("nodeType_") == "allocate" and node.get("ldsIdx_") == lds_idx:
            if node.get("component_") != "hbm":
                return None
            return next(iter(node["startAddressCoreCorelet_"]["data_"].values()), None)
    return None


def _hbm_start_by_core(dl: dict, lds_idx: int) -> dict[int, int]:
    for node in dl["scheduleTree_"]:
        if node.get("nodeType_") != "allocate" or node.get("ldsIdx_") != lds_idx:
            continue
        if node.get("component_") != "hbm":
            return {}
        starts: dict[int, int] = {}
        for key, value in node.get("startAddressCoreCorelet_", {}).get(
            "data_", {}
        ).items():
            core_id = int(str(key).strip("[]").split(",", 1)[0])
            starts[core_id] = int(value)
        return starts
    return {}


def _label_indices(labels: list[str]) -> list[int]:
    return [int(lbl.rsplit("-idx", 1)[1]) for lbl in labels]


def _producer_output_indices(dl: dict) -> list[int]:
    return _label_indices(dl["computeOp_"][0]["outputLabeledDs"])


def _consumer_input_indices(dl: dict) -> list[int]:
    return _label_indices(dl["computeOp_"][0]["inputLabeledDs"])


def _future_consumers(sdscs_json: list[dict], start: int, hbm_addr: str):
    consumers = []
    for c in range(start + 1, len(sdscs_json)):
        cons = sdscs_json[c]
        cons_dl = _dl_op(cons)
        for in_idx in _consumer_input_indices(cons_dl):
            if _hbm_base(cons_dl, in_idx) != hbm_addr:
                continue
            # Scratch HBM addresses are reused.  Treat an input as belonging to
            # this producer only when no later producer between start and the
            # consumer wrote the same address.
            latest = None
            for p in range(c - 1, -1, -1):
                prod_dl = _dl_op(sdscs_json[p])
                if any(
                    _hbm_base(prod_dl, out_idx) == hbm_addr
                    for out_idx in _producer_output_indices(prod_dl)
                ):
                    latest = p
                    break
            if latest == start:
                consumers.append((c, cons, in_idx))
    return consumers


def _iter_sizes_from_dl(dl: dict, shard: dict[str, int]) -> dict[str, int] | None:
    sizes = dl.get("N_", {})
    iter_sizes: dict[str, int] = {}
    for dim in shard:
        key = f"{dim}_"
        if key not in sizes:
            return None
        iter_sizes[key] = int(sizes[key])
    return iter_sizes


def detect_attention_score_handoff(
    sdscs_json: list[dict],
    min_handoff_bytes: int = 1 << 20,
):
    """Find stock SDPA's same-stick QK^T score handoff fanout.

    The score matrix feeds both softmax max and sub.  Once the producer output
    is flipped to LX, both consumers must be fed from LX too; realizing only one
    leg leaves the other consumer reading a stale HBM address.

    Use the proven splice's same-stick roundtrip geometry for each fanout leg:
    split dim first, remaining non-stick dims next, stick dim last.  That shape
    exercises real cross-core L3 traffic and keeps the implementation out of the
    uncertified PT-LX/ReStickify path.
    """
    for p, prod in enumerate(sdscs_json):
        if _op_name(prod) != "batchmatmul":
            continue
        prod_dl = _dl_op(prod)
        out_indices = _producer_output_indices(prod_dl)
        if len(out_indices) != 1:
            continue
        out_idx = out_indices[0]
        prod_addr = _hbm_base(prod_dl, out_idx)
        if prod_addr is None:
            continue

        consumers = _future_consumers(sdscs_json, p, prod_addr)
        max_edges = [edge for edge in consumers if _op_name(edge[1]) == "max"]
        sub_edges = [edge for edge in consumers if _op_name(edge[1]) == "sub"]
        if len(max_edges) != 1 or len(sub_edges) != 1:
            continue
        if any(_op_name(edge[1]) not in {"max", "sub"} for edge in consumers):
            continue

        prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
        if prod_stick is None:
            continue
        bridged_edges = [max_edges[0], sub_edges[0]]
        if any(
            _stick_dim_for_lds(_dl_op(cons), in_idx) != prod_stick
            for _c, cons, in_idx in bridged_edges
        ):
            continue

        # Use the proven splice's data-op geometry: split dim first, remaining
        # non-stick dims next, stick dim last.  This is not necessarily the DL
        # primaryDsInfo_ order.
        _sub_c, sub_cons, sub_in_idx = sub_edges[0]
        sub_dl = _dl_op(sub_cons)
        shard = _body(sub_cons).get("numWkSlicesPerDim_", {})
        split_dim = _single_split_dim(shard)
        if split_dim is None:
            continue
        layout = [split_dim]
        for dim in shard:
            sym = _symbol_dim(dim)
            if sym not in (split_dim, prod_stick):
                layout.append(sym)
        layout.append(prod_stick)
        num_cores = int(_body(sub_cons).get("numCoresUsed_", 0))
        if num_cores <= 0:
            continue
        iter_sizes = _iter_sizes_for_layout(sub_dl, layout)
        if iter_sizes is None:
            continue
        if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
            continue
        lds = _lds_by_idx(prod_dl, out_idx)
        word_length = int((lds or {}).get("wordLength", 2))
        handoff_bytes = _handoff_bytes(iter_sizes, word_length)
        if handoff_bytes < min_handoff_bytes:
            continue
        slice_bytes = _reserve_bridge_region_bytes(
            per_core_same_stick_slice_bytes(
                iter_sizes, split_dim, prod_stick, STICK_SIZE, num_cores, word_length
            )
        )
        try:
            bases = allocate_lx_bases(3, slice_bytes, region0=0)
        except ValueError:
            continue
        return {
            "producer": prod,
            "producer_out_idx": out_idx,
            "consumers": bridged_edges,
            "iter_sizes": iter_sizes,
            "layout": layout,
            "stick_dim": prod_stick,
            "split_dim": split_dim,
            "num_cores": num_cores,
            "word_length": word_length,
            "handoff_bytes": handoff_bytes,
            "slice_bytes": slice_bytes,
            "producer_base": bases[0],
            "scratch_base": bases[1],
            "consumer_base": bases[2],
        }
    return None


def realize_attention_score_handoff(
    sdscs_json: list[dict],
    min_handoff_bytes: int = 1 << 20,
) -> bool:
    edge = detect_attention_score_handoff(sdscs_json, min_handoff_bytes)
    if edge is None:
        return False

    prod = edge["producer"]
    apply_lx_flip(
        prod,
        LxFlip(edge["producer_out_idx"], edge["producer_base"], "producer-output"),
    )
    for _c, cons, in_idx in edge["consumers"]:
        apply_lx_flip(
            cons,
            LxFlip(in_idx, edge["consumer_base"], "consumer-input"),
        )
        datadscs, opfuncs, sched = build_roundtrip_bridge(
            dim_pool=edge["layout"],
            iter_sizes=edge["iter_sizes"],
            stick_size=STICK_SIZE,
            num_cores=edge["num_cores"],
            lx_size=edge["slice_bytes"],
            producer_base=edge["producer_base"],
            scratch_base=edge["scratch_base"],
            consumer_base=edge["consumer_base"],
            layout=edge["layout"],
            stick_dim=edge["stick_dim"],
            split_dim=edge["split_dim"],
        )
        body = _body(cons)
        body["coreIdToDscSchedule"] = sched
        body["datadscs_"] = datadscs
        body["opFuncsUsed_"] = opfuncs
        _dl_op(cons)["numCoreletsUsed_DSC2_"] = 1
    return True


def detect_static_matmul_handoff(
    sdscs_json: list[dict],
    min_handoff_bytes: int = 1 << 20,
):
    """Find a static same-stick ``batchmatmul -> batchmatmul`` handoff.

    This targets the MoE static routing proxy: ``(perm @ x) @ w`` and
    ``(perm_w @ y) @ w``.  The routed activation is a single HBM-backed producer
    output consumed by one later matmul input.  Producer and consumer may name
    the hidden stick axis differently (``out_`` vs ``in_``), but the physical
    layout must preserve the stick position and split the same token/slot dim.
    """
    for p, prod in enumerate(sdscs_json):
        if _op_name(prod) != "batchmatmul":
            continue
        prod_dl = _dl_op(prod)
        out_indices = _producer_output_indices(prod_dl)
        if len(out_indices) != 1:
            continue
        out_idx = out_indices[0]
        prod_addr = _hbm_base(prod_dl, out_idx)
        if prod_addr is None:
            continue

        consumers = _future_consumers(sdscs_json, p, prod_addr)
        if len(consumers) != 1:
            continue
        _c, cons, in_idx = consumers[0]
        if _op_name(cons) != "batchmatmul":
            continue
        cons_dl = _dl_op(cons)

        prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
        cons_stick = _stick_dim_for_lds(cons_dl, in_idx)
        prod_layout = _layout_for_lds(prod_dl, out_idx)
        cons_layout = _layout_for_lds(cons_dl, in_idx)
        if (
            prod_stick is None
            or cons_stick is None
            or prod_layout is None
            or cons_layout is None
        ):
            continue
        if not _same_physical_stick_layout(
            prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
        ):
            continue

        prod_shard = _body(prod).get("numWkSlicesPerDim_", {})
        cons_shard = _body(cons).get("numWkSlicesPerDim_", {})
        prod_split = _single_split_dim(prod_shard)
        cons_split = _single_split_dim(cons_shard)
        if prod_split is None or cons_split is None or prod_split != cons_split:
            continue
        num_cores = int(_body(cons).get("numCoresUsed_", 0))
        split_factor = _split_factor(cons_shard, cons_split)
        if num_cores <= 0 or split_factor != num_cores:
            continue
        if _split_factor(prod_shard, prod_split) != split_factor:
            continue

        iter_sizes = _iter_sizes_for_layout(cons_dl, cons_layout)
        if iter_sizes is None:
            continue
        if cons_split not in iter_sizes or iter_sizes[cons_split] % num_cores != 0:
            continue
        lds = _lds_by_idx(prod_dl, out_idx)
        word_length = int((lds or {}).get("wordLength", 2))
        handoff_bytes = _handoff_bytes(iter_sizes, word_length)
        if handoff_bytes < min_handoff_bytes:
            continue
        slice_bytes = per_core_same_stick_slice_bytes(
            iter_sizes, cons_split, cons_stick, STICK_SIZE, num_cores, word_length
        )
        try:
            bases = allocate_lx_bases(3, slice_bytes, region0=0)
        except ValueError:
            continue
        return {
            "producer": prod,
            "producer_out_idx": out_idx,
            "consumer": cons,
            "consumer_in_idx": in_idx,
            "iter_sizes": iter_sizes,
            "layout": cons_layout,
            "stick_dim": cons_stick,
            "split_dim": cons_split,
            "num_cores": num_cores,
            "word_length": word_length,
            "handoff_bytes": handoff_bytes,
            "slice_bytes": slice_bytes,
            "producer_base": bases[0],
            "scratch_base": bases[1],
            "consumer_base": bases[2],
        }
    return None


def realize_static_matmul_handoff(
    sdscs_json: list[dict],
    min_handoff_bytes: int = 1 << 20,
) -> bool:
    edge = detect_static_matmul_handoff(sdscs_json, min_handoff_bytes)
    if edge is None:
        return False

    prod = edge["producer"]
    cons = edge["consumer"]
    apply_lx_flip(
        prod,
        LxFlip(edge["producer_out_idx"], edge["producer_base"], "producer-output"),
    )
    apply_lx_flip(
        cons,
        LxFlip(edge["consumer_in_idx"], edge["consumer_base"], "consumer-input"),
    )
    datadscs, opfuncs, sched = build_roundtrip_bridge(
        dim_pool=edge["layout"],
        iter_sizes=edge["iter_sizes"],
        stick_size=STICK_SIZE,
        num_cores=edge["num_cores"],
        lx_size=edge["slice_bytes"],
        producer_base=edge["producer_base"],
        scratch_base=edge["scratch_base"],
        consumer_base=edge["consumer_base"],
        layout=edge["layout"],
        stick_dim=edge["stick_dim"],
        split_dim=edge["split_dim"],
    )
    body = _body(cons)
    body["coreIdToDscSchedule"] = sched
    body["datadscs_"] = datadscs
    body["opFuncsUsed_"] = opfuncs
    _dl_op(cons)["numCoreletsUsed_DSC2_"] = 1
    return True


def _exact_tile_bytes_for_tiles(slice_bytes: int, num_tiles: int) -> int | None:
    """Return a stick-aligned tile size that yields exactly ``num_tiles``."""
    if slice_bytes <= 0 or num_tiles <= 0:
        return None
    for tile_bytes in range(STICK_BYTES, slice_bytes + STICK_BYTES, STICK_BYTES):
        if num_stream_tiles(slice_bytes, tile_bytes) == num_tiles:
            return tile_bytes
    return None


def _flash_pipeline_row_dim(layout: list[str], split_dim: str, iter_sizes: dict):
    candidates = [dim for dim in layout if dim != split_dim]
    if not candidates:
        return None
    return max(candidates, key=lambda dim: int(iter_sizes.get(dim, 0)))


def build_flash_attention_pipeline_artifact(
    sdscs_json: list[dict],
    *,
    overlap: bool = False,
    name: str = "mixed_flash_pipeline_artifact",
) -> dict | None:
    """Build a non-executed mixed-SDSC proof from real flash-prefill compute SDSCs.

    The artifact combines the generated batchmatmul compute DSCs from one
    flash-prefill bundle with the Stage009 double-buffered STCDPOpLx schedule.
    It is intentionally sidecar-only: no producer/consumer descriptors are
    flipped and bundle.mlir should not execute it yet.
    """
    tile_sdscs = [sdsc for sdsc in sdscs_json if _op_name(sdsc) == "batchmatmul"]
    if not tile_sdscs:
        return None

    first = tile_sdscs[0]
    first_body = _body(first)
    num_cores = int(first_body.get("numCoresUsed_", 0))
    if num_cores <= 0:
        return None
    first_dl = _dl_op(first)
    out_indices = _producer_output_indices(first_dl)
    if len(out_indices) != 1:
        return None
    out_idx = out_indices[0]
    layout = _layout_for_lds(first_dl, out_idx)
    stick_dim = _stick_dim_for_lds(first_dl, out_idx)
    split_dim = _single_split_dim(first_body.get("numWkSlicesPerDim_", {}))
    if layout is None or stick_dim is None or split_dim is None:
        return None
    iter_sizes = _iter_sizes_for_layout(first_dl, layout)
    if iter_sizes is None:
        return None

    row_dim = _flash_pipeline_row_dim(layout, split_dim, iter_sizes)
    if row_dim is None:
        return None
    num_tiles = len(tile_sdscs)
    slice_bytes = per_core_same_stick_slice_bytes(
        iter_sizes,
        split_dim,
        stick_dim,
        STICK_SIZE,
        num_cores,
    )
    tile_bytes = _exact_tile_bytes_for_tiles(slice_bytes, num_tiles)
    if tile_bytes is None:
        return None
    try:
        bases = allocate_flash_attention_pipeline_bases(
            num_lanes=2,
            tile_bytes=tile_bytes,
            scratch_regions=2,
            region0=PRODUCER_LX_BASE,
        )
    except ValueError:
        return None

    datadscs, opfuncs, schedule = build_flash_attention_pipeline_bridge(
        dim_pool=layout,
        iter_sizes=iter_sizes,
        stick_size=STICK_SIZE,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
        src_bases=bases["source_bases"],
        dst_lane_bases=bases["lane_bases"],
        layout=layout,
        stick_dim=stick_dim,
        split_dim=split_dim,
        row_dim=row_dim,
        lane_names=["k", "v"],
        tile_bytes=tile_bytes,
        overlap=overlap,
    )
    compute_dscs = [copy.deepcopy(_body(sdsc)["dscs_"][0]) for sdsc in tile_sdscs]
    artifact = build_flash_attention_pipeline_mixed_sdsc(
        name,
        datadscs,
        opfuncs,
        schedule,
        compute_dscs,
        num_cores,
    )
    root = artifact[name]
    for key in (
        "sdscFoldProps_",
        "sdscFolds_",
        "coreFoldProp_",
        "coreletFoldProp_",
        "coreIdToDsc_",
        "numWkSlicesPerDim_",
        "coreIdToWkSlice_",
    ):
        if key in first_body:
            root[key] = copy.deepcopy(first_body[key])
    root["flashAttentionPipeline_"].update(
        {
            "source": "generated-flash-prefill-batchmatmul-tiles",
            "row_dim": row_dim,
            "split_dim": split_dim,
            "stick_dim": stick_dim,
            "layout": layout,
            "iter_sizes": iter_sizes,
            "tile_bytes": tile_bytes,
        }
    )
    return artifact


def build_flash_attention_pipeline_tile_artifacts(
    sdscs_json: list[dict],
    *,
    name_prefix: str = "mixed_flash_pipeline_tile",
    overlap_prefix: bool = False,
) -> list[dict]:
    """Build DXP-compatible one-compute mixed sidecars for flash-prefill tiles."""
    artifacts = []
    tile_sdscs = [sdsc for sdsc in sdscs_json if _op_name(sdsc) == "batchmatmul"]
    for tile_index, sdsc in enumerate(tile_sdscs):
        artifact = None
        overlap_reasons: list[str] = []
        if overlap_prefix and tile_index + 1 < len(tile_sdscs):
            overlap_reasons = (
                flash_attention_overlap_prefix_rejection_reasons(
                    tile_sdscs[tile_index: tile_index + 2],
                )
            )
            if not overlap_reasons:
                artifact = build_flash_attention_pipeline_ifn_prefix_tile_artifact(
                    tile_sdscs[tile_index: tile_index + 2],
                    tile_index,
                    name_prefix=name_prefix,
                )
            if artifact is None:
                artifact = build_flash_attention_pipeline_overlap_prefix_tile_artifact(
                    tile_sdscs[tile_index: tile_index + 2],
                    tile_index,
                    name_prefix=name_prefix,
                )
        if artifact is None:
            artifact = build_flash_attention_pipeline_artifact(
                [sdsc],
                overlap=False,
                name=f"{name_prefix}_{tile_index}",
            )
        if artifact is None:
            continue
        root = artifact[next(iter(artifact))]
        root["flashAttentionPipeline_"]["tile_index"] = tile_index
        root["flashAttentionPipeline_"]["replaces_sdsc"] = next(iter(sdsc))
        root["flashAttentionPipeline_"].setdefault("overlap_prefix", False)
        if overlap_prefix and not root["flashAttentionPipeline_"]["overlap_prefix"]:
            root["flashAttentionPipeline_"]["overlap_prefix_requested"] = True
            root["flashAttentionPipeline_"][
                "overlap_prefix_rejection_reasons"
            ] = overlap_reasons or ["not_enough_following_tiles"]
        artifacts.append(artifact)
    return artifacts


def flash_attention_overlap_prefix_rejection_reasons(
    tile_sdscs: list[dict],
) -> list[str]:
    """Explain why an overlap-prefix sidecar would fail closed for these tiles."""
    if len(tile_sdscs) < 2:
        return ["needs_two_batchmatmul_tiles"]

    first = tile_sdscs[0]
    first_body = _body(first)
    num_cores = int(first_body.get("numCoresUsed_", 0))
    if num_cores <= 0:
        return ["invalid_num_cores"]
    first_dl = _dl_op(first)
    out_indices = _producer_output_indices(first_dl)
    if len(out_indices) != 1:
        return ["first_tile_output_count_not_one"]
    out_idx = out_indices[0]
    layout = _layout_for_lds(first_dl, out_idx)
    stick_dim = _stick_dim_for_lds(first_dl, out_idx)
    split_dim = _single_split_dim(first_body.get("numWkSlicesPerDim_", {}))
    if layout is None:
        return ["missing_first_output_layout"]
    if stick_dim is None:
        return ["missing_first_output_stick_dim"]
    if split_dim is None:
        return ["missing_single_split_dim"]
    iter_sizes = _iter_sizes_for_layout(first_dl, layout)
    if iter_sizes is None:
        return ["missing_iter_sizes"]

    second = tile_sdscs[1]
    second_body = _body(second)
    if int(second_body.get("numCoresUsed_", 0)) != num_cores:
        return ["next_tile_num_cores_mismatch"]
    second_dl = _dl_op(second)
    second_out_indices = _producer_output_indices(second_dl)
    if len(second_out_indices) != 1:
        return ["next_tile_output_count_not_one"]
    second_out_idx = second_out_indices[0]
    if _layout_for_lds(second_dl, second_out_idx) != layout:
        return ["next_tile_output_layout_mismatch"]
    if _stick_dim_for_lds(second_dl, second_out_idx) != stick_dim:
        return ["next_tile_output_stick_dim_mismatch"]
    if _single_split_dim(second_body.get("numWkSlicesPerDim_", {})) != split_dim:
        return ["next_tile_split_dim_mismatch"]
    if _iter_sizes_for_layout(second_dl, layout) != iter_sizes:
        return ["next_tile_iter_sizes_mismatch"]

    row_dim = _flash_pipeline_row_dim(layout, split_dim, iter_sizes)
    if row_dim is None:
        return ["missing_row_dim"]
    slice_bytes = per_core_same_stick_slice_bytes(
        iter_sizes,
        split_dim,
        stick_dim,
        STICK_SIZE,
        num_cores,
    )
    if _exact_tile_bytes_for_tiles(slice_bytes, 2) is None:
        return ["cannot_make_two_prefetch_tiles"]
    try:
        allocate_flash_attention_pipeline_bases(
            num_lanes=2,
            tile_bytes=_exact_tile_bytes_for_tiles(slice_bytes, 2),
            scratch_regions=2,
            region0=PRODUCER_LX_BASE,
        )
    except ValueError:
        return ["lx_allocation_exceeds_capacity"]

    return []


def build_flash_attention_pipeline_ifn_prefix_tile_artifact(
    tile_sdscs: list[dict],
    tile_index: int,
    *,
    name_prefix: str = "mixed_flash_pipeline_tile",
) -> dict | None:
    """Build a one-row InputFetchNeighbor-shaped flash overlap probe.

    This attaches one data-op to the first batchmatmul input and rewrites that
    input LX-resident, so Deeptools can lower it through its paired-row
    InputFetchNeighbor path instead of the independent synthetic sidecar path.
    The first probe intentionally targets lds0 only; K/V streaming needs broader
    descriptor and Deeptools support for non-split-dim inputs.
    """
    if len(tile_sdscs) < 2:
        return None

    first = tile_sdscs[0]
    first_body = _body(first)
    num_cores = int(first_body.get("numCoresUsed_", 0))
    if num_cores <= 0:
        return None

    first_dl = _dl_op(first)
    input_indices = _consumer_input_indices(first_dl)
    if not input_indices:
        return None
    input_idx = input_indices[0]
    if input_idx != INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX:
        return None

    layout = _layout_for_lds(first_dl, input_idx)
    stick_dim = _stick_dim_for_lds(first_dl, input_idx)
    split_dim = _single_split_dim(first_body.get("numWkSlicesPerDim_", {}))
    if layout is None or stick_dim is None or split_dim is None:
        return None
    if split_dim not in layout:
        return None
    iter_sizes = _iter_sizes_for_layout(first_dl, layout)
    if iter_sizes is None:
        return None
    if iter_sizes[split_dim] % num_cores != 0:
        return None

    compute_dsc = copy.deepcopy(first_body["dscs_"][0])
    compute_root = {next(iter(first)): {"dscs_": [compute_dsc]}}
    apply_lx_flip(
        compute_root,
        LxFlip(input_idx, CONSUMER_LX_BASE, "ifn-consumer-input"),
    )
    compute_dl = next(iter(compute_dsc.values()))
    _add_input_fetch_neighbor_transfer(compute_dl, input_idx)
    _add_input_fetch_neighbor_legacy_dims(
        compute_dl,
        first_body.get("numWkSlicesPerDim_", {}),
    )

    datadsc = make_datadsc(
        f"0_STCDPOpLx_prefetch_ifn_Tensor0_idx{input_idx}_tile{tile_index}",
        _stcdp_op(),
        layout,
        src=Endpoint(layout, stick_dim, split_dim, CONSUMER_LX_BASE),
        dst=Endpoint(layout, stick_dim, split_dim, CONSUMER_LX_BASE),
        iter_sizes=iter_sizes,
        stick_size=STICK_SIZE,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
    )
    schedule = {
        str(core_id): [[0, 0, 0, 0]]
        for core_id in range(num_cores)
    }

    name = f"{name_prefix}_{tile_index}"
    artifact = build_flash_attention_pipeline_mixed_sdsc(
        name,
        [datadsc],
        ["STCDPOpLx"],
        schedule,
        [compute_dsc],
        num_cores,
    )
    root = artifact[name]
    for key in (
        "sdscFoldProps_",
        "sdscFolds_",
        "coreFoldProp_",
        "coreletFoldProp_",
        "coreIdToDsc_",
        "numWkSlicesPerDim_",
        "coreIdToWkSlice_",
    ):
        if key in first_body:
            root[key] = copy.deepcopy(first_body[key])
    root["flashAttentionPipeline_"].update(
        {
            "source": "generated-flash-prefill-overlap-prefix-ifn-tile",
            "split_dim": split_dim,
            "stick_dim": stick_dim,
            "layout": layout,
            "iter_sizes": iter_sizes,
            "tile_index": tile_index,
            "replaces_sdsc": next(iter(first)),
            "ifn_attached_input_idx": input_idx,
            "ifn_input_lx_base": CONSUMER_LX_BASE,
            "ifn_runtime_safe": False,
            "ifn_runtime_rejection_reason": "single_sdsc_ifn_no_real_predecessor",
            "compute_tile_count": 1,
            "overlap_prefix": True,
        }
    )
    return artifact


def _flash_attention_ifn_pair_edge(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    input_idx: int = INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX,
) -> tuple[dict | None, list[str]]:
    tile = _flash_value_flow_tile(sdscs_json, tile_index)
    if tile is None:
        return None, ["tile_not_found"]
    c, cons = tile
    cons_dl = _dl_op(cons)
    if input_idx not in _consumer_input_indices(cons_dl):
        return None, [f"input{input_idx}:not_consumer_input"]
    if input_idx != INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX:
        return None, [f"input{input_idx}:not_supported_ifn_input"]

    cons_body = _body(cons)
    num_cores = int(cons_body.get("numCoresUsed_", 0))
    if num_cores <= 0:
        return None, ["invalid_num_cores"]
    addr = _hbm_base(cons_dl, input_idx)
    if addr is None:
        return None, [f"input{input_idx}:not_hbm_backed"]

    producer = _latest_producer_of_hbm(sdscs_json, c, addr)
    if producer is None:
        return None, [f"input{input_idx}:no_latest_producer"]
    p, prod, out_idx = producer

    future = _future_consumers(sdscs_json, p, addr)
    if len(future) != 1 or future[0][0] != c or future[0][2] != input_idx:
        future_names = [
            f"{next(iter(fcons))}:input{fin_idx}"
            for _fc, fcons, fin_idx in future
        ]
        return (
            None,
            [
                f"input{input_idx}:not_single_consumer:"
                f"{','.join(future_names)}"
            ],
        )

    prod_dl = _dl_op(prod)
    prod_layout = _layout_for_lds(prod_dl, out_idx)
    prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
    cons_layout = _layout_for_lds(cons_dl, input_idx)
    cons_stick = _stick_dim_for_lds(cons_dl, input_idx)
    split_dim = _single_split_dim(cons_body.get("numWkSlicesPerDim_", {}))
    if (
        prod_layout is None
        or prod_stick is None
        or cons_layout is None
        or cons_stick is None
        or split_dim is None
    ):
        return None, [f"input{input_idx}:missing_layout_stick_or_split"]
    mismatch_reason = _layout_mismatch_reason(
        f"input{input_idx}",
        prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
    )
    if mismatch_reason is not None:
        return None, [mismatch_reason]

    iter_sizes = _iter_sizes_for_layout(cons_dl, cons_layout)
    if iter_sizes is None:
        return None, [f"input{input_idx}:missing_iter_sizes"]
    if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
        return None, [f"input{input_idx}:invalid_split:{split_dim}"]

    slice_bytes = _reserve_bridge_region_bytes(
        per_core_same_stick_slice_bytes(
            iter_sizes,
            split_dim,
            cons_stick,
            STICK_SIZE,
            num_cores,
        )
    )
    try:
        allocate_lx_bases(2, slice_bytes, region0=CONSUMER_LX_BASE)
    except ValueError:
        return None, [f"input{input_idx}:lx_allocation_exceeds_capacity"]

    return (
        {
            "producer_index": p,
            "consumer_index": c,
            "producer": prod,
            "consumer": cons,
            "producer_idx": out_idx,
            "consumer_idx": input_idx,
            "shared_hbm_addr": addr,
            "layout": cons_layout,
            "stick_dim": cons_stick,
            "split_dim": split_dim,
            "iter_sizes": iter_sizes,
            "slice_bytes": slice_bytes,
            "producer_layout": prod_layout,
            "producer_stick_dim": prod_stick,
        },
        [],
    )


def flash_attention_ifn_pair_tile_rejection_reasons(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    input_idx: int = INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX,
) -> list[str]:
    """Explain why an explicit same-physical LX-copy pair cannot be emitted."""
    _edge, reasons = _flash_attention_ifn_pair_edge(
        sdscs_json,
        tile_index,
        input_idx=input_idx,
    )
    return reasons


def _flash_attention_layout_xform_pair_edge(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    input_idx: int = INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX,
    allow_nonzero_input: bool = False,
) -> tuple[dict | None, list[str]]:
    tile = _flash_value_flow_tile(sdscs_json, tile_index)
    if tile is None:
        return None, ["tile_not_found"]
    c, cons = tile
    cons_dl = _dl_op(cons)
    if input_idx not in _consumer_input_indices(cons_dl):
        return None, [f"input{input_idx}:not_consumer_input"]
    if input_idx != INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX and not allow_nonzero_input:
        return None, [f"input{input_idx}:not_supported_layout_xform_input"]

    cons_body = _body(cons)
    num_cores = int(cons_body.get("numCoresUsed_", 0))
    if num_cores <= 0:
        return None, ["invalid_num_cores"]
    addr = _hbm_base(cons_dl, input_idx)
    if addr is None:
        return None, [f"input{input_idx}:not_hbm_backed"]

    producer = _latest_producer_of_hbm(sdscs_json, c, addr)
    if producer is None:
        return None, [f"input{input_idx}:no_latest_producer"]
    p, prod, out_idx = producer

    future = _future_consumers(sdscs_json, p, addr)
    if len(future) != 1 or future[0][0] != c or future[0][2] != input_idx:
        future_names = [
            f"{next(iter(fcons))}:input{fin_idx}"
            for _fc, fcons, fin_idx in future
        ]
        return (
            None,
            [
                f"input{input_idx}:not_single_consumer:"
                f"{','.join(future_names)}"
            ],
        )

    prod_dl = _dl_op(prod)
    prod_layout = _layout_for_lds(prod_dl, out_idx)
    prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
    cons_layout = _layout_for_lds(cons_dl, input_idx)
    cons_stick = _stick_dim_for_lds(cons_dl, input_idx)
    cons_split = _single_split_dim(cons_body.get("numWkSlicesPerDim_", {}))
    prod_split = _single_split_dim(_body(prod).get("numWkSlicesPerDim_", {}))
    if (
        prod_layout is None
        or prod_stick is None
        or cons_layout is None
        or cons_stick is None
        or cons_split is None
    ):
        return None, [f"input{input_idx}:missing_layout_stick_or_split"]

    mismatch_reason = _layout_mismatch_reason(
        f"input{input_idx}",
        prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
    )
    if mismatch_reason is None:
        return None, [f"input{input_idx}:same_physical_layout_use_ifn_pair"]
    if not mismatch_reason.startswith(f"input{input_idx}:layout_transform_required:"):
        return None, [mismatch_reason]

    dim_map = _layout_transform_dim_map(
        prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
    )
    if dim_map is None:
        return None, [f"input{input_idx}:layout_transform_dim_map_missing"]
    source_layout = [dim_map[dim] for dim in prod_layout]
    if set(source_layout) != set(cons_layout):
        return (
            None,
            [
                f"input{input_idx}:layout_transform_dim_set_mismatch:"
                f"producer={source_layout}:consumer={cons_layout}"
            ],
        )

    iter_sizes = _iter_sizes_for_layout(cons_dl, cons_layout)
    if iter_sizes is None or any(dim not in iter_sizes for dim in source_layout):
        return None, [f"input{input_idx}:missing_iter_sizes"]
    xform_split = cons_split
    if xform_split not in iter_sizes and prod_split is not None:
        xform_split = dim_map.get(prod_split)
    source_pieces = _mapped_work_slice_piece_info(
        _body(prod),
        prod_layout,
        dim_map,
        iter_sizes,
        PRODUCER_LX_BASE,
    )
    if source_pieces is None:
        return None, [f"input{input_idx}:producer_piece_map_missing"]
    if (
        xform_split is None
        or xform_split not in iter_sizes
        or iter_sizes[xform_split] % num_cores != 0
    ):
        prod_num_cores = int(_body(prod).get("numCoresUsed_", 0))
        if (
            input_idx != INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX
            and prod_split is not None
            and xform_split == dim_map.get(prod_split)
            and xform_split in iter_sizes
            and cons_split not in iter_sizes
            and prod_num_cores > 0
            and iter_sizes[xform_split] % prod_num_cores == 0
        ):
            return (
                None,
                [
                    f"input{input_idx}:requires_kv_repack_broadcast:"
                    f"producer_split={prod_split}:mapped_split={xform_split}:"
                    f"consumer_split={cons_split}:"
                    f"producer_cores={prod_num_cores}:"
                    f"consumer_cores={num_cores}"
                ],
            )
        return None, [f"input{input_idx}:invalid_split:{cons_split}"]

    slice_bytes = _reserve_bridge_region_bytes(
        per_core_same_stick_slice_bytes(
            iter_sizes,
            xform_split,
            cons_stick,
            STICK_SIZE,
            num_cores,
        )
    )
    try:
        allocate_lx_bases(2, slice_bytes, region0=CONSUMER_LX_BASE)
    except ValueError:
        return None, [f"input{input_idx}:lx_allocation_exceeds_capacity"]

    return (
        {
            "producer_index": p,
            "consumer_index": c,
            "producer": prod,
            "consumer": cons,
            "producer_idx": out_idx,
            "consumer_idx": input_idx,
            "shared_hbm_addr": addr,
            "source_layout": source_layout,
            "source_pieces": source_pieces,
            "dim_map": dim_map,
            "consumer_layout": cons_layout,
            "dim_pool": source_layout,
            "stick_dim": cons_stick,
            "split_dim": xform_split,
            "iter_sizes": iter_sizes,
            "slice_bytes": slice_bytes,
            "producer_layout": prod_layout,
            "producer_stick_dim": prod_stick,
        },
        [],
    )


def flash_attention_layout_xform_pair_tile_rejection_reasons(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    input_idx: int = INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX,
) -> list[str]:
    """Explain why a layout-transforming flash pair cannot be emitted."""
    if (
        tile_index == LAYOUT_XFORM_PAIR_AUTO_TILE
        and input_idx == INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX
    ):
        return flash_attention_layout_xform_pair_rejection_reasons(
            sdscs_json,
            tile_index,
        )
    _edge, reasons = _flash_attention_layout_xform_pair_edge(
        sdscs_json,
        tile_index,
        input_idx=input_idx,
    )
    return reasons


def _resolve_flash_attention_layout_xform_pair_edge(
    sdscs_json: list[dict],
    tile_index: int,
) -> tuple[int | None, dict | None, list[str]]:
    if tile_index != LAYOUT_XFORM_PAIR_AUTO_TILE:
        edge, reasons = _flash_attention_layout_xform_pair_edge(sdscs_json, tile_index)
        return (tile_index if edge is not None else None), edge, reasons

    reasons: list[str] = []
    for candidate in range(_flash_value_flow_tile_count(sdscs_json)):
        edge, candidate_reasons = _flash_attention_layout_xform_pair_edge(
            sdscs_json,
            candidate,
        )
        if edge is not None:
            return candidate, edge, []
        reasons.extend(f"tile{candidate}:{reason}" for reason in candidate_reasons)
    return None, None, reasons or ["auto:no_candidate_tiles"]


def flash_attention_layout_xform_pair_rejection_reasons(
    sdscs_json: list[dict],
    tile_index: int,
) -> list[str]:
    """Explain why a requested or auto-selected layout-transform pair failed."""
    _tile, _edge, reasons = _resolve_flash_attention_layout_xform_pair_edge(
        sdscs_json,
        tile_index,
    )
    return reasons


def build_flash_attention_ifn_pair_tile_artifacts(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    name_prefix: str = "mixed_flash_ifn_pair_tile",
) -> dict | None:
    """Build an explicit predecessor+consumer LX-copy pair for one real edge."""
    edge, reasons = _flash_attention_ifn_pair_edge(sdscs_json, tile_index)
    if edge is None:
        return None

    prod_name = next(iter(edge["producer"]))
    cons_name = next(iter(edge["consumer"]))
    pred_sidecar = f"{name_prefix}_{tile_index}_predecessor"
    cons_sidecar = f"{name_prefix}_{tile_index}_consumer"

    producer_artifact = {pred_sidecar: copy.deepcopy(_body(edge["producer"]))}
    apply_lx_flip(
        producer_artifact,
        LxFlip(
            edge["producer_idx"],
            PRODUCER_LX_BASE,
            "ifn-predecessor-output",
        ),
    )
    producer_artifact[pred_sidecar].setdefault("flashAttentionPipeline_", {}).update(
        {
            "source": "generated-flash-prefill-predecessor-ifn-pair-producer",
            "ifn_mode": "predecessor_backed_pair",
            "ifn_pair_role": "predecessor",
            "ifn_runtime_safe": True,
            "ifn_predecessor_sdsc": prod_name,
            "ifn_predecessor_sidecar": pred_sidecar,
            "ifn_consumer_sdsc": cons_name,
            "ifn_consumer_sidecar": cons_sidecar,
            "ifn_predecessor_output_idx": edge["producer_idx"],
            "ifn_shared_hbm_addr": edge["shared_hbm_addr"],
            "ifn_predecessor_lx_base": PRODUCER_LX_BASE,
            "replaces_sdsc": prod_name,
            "tile_index": tile_index,
        }
    )

    compute_dsc = copy.deepcopy(_body(edge["consumer"])["dscs_"][0])
    compute_root = {cons_name: {"dscs_": [compute_dsc]}}
    apply_lx_flip(
        compute_root,
        LxFlip(
            edge["consumer_idx"],
            CONSUMER_LX_BASE,
            "ifn-consumer-input",
        ),
    )
    compute_dl = next(iter(compute_dsc.values()))
    _add_input_fetch_neighbor_transfer(compute_dl, edge["consumer_idx"])
    _add_input_fetch_neighbor_legacy_dims(
        compute_dl,
        _body(edge["consumer"]).get("numWkSlicesPerDim_", {}),
    )

    placeholder = make_datadsc(
        (
            "0_STCDPOpLx_predecessor_fetch_"
            f"Tensor0_idx{edge['consumer_idx']}_tile{tile_index}"
        ),
        _stcdp_op(),
        edge["layout"],
        src=Endpoint(
            edge["layout"],
            edge["stick_dim"],
            edge["split_dim"],
            PRODUCER_LX_BASE,
        ),
        dst=Endpoint(
            edge["layout"],
            edge["stick_dim"],
            edge["split_dim"],
            CONSUMER_LX_BASE,
        ),
        iter_sizes=edge["iter_sizes"],
        stick_size=STICK_SIZE,
        num_cores=int(_body(edge["consumer"]).get("numCoresUsed_", 0)),
        lx_size=DATAOP_LX_SIZE,
    )
    schedule = {
        str(core_id): [[0, -1, 0, 1], [-1, 0, 1, 0]]
        for core_id in range(int(_body(edge["consumer"]).get("numCoresUsed_", 0)))
    }
    consumer_artifact = build_flash_attention_pipeline_mixed_sdsc(
        cons_sidecar,
        [placeholder],
        ["STCDPOpLx"],
        schedule,
        [compute_dsc],
        int(_body(edge["consumer"]).get("numCoresUsed_", 0)),
    )
    consumer_root = consumer_artifact[cons_sidecar]
    cons_body = _body(edge["consumer"])
    for key in (
        "sdscFoldProps_",
        "sdscFolds_",
        "coreFoldProp_",
        "coreletFoldProp_",
        "coreIdToDsc_",
        "numWkSlicesPerDim_",
        "coreIdToWkSlice_",
    ):
        if key in cons_body:
            consumer_root[key] = copy.deepcopy(cons_body[key])
    consumer_root["flashAttentionPipeline_"].update(
        {
            "source": "generated-flash-prefill-predecessor-ifn-pair-consumer",
            "ifn_mode": "predecessor_backed_lx_copy_pair",
            "ifn_pair_role": "consumer",
            "ifn_runtime_safe": True,
            "ifn_predecessor_sdsc": prod_name,
            "ifn_predecessor_sidecar": pred_sidecar,
            "ifn_consumer_sdsc": cons_name,
            "ifn_consumer_sidecar": cons_sidecar,
            "ifn_predecessor_output_idx": edge["producer_idx"],
            "ifn_attached_input_idx": edge["consumer_idx"],
            "ifn_shared_hbm_addr": edge["shared_hbm_addr"],
            "ifn_predecessor_lx_base": PRODUCER_LX_BASE,
            "ifn_input_lx_base": CONSUMER_LX_BASE,
            "ifn_predecessor_layout": edge["producer_layout"],
            "ifn_consumer_layout": edge["layout"],
            "ifn_predecessor_stick_dim": edge["producer_stick_dim"],
            "ifn_consumer_stick_dim": edge["stick_dim"],
            "split_dim": edge["split_dim"],
            "iter_sizes": edge["iter_sizes"],
            "slice_bytes": edge["slice_bytes"],
            "replaces_sdsc": cons_name,
            "tile_index": tile_index,
            "compute_tile_count": 1,
        }
    )

    return {
        "artifacts": [producer_artifact, consumer_artifact],
        "replacements": {
            prod_name: pred_sidecar,
            cons_name: cons_sidecar,
        },
        "bundle_attrs": {},
        "rejection_reasons": reasons,
    }


def build_flash_attention_layout_xform_pair_tile_artifacts(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    name_prefix: str = "mixed_flash_layout_xform_pair_tile",
    overlap_consumer: bool = False,
) -> dict | None:
    """Build an experimental explicit LX-copy pair for a layout-transform edge."""
    selected_tile, edge, reasons = _resolve_flash_attention_layout_xform_pair_edge(
        sdscs_json,
        tile_index,
    )
    if edge is None or selected_tile is None:
        return None

    prod_name = next(iter(edge["producer"]))
    cons_name = next(iter(edge["consumer"]))
    pred_sidecar = f"{name_prefix}_{selected_tile}_predecessor"
    cons_sidecar = f"{name_prefix}_{selected_tile}_consumer"
    num_cores = int(_body(edge["consumer"]).get("numCoresUsed_", 0))

    producer_artifact = {pred_sidecar: copy.deepcopy(_body(edge["producer"]))}
    apply_lx_flip(
        producer_artifact,
        LxFlip(
            edge["producer_idx"],
            PRODUCER_LX_BASE,
            "layout-xform-predecessor-output",
        ),
    )
    producer_artifact[pred_sidecar].setdefault("flashAttentionPipeline_", {}).update(
        {
            "source": (
                "generated-flash-prefill-layout-xform-overlap-pair-producer"
                if overlap_consumer
                else "generated-flash-prefill-layout-xform-pair-producer"
            ),
            "layout_xform_mode": "same_dim_lx_copy_pair",
            "layout_xform_pair_role": "predecessor",
            "layout_xform_experimental": True,
            "layout_xform_overlap_consumer": overlap_consumer,
            "layout_xform_predecessor_sdsc": prod_name,
            "layout_xform_predecessor_sidecar": pred_sidecar,
            "layout_xform_consumer_sdsc": cons_name,
            "layout_xform_consumer_sidecar": cons_sidecar,
            "layout_xform_predecessor_output_idx": edge["producer_idx"],
            "layout_xform_shared_hbm_addr": edge["shared_hbm_addr"],
            "layout_xform_predecessor_lx_base": PRODUCER_LX_BASE,
            "replaces_sdsc": prod_name,
            "tile_index": selected_tile,
            "requested_tile_index": tile_index,
        }
    )

    compute_dsc = copy.deepcopy(_body(edge["consumer"])["dscs_"][0])
    compute_root = {cons_name: {"dscs_": [compute_dsc]}}
    apply_lx_flip(
        compute_root,
        LxFlip(
            edge["consumer_idx"],
            CONSUMER_LX_BASE,
            "layout-xform-consumer-input",
        ),
    )
    compute_dl = next(iter(compute_dsc.values()))
    _add_input_fetch_neighbor_transfer(compute_dl, edge["consumer_idx"])
    _add_input_fetch_neighbor_legacy_dims(
        compute_dl,
        _body(edge["consumer"]).get("numWkSlicesPerDim_", {}),
    )

    dataop_prefix = (
        "0_STCDPOpLx_prefetch_layout_xform_"
        if overlap_consumer
        else "0_STCDPOpLx_layout_xform_"
    )
    dataop = make_datadsc(
        f"{dataop_prefix}Tensor0_idx{edge['consumer_idx']}_tile{selected_tile}",
        _stcdp_op(),
        edge["dim_pool"],
        src=Endpoint(
            edge["source_layout"],
            edge["stick_dim"],
            edge["split_dim"],
            PRODUCER_LX_BASE,
        ),
        dst=Endpoint(
            edge["consumer_layout"],
            edge["stick_dim"],
            edge["split_dim"],
            CONSUMER_LX_BASE,
        ),
        iter_sizes=edge["iter_sizes"],
        stick_size=STICK_SIZE,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
    )
    next(iter(dataop.values()))["labeledDs_"][0]["PieceInfo"] = edge[
        "source_pieces"
    ]
    rows = (
        [[0, 0, 0, 0]]
        if overlap_consumer
        else [[0, -1, 0, 1], [-1, 0, 1, 0]]
    )
    schedule = {
        str(core_id): [list(row) for row in rows]
        for core_id in range(num_cores)
    }
    consumer_artifact = build_flash_attention_pipeline_mixed_sdsc(
        cons_sidecar,
        [dataop],
        ["STCDPOpLx"],
        schedule,
        [compute_dsc],
        num_cores,
    )
    consumer_root = consumer_artifact[cons_sidecar]
    cons_body = _body(edge["consumer"])
    for key in (
        "sdscFoldProps_",
        "sdscFolds_",
        "coreFoldProp_",
        "coreletFoldProp_",
        "coreIdToDsc_",
        "numWkSlicesPerDim_",
        "coreIdToWkSlice_",
    ):
        if key in cons_body:
            consumer_root[key] = copy.deepcopy(cons_body[key])
    consumer_root["flashAttentionPipeline_"].update(
        {
            "source": (
                "generated-flash-prefill-layout-xform-overlap-pair-consumer"
                if overlap_consumer
                else "generated-flash-prefill-layout-xform-pair-consumer"
            ),
            "layout_xform_mode": "same_dim_lx_copy_pair",
            "layout_xform_pair_role": "consumer",
            "layout_xform_experimental": True,
            "layout_xform_overlap_consumer": overlap_consumer,
            "layout_xform_runtime_safe": not overlap_consumer,
            "layout_xform_runtime_forced": overlap_consumer,
            "layout_xform_predecessor_sdsc": prod_name,
            "layout_xform_predecessor_sidecar": pred_sidecar,
            "layout_xform_consumer_sdsc": cons_name,
            "layout_xform_consumer_sidecar": cons_sidecar,
            "layout_xform_predecessor_output_idx": edge["producer_idx"],
            "layout_xform_attached_input_idx": edge["consumer_idx"],
            "layout_xform_shared_hbm_addr": edge["shared_hbm_addr"],
            "layout_xform_predecessor_lx_base": PRODUCER_LX_BASE,
            "layout_xform_input_lx_base": CONSUMER_LX_BASE,
            "layout_xform_original_predecessor_layout": edge["producer_layout"],
            "layout_xform_source_layout": edge["source_layout"],
            "layout_xform_consumer_layout": edge["consumer_layout"],
            "layout_xform_original_predecessor_stick_dim": (
                edge["producer_stick_dim"]
            ),
            "layout_xform_stick_dim": edge["stick_dim"],
            "split_dim": edge["split_dim"],
            "iter_sizes": edge["iter_sizes"],
            "slice_bytes": edge["slice_bytes"],
            "replaces_sdsc": cons_name,
            "tile_index": selected_tile,
            "requested_tile_index": tile_index,
            "compute_tile_count": 1,
        }
    )

    return {
        "artifacts": [producer_artifact, consumer_artifact],
        "replacements": {
            prod_name: pred_sidecar,
            cons_name: cons_sidecar,
        },
        "bundle_attrs": {},
        "pointwise_lx_region0": layout_xform_compose_pointwise_lx_base(
            edge["slice_bytes"]
        ),
        "rejection_reasons": reasons,
    }


def _layout_xform_source_pieces(edge: dict, producer_base: int) -> list[dict] | None:
    return _mapped_work_slice_piece_info(
        _body(edge["producer"]),
        edge["producer_layout"],
        edge["dim_map"],
        edge["iter_sizes"],
        producer_base,
    )


def _operand_region_bytes(
    iter_sizes: dict[str, int],
    stick_dim: str,
    stick_size: int,
    word_length: int = WORD_LENGTH,
) -> int:
    elems = 1
    for dim, size in iter_sizes.items():
        if dim == stick_dim:
            size = _align_up(size, stick_size)
        elems *= size
    return _align_up(elems * word_length, STICK_BYTES)


def _kv_repack_labeled_ds(
    pds_name: str,
    layout: list[str],
    stick_dim: str,
    iter_sizes: dict[str, int],
    pieces: list[dict],
) -> dict:
    return {
        "ldsName_": f"{pds_name}_L0",
        "pdsName_": pds_name,
        "wordLength": WORD_LENGTH,
        "dataformat": DATA_FORMAT,
        "isExternal_": 0,
        "segment_": "output",
        "layoutDimOrder_": list(layout),
        "stickDimOrder_": [stick_dim],
        "dimToLayoutSize_": {dim: iter_sizes[dim] for dim in layout},
        "dimToStickSize_": {stick_dim: STICK_SIZE},
        "validGap_": {dim: [[iter_sizes[dim], 0]] for dim in layout},
        "totElements": -1,
        "hbmSize_": 0,
        "hbmStartAddress_": 0,
        "lxSize_": DATAOP_LX_SIZE,
        "lxStartAddress_": {},
        "PieceInfo": pieces,
    }


def _kv_repack_full_valid_gap(edge: dict, layout: list[str]) -> dict:
    return {
        dim: [[int(edge["iter_sizes"][dim]), 0]]
        for dim in layout
    }


def _kv_repack_broadcast_dst_pieces(
    source_pieces: list[dict],
    consumer_num_cores: int,
    consumer_base: int,
    edge: dict,
    *,
    include_broadcast_metadata: bool = True,
    consumer_core_ids: list[int] | None = None,
    skip_source_core_destinations: bool = False,
) -> list[dict]:
    pieces = []
    ordinal = 1
    core_ids = (
        list(range(consumer_num_cores))
        if consumer_core_ids is None
        else list(consumer_core_ids)
    )
    for core_id in core_ids:
        for source_piece in source_pieces:
            if skip_source_core_destinations and core_id in _piece_lx_mem_ids(
                source_piece
            ):
                continue
            start = copy.deepcopy(source_piece["dimToStartCordinate"])
            size = copy.deepcopy(source_piece["dimToSize_"])
            lx_start_addr = consumer_base + _kv_repack_piece_offset_bytes(edge, start)
            piece = {
                "key_": f"p{ordinal}",
                "dimToStartCordinate": start,
                "dimToSize_": size,
                "validGap_": _kv_repack_full_valid_gap(
                    edge, edge["consumer_layout"]
                ),
                "PlacementInfo": [
                    {
                        "type": "lx",
                        "memId": [core_id],
                        "startAddr": [lx_start_addr],
                    }
                ],
            }
            if include_broadcast_metadata:
                piece["broadcastSourcePieceKey_"] = source_piece.get("key_")
                piece["broadcastConsumerCore_"] = core_id
            pieces.append(piece)
            ordinal += 1
    return pieces


def _piece_lx_mem_ids(piece: dict) -> set[int]:
    mem_ids = set()
    for placement in piece.get("PlacementInfo", []):
        if placement.get("type") == "lx":
            mem_ids.update(int(mem_id) for mem_id in placement.get("memId", []))
    return mem_ids


def _piece_lx_start_by_core(piece: dict) -> dict[int, int]:
    starts: dict[int, int] = {}
    for placement in piece.get("PlacementInfo", []):
        if placement.get("type") != "lx":
            continue
        mem_ids = [int(mem_id) for mem_id in placement.get("memId", [])]
        start_addrs = [int(addr) for addr in placement.get("startAddr", [])]
        for mem_id, start_addr in zip(mem_ids, start_addrs):
            starts[mem_id] = start_addr
    return starts


def _move_lx_piece_start_addrs(pieces: list[dict], lx_base: int) -> None:
    for piece in pieces:
        for placement in piece.get("PlacementInfo", []):
            if placement.get("type") == "lx":
                placement["startAddr"] = [lx_base]


def _kv_repack_single_loader_source_pieces(
    edge: dict,
    *,
    loader_core_id: int,
    lx_base: int,
) -> list[dict]:
    pieces = []
    for source_piece in edge["source_pieces"]:
        piece = copy.deepcopy(source_piece)
        lx_start_addr = lx_base + _kv_repack_piece_offset_bytes(
            edge, piece["dimToStartCordinate"]
        )
        piece["validGap_"] = _kv_repack_full_valid_gap(edge, edge["source_layout"])
        piece["PlacementInfo"] = [
            {
                "type": "lx",
                "memId": [loader_core_id],
                "startAddr": [lx_start_addr],
            }
        ]
        pieces.append(piece)
    return pieces


def _kv_repack_single_loader_full_tile_source_pieces(
    edge: dict,
    *,
    loader_core_id: int,
    lx_base: int,
) -> list[dict]:
    layout = list(edge["source_layout"])
    start = {dim: 0 for dim in layout}
    size = {dim: int(edge["iter_sizes"][dim]) for dim in layout}
    return [
        {
            "key_": "p1",
            "dimToStartCordinate": start,
            "dimToSize_": size,
            "validGap_": {dim: [[size[dim], 0]] for dim in layout},
            "PlacementInfo": [
                {
                    "type": "lx",
                    "memId": [loader_core_id],
                    "startAddr": [lx_base],
                }
            ],
        }
    ]


def _kv_repack_broadcast_dst_piece_count(
    edge: dict,
    *,
    consumer_core_ids: list[int] | None = None,
    skip_source_core_destinations: bool = False,
) -> int:
    core_ids = (
        list(range(edge["consumer_num_cores"]))
        if consumer_core_ids is None
        else list(consumer_core_ids)
    )
    count = 0
    for core_id in core_ids:
        for source_piece in edge["source_pieces"]:
            if skip_source_core_destinations and core_id in _piece_lx_mem_ids(
                source_piece
            ):
                continue
            count += 1
    return count


def _kv_repack_source_core_ids(edge: dict) -> list[int]:
    core_ids = set()
    for source_piece in edge["source_pieces"]:
        core_ids.update(_piece_lx_mem_ids(source_piece))
    return sorted(core_ids)


def _make_kv_repack_broadcast_dataop(
    name: str,
    edge: dict,
    *,
    include_broadcast_metadata: bool = True,
    stcdp_subpiece_reuse: bool = True,
    consumer_core_ids: list[int] | None = None,
    dataop_core_ids: list[int] | None = None,
    skip_source_core_destinations: bool = False,
    stcdp_use_unicast: int = -1,
    stcdp_use_lxsfp_lx_transfers: int = -1,
    stcdp_force_mc_mode: int = -1,
) -> dict:
    source_pieces = edge["source_pieces"]
    dst_pieces = _kv_repack_broadcast_dst_pieces(
        source_pieces,
        edge["consumer_num_cores"],
        edge["consumer_lx_base"],
        edge,
        include_broadcast_metadata=include_broadcast_metadata,
        consumer_core_ids=consumer_core_ids,
        skip_source_core_destinations=skip_source_core_destinations,
    )
    core_ids = (
        list(range(edge["consumer_num_cores"]))
        if dataop_core_ids is None
        else list(dataop_core_ids)
    )
    in_ld = _kv_repack_labeled_ds(
        "dataIN",
        edge["source_layout"],
        edge["stick_dim"],
        edge["iter_sizes"],
        source_pieces,
    )
    out_ld = _kv_repack_labeled_ds(
        "dataOUT",
        edge["consumer_layout"],
        edge["stick_dim"],
        edge["iter_sizes"],
        dst_pieces,
    )
    op = _stcdp_op()
    if not stcdp_subpiece_reuse:
        op["enSubPieceReuse"] = 0
    if stcdp_use_unicast in (0, 1):
        op["useUnicast"] = stcdp_use_unicast
    if stcdp_use_lxsfp_lx_transfers in (0, 1):
        op["useLXSFPLXTransfers"] = stcdp_use_lxsfp_lx_transfers
    if stcdp_force_mc_mode >= 0:
        op["forceModeMC"] = {"force": 1, "val": stcdp_force_mc_mode}
    return {
        name: {
            "coreIdsUsed_": core_ids,
            "dimPool_": list(edge["source_layout"]),
            "outDimTodimRelation_": [],
            "primaryDs_": [
                {"name_": "dataIN", "dimNames": list(edge["source_layout"])},
                {"name_": "dataOUT", "dimNames": list(edge["consumer_layout"])},
            ],
            "labeledDs_": [in_ld, out_ld],
            "op": op,
        }
    }


def _hbm_dataop_addr_units(byte_addr: int) -> int:
    """Convert DL HBM byte addresses to STCDPOpHBM 128-byte address units."""
    if byte_addr % STICK_BYTES != 0:
        raise ValueError(
            f"STCDPOpHBM HBM address must be {STICK_BYTES}-byte aligned: "
            f"{byte_addr}"
        )
    return byte_addr // STICK_BYTES


def _kv_repack_piece_offset_bytes(edge: dict, start: dict[str, int]) -> int:
    """Byte offset for a consumer-layout K/V piece inside its allocation."""
    offset_elems = 0
    stride = 1
    for dim in reversed(edge["consumer_layout"]):
        offset_elems += int(start.get(dim, 0)) * stride
        extent = int(edge["iter_sizes"][dim])
        if dim == edge["stick_dim"]:
            extent = _align_up(extent, STICK_SIZE)
        stride *= extent
    return offset_elems * WORD_LENGTH


def _kv_repack_hbm_piece_offset_units(edge: dict, start: dict[str, int]) -> int:
    """STCDPOpHBM address-unit offset for a consumer-layout K/V HBM piece."""
    return _hbm_dataop_addr_units(_kv_repack_piece_offset_bytes(edge, start))


def _kv_repack_hbm_piece_addr(
    edge: dict,
    source_piece: dict,
    *,
    prefer_source_core_hbm_addr: bool = True,
) -> int:
    source_hbm_start_by_core = edge.get("source_hbm_start_by_core", {})
    source_core_ids = sorted(_piece_lx_mem_ids(source_piece))
    if (
        prefer_source_core_hbm_addr
        and source_core_ids
        and len(set(source_hbm_start_by_core.values())) > 1
    ):
        source_core = source_core_ids[0]
        if source_core in source_hbm_start_by_core:
            return _hbm_dataop_addr_units(int(source_hbm_start_by_core[source_core]))
    start = source_piece["dimToStartCordinate"]
    return _hbm_dataop_addr_units(
        int(edge["shared_hbm_addr"])
    ) + _kv_repack_hbm_piece_offset_units(edge, start)


def _make_hbm_direct_fill_dataop(
    name: str,
    *,
    layout: list[str],
    stick_dim: str,
    iter_sizes: dict[str, int],
    hbm_start_by_core: dict[int, int],
    lx_base: int,
    core_ids: list[int],
) -> dict:
    """Build an STCDPOpHBM direct HBM->LX fill with no LX->LX roundtrip."""
    input_pieces = []
    core_id_to_an_info = {}
    fallback_hbm = next(iter(hbm_start_by_core.values()))
    size = {dim: int(iter_sizes[dim]) for dim in layout}
    start = {dim: 0 for dim in layout}
    gap = {dim: [[size[dim], 0]] for dim in layout}
    for ordinal, core_id in enumerate(core_ids, start=1):
        key = f"p{ordinal}"
        core_id_to_an_info[str(core_id)] = {
            "isAnalyticalMode": 0,
            "inpPieceOrder": [key],
            "outPieceOrder": [key],
        }
        input_pieces.append(
            {
                "key_": key,
                "dimToStartCordinate": copy.deepcopy(start),
                "dimToSize_": copy.deepcopy(size),
                "validGap_": copy.deepcopy(gap),
                "PlacementInfo": [
                    {
                        "type": "lx",
                        "memId": [core_id],
                        "startAddr": [lx_base],
                    },
                    {
                        "type": "hbm",
                        "memId": [-1],
                        "startAddr": [
                            _hbm_dataop_addr_units(
                                int(hbm_start_by_core.get(core_id, fallback_hbm))
                            )
                        ],
                    },
                ],
            }
        )

    input_ld = _kv_repack_labeled_ds(
        "dataIN",
        layout,
        stick_dim,
        iter_sizes,
        input_pieces,
    )
    output_ld = _kv_repack_labeled_ds(
        "dataOUT",
        layout,
        stick_dim,
        iter_sizes,
        [],
    )
    output_ld["pdsName_"] = "dataIN"
    return {
        name: {
            "coreIdsUsed_": list(core_ids),
            "dimPool_": list(layout),
            "outDimTodimRelation_": [],
            "primaryDs_": [
                {"name_": "dataIN", "dimNames": list(layout)},
            ],
            "labeledDs_": [input_ld, output_ld],
            "op": {
                "name": "STCDPOpHBM",
                "reqMulticast": 0,
                "coreIDtoANInfo": core_id_to_an_info,
            },
        }
    }


def _kv_repack_consumer_hbm_prefetch_pieces(edge: dict) -> list[dict]:
    """Consumer-shaped HBM chunks for K/V prefetch into batchmatmul LX."""
    layout = list(edge["consumer_layout"])
    iter_sizes = edge["iter_sizes"]
    chunk_dim = None
    chunk_size = 1

    pieces = []
    starts = [0]
    if chunk_dim is not None:
        starts = list(range(0, int(iter_sizes[chunk_dim]), chunk_size))
    for ordinal, chunk_start in enumerate(starts, start=1):
        start = {dim: 0 for dim in layout}
        size = {dim: int(iter_sizes[dim]) for dim in layout}
        if chunk_dim is not None:
            start[chunk_dim] = chunk_start
            size[chunk_dim] = min(chunk_size, int(iter_sizes[chunk_dim]) - chunk_start)
        lx_start_addr = edge["consumer_lx_base"] + _kv_repack_piece_offset_bytes(
            edge, start
        )
        pieces.append(
            {
                "key_": f"p{ordinal}",
                "dimToStartCordinate": start,
                "dimToSize_": size,
                "validGap_": {dim: [[size[dim], 0]] for dim in size},
                "PlacementInfo": [
                    {
                        "type": "lx",
                        "memId": [0],
                        "startAddr": [lx_start_addr],
                    }
                ],
            }
        )
    return pieces


def _kv_repack_copyback_core(edge: dict, requested_core: int) -> int:
    if requested_core >= 0:
        if requested_core >= edge["consumer_num_cores"]:
            raise ValueError(
                "copyback readback core out of range: "
                f"{requested_core} >= {edge['consumer_num_cores']}"
            )
        return requested_core
    return edge["consumer_num_cores"] - 1


def _make_kv_repack_copyback_hbm_dataop(
    name: str,
    edge: dict,
    *,
    source_piece: dict,
    dataop_core_ids: list[int] | None = None,
    lx_start_by_core: dict[int, int] | None = None,
) -> dict:
    """Build an STCDPOpHBM LX->HBM readback for one consumer-core replica."""
    core_ids = (
        list(range(edge["consumer_num_cores"]))
        if dataop_core_ids is None
        else list(dataop_core_ids)
    )
    start = copy.deepcopy(source_piece["dimToStartCordinate"])
    size = copy.deepcopy(source_piece["dimToSize_"])
    gap = copy.deepcopy(source_piece["validGap_"])
    hbm_addr = _kv_repack_hbm_piece_addr(edge, source_piece)
    input_pieces = []
    hbm_pieces = []
    core_piece_order = {}
    for ordinal, core_id in enumerate(core_ids, start=1):
        key = f"p{ordinal}"
        core_piece_order[core_id] = [key]
        lx_start_addr = (
            lx_start_by_core.get(core_id)
            if lx_start_by_core is not None
            else edge["consumer_lx_base"]
        )
        input_pieces.append(
            {
                "key_": key,
                "dimToStartCordinate": copy.deepcopy(start),
                "dimToSize_": copy.deepcopy(size),
                "validGap_": copy.deepcopy(gap),
                "PlacementInfo": [
                    {
                        "type": "lx",
                        "memId": [core_id],
                        "startAddr": [lx_start_addr],
                    },
                ],
            }
        )
        hbm_pieces.append(
            {
                "key_": key,
                "dimToStartCordinate": copy.deepcopy(start),
                "dimToSize_": copy.deepcopy(size),
                "validGap_": copy.deepcopy(gap),
                "PlacementInfo": [
                    {
                        "type": "lx",
                        "memId": [core_id],
                        "startAddr": [lx_start_addr],
                    },
                    {
                        "type": "hbm",
                        "memId": [-1],
                        "startAddr": [hbm_addr],
                    },
                ],
            }
        )

    input_ld = _kv_repack_labeled_ds(
        "dataIN",
        edge["consumer_layout"],
        edge["stick_dim"],
        edge["iter_sizes"],
        input_pieces,
    )
    output_ld = _kv_repack_labeled_ds(
        "dataOUT",
        edge["consumer_layout"],
        edge["stick_dim"],
        edge["iter_sizes"],
        hbm_pieces,
    )
    core_id_to_an_info = {
        str(core_id): {
            "isAnalyticalMode": 0,
            "inpPieceOrder": core_piece_order[core_id],
            "outPieceOrder": core_piece_order[core_id],
        }
        for core_id in core_ids
    }
    return {
        name: {
            "coreIdsUsed_": core_ids,
            "dimPool_": list(edge["consumer_layout"]),
            "outDimTodimRelation_": [],
            "primaryDs_": [
                {"name_": "dataIN", "dimNames": list(edge["consumer_layout"])},
                {"name_": "dataOUT", "dimNames": list(edge["consumer_layout"])},
            ],
            "labeledDs_": [input_ld, output_ld],
            "op": {
                "name": "STCDPOpHBM",
                "reqMulticast": 0,
                "coreIDtoANInfo": core_id_to_an_info,
            },
        }
    }


def _make_kv_repack_hbm_roundtrip_dataop(
    name: str,
    edge: dict,
    *,
    source_piece: dict,
    phase: str,
    dataop_core_ids: list[int] | None = None,
    lx_start_by_core: dict[int, int] | None = None,
    hbm_piece_addr: bool = True,
    prefer_source_core_hbm_addr: bool = True,
    emit_lx_roundtrip: bool = True,
    corelet_id: int | None = None,
) -> dict:
    """Build one ordered STCDPOpHBM phase for a source-piece roundtrip."""
    if phase not in ("load", "store"):
        raise ValueError(f"unsupported HBM roundtrip phase: {phase}")
    if not emit_lx_roundtrip and phase != "load":
        raise ValueError("direct STCDPOpHBM fill is only valid for load phases")
    core_ids = (
        list(range(edge["consumer_num_cores"]))
        if dataop_core_ids is None
        else list(dataop_core_ids)
    )
    start = copy.deepcopy(source_piece["dimToStartCordinate"])
    size = copy.deepcopy(source_piece["dimToSize_"])
    gap = copy.deepcopy(source_piece["validGap_"])
    hbm_addr = (
        _kv_repack_hbm_piece_addr(
            edge,
            source_piece,
            prefer_source_core_hbm_addr=prefer_source_core_hbm_addr,
        )
        if hbm_piece_addr
        else _hbm_dataop_addr_units(int(edge["shared_hbm_addr"]))
    )
    input_pieces = []
    output_pieces = []
    core_piece_order = {}
    for ordinal, core_id in enumerate(core_ids, start=1):
        key = f"p{ordinal}"
        core_piece_order[core_id] = [key]
        lx_start_addr = (
            lx_start_by_core.get(core_id)
            if lx_start_by_core is not None
            else edge["source_lx_base"]
        )
        lx_placement = {
            "type": "lx",
            "memId": [core_id],
            "startAddr": [lx_start_addr],
        }
        hbm_placement = {
            "type": "hbm",
            "memId": [-1],
            "startAddr": [hbm_addr],
        }
        input_placements = (
            [lx_placement, hbm_placement] if phase == "load" else [lx_placement]
        )
        output_placements = (
            [lx_placement] if phase == "load" else [lx_placement, hbm_placement]
        )
        input_pieces.append(
            {
                "key_": key,
                "dimToStartCordinate": copy.deepcopy(start),
                "dimToSize_": copy.deepcopy(size),
                "validGap_": copy.deepcopy(gap),
                "PlacementInfo": copy.deepcopy(input_placements),
            }
        )
        if emit_lx_roundtrip:
            output_pieces.append(
                {
                    "key_": key,
                    "dimToStartCordinate": copy.deepcopy(start),
                    "dimToSize_": copy.deepcopy(size),
                    "validGap_": copy.deepcopy(gap),
                    "PlacementInfo": copy.deepcopy(output_placements),
                }
            )

    input_ld = _kv_repack_labeled_ds(
        "dataIN",
        edge["consumer_layout"],
        edge["stick_dim"],
        edge["iter_sizes"],
        input_pieces,
    )
    output_ld = _kv_repack_labeled_ds(
        "dataOUT",
        edge["consumer_layout"],
        edge["stick_dim"],
        edge["iter_sizes"],
        output_pieces,
    )
    output_ld["pdsName_"] = "dataIN"
    core_id_to_an_info = {
        str(core_id): {
            "isAnalyticalMode": 0,
            "inpPieceOrder": core_piece_order[core_id],
            "outPieceOrder": core_piece_order[core_id],
        }
        for core_id in core_ids
    }
    op = {
        "name": "STCDPOpHBM",
        "reqMulticast": 0,
        "coreIDtoANInfo": core_id_to_an_info,
    }
    if corelet_id is not None:
        if corelet_id not in (0, 1):
            raise ValueError(f"unsupported STCDPOpHBM corelet_id: {corelet_id}")
        op["coreletId"] = int(corelet_id)

    return {
        name: {
            "coreIdsUsed_": core_ids,
            "dimPool_": list(edge["consumer_layout"]),
            "outDimTodimRelation_": [],
            "primaryDs_": [
                {"name_": "dataIN", "dimNames": list(edge["consumer_layout"])},
            ],
            "labeledDs_": [input_ld, output_ld],
            "op": op,
        }
    }


def _make_nop_dataop(name: str, core_ids: list[int]) -> dict:
    return {
        name: {
            "coreIdsUsed_": list(core_ids),
            "dimPool_": [],
            "outDimTodimRelation_": [],
            "primaryDs_": [],
            "labeledDs_": [],
            "op": {"name": "nop"},
        }
    }


def _kv_repack_copyback_schedule(
    consumer_num_cores: int,
    group_dataop_core_ids: list[list[int]],
    *,
    hbm_dataop_core_ids: list[list[int]],
    barrier_dataop_core_ids: list[int] | None = None,
    include_compute: bool = True,
) -> dict[str, list[list[int]]]:
    return _kv_repack_copyback_schedule_for_dataops(
        consumer_num_cores,
        [*group_dataop_core_ids, *hbm_dataop_core_ids],
        barrier_dataop_core_ids=barrier_dataop_core_ids,
        include_compute=include_compute,
    )


def _kv_repack_copyback_schedule_for_dataops(
    consumer_num_cores: int,
    dataop_core_ids: list[list[int]],
    *,
    barrier_dataop_core_ids: list[int] | None = None,
    include_compute: bool = True,
) -> dict[str, list[list[int]]]:
    dataop_core_sets = [set(core_ids) for core_ids in dataop_core_ids]
    barrier_idx = len(dataop_core_sets)
    barrier_cores = (
        set(barrier_dataop_core_ids)
        if barrier_dataop_core_ids is not None
        else set()
    )
    schedule: dict[str, list[list[int]]] = {}
    for core_id in range(consumer_num_cores):
        local_ops = [
            (dataop_idx, -1)
            for dataop_idx, core_ids in enumerate(dataop_core_sets)
            if core_id in core_ids
        ]
        if core_id in barrier_cores:
            local_ops.append((barrier_idx, -1))
        if include_compute:
            local_ops.append((-1, 0))
        schedule[str(core_id)] = [
            [
                dataop_idx,
                dsc_idx,
                1 if local_idx > 0 else 0,
                1 if local_idx < len(local_ops) - 1 else 0,
            ]
            for local_idx, (dataop_idx, dsc_idx) in enumerate(local_ops)
        ]
    return schedule


def _build_flash_attention_dataop_only_sdsc(
    name: str,
    datadscs: list[dict],
    opfuncs: list[str],
    schedule: dict[str, list[list[int]]],
    num_cores: int,
) -> dict:
    if not name:
        raise ValueError("name is required")
    if num_cores <= 0:
        raise ValueError(f"num_cores must be positive, got {num_cores}")
    data_count = len(datadscs)
    for core_id, rows in schedule.items():
        int(core_id)
        for row in rows:
            if len(row) != 4:
                raise ValueError(f"schedule row must have 4 fields, got {row}")
            data_idx, dsc_idx = int(row[0]), int(row[1])
            if dsc_idx != -1:
                raise ValueError(f"data-only schedule cannot run compute: {row}")
            if data_idx < 0 or data_idx >= data_count:
                raise ValueError(f"data-op index out of range: {row}")
    return {
        name: {
            "numCoresUsed_": num_cores,
            "coreIdToDscSchedule": {
                str(core_id): [list(row) for row in rows]
                for core_id, rows in schedule.items()
            },
            "datadscs_": [dict(dataop) for dataop in datadscs],
            "dscs_": [],
            "opFuncsUsed_": list(opfuncs),
            "flashAttentionPipeline_": {
                "tile_count": 0,
                "dataop_count": data_count,
                "overlap_candidate": False,
            },
        }
    }


def _flash_attention_kv_repack_broadcast_edge(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    input_idx: int,
    allow_additional_consumers: bool = False,
) -> tuple[dict | None, list[str]]:
    if input_idx == INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX:
        return None, [f"input{input_idx}:not_kv_operand"]
    tile = _flash_value_flow_tile(sdscs_json, tile_index)
    if tile is None:
        return None, ["tile_not_found"]
    c, cons = tile
    cons_dl = _dl_op(cons)
    if input_idx not in _consumer_input_indices(cons_dl):
        return None, [f"input{input_idx}:not_consumer_input"]

    cons_body = _body(cons)
    consumer_num_cores = int(cons_body.get("numCoresUsed_", 0))
    if consumer_num_cores <= 0:
        return None, ["invalid_num_cores"]
    addr = _hbm_base(cons_dl, input_idx)
    if addr is None:
        return None, [f"input{input_idx}:not_hbm_backed"]
    producer = _latest_producer_of_hbm(sdscs_json, c, addr)
    if producer is None:
        return None, [f"input{input_idx}:no_latest_producer"]
    p, prod, out_idx = producer

    future = _future_consumers(sdscs_json, p, addr)
    target_consumers = [
        (fc, fcons, fin_idx)
        for fc, fcons, fin_idx in future
        if fc == c and fin_idx == input_idx
    ]
    if (
        len(target_consumers) != 1
        or (not allow_additional_consumers and len(future) != 1)
    ):
        future_names = [
            f"{next(iter(fcons))}:input{fin_idx}"
            for _fc, fcons, fin_idx in future
        ]
        return (
            None,
            [
                f"input{input_idx}:not_single_consumer:"
                f"{','.join(future_names)}"
            ],
        )
    additional_consumers = [
        f"{next(iter(fcons))}:input{fin_idx}"
        for fc, fcons, fin_idx in future
        if fc != c or fin_idx != input_idx
    ]
    if _op_name(prod) != "ReStickifyOpHBM":
        return None, [f"input{input_idx}:producer_not_restickify_hbm:{_op_name(prod)}"]

    prod_dl = _dl_op(prod)
    prod_layout = _layout_for_lds(prod_dl, out_idx)
    prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
    cons_layout = _layout_for_lds(cons_dl, input_idx)
    cons_stick = _stick_dim_for_lds(cons_dl, input_idx)
    prod_shard = _body(prod).get("numWkSlicesPerDim_", {})
    prod_split_dims = _split_dims(prod_shard)
    cons_split = _single_split_dim(cons_body.get("numWkSlicesPerDim_", {}))
    producer_num_cores = int(_body(prod).get("numCoresUsed_", 0))
    if (
        prod_layout is None
        or prod_stick is None
        or cons_layout is None
        or cons_stick is None
        or not prod_split_dims
        or cons_split is None
        or producer_num_cores <= 0
    ):
        return None, [f"input{input_idx}:missing_layout_stick_or_split"]
    if prod_stick != cons_stick:
        return (
            None,
            [
                f"input{input_idx}:stick_transform_required:"
                f"producer={prod_stick}:consumer={cons_stick}"
            ],
        )

    dim_map = _layout_transform_dim_map(
        prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
    )
    if dim_map is None:
        return None, [f"input{input_idx}:layout_transform_dim_map_missing"]
    mapped_split_dims = [dim_map.get(dim) for dim in prod_split_dims]
    iter_sizes = _iter_sizes_for_layout(cons_dl, cons_layout)
    source_layout = [dim_map[dim] for dim in prod_layout]
    if (
        iter_sizes is None
        or any(dim is None or dim not in iter_sizes for dim in mapped_split_dims)
        or any(dim not in iter_sizes for dim in source_layout)
    ):
        return None, [f"input{input_idx}:missing_iter_sizes"]
    if cons_split in iter_sizes:
        return None, [f"input{input_idx}:consumer_split_present:{cons_split}"]
    producer_split_factor = 1
    for prod_dim, mapped_dim in zip(prod_split_dims, mapped_split_dims):
        factor = int(prod_shard.get(prod_dim.removesuffix("_"), 1))
        if factor <= 0 or iter_sizes[mapped_dim] % factor != 0:
            return (
                None,
                [
                    f"input{input_idx}:invalid_producer_split:"
                    f"{mapped_dim}:factor={factor}"
                ],
            )
        producer_split_factor *= factor
    if producer_split_factor != producer_num_cores:
        return (
            None,
            [
                f"input{input_idx}:producer_split_factor_mismatch:"
                f"factor={producer_split_factor}:cores={producer_num_cores}"
            ],
        )
    if producer_num_cores >= consumer_num_cores:
        return (
            None,
            [
                f"input{input_idx}:not_low_core_to_consumer:"
                f"producer={producer_num_cores}:consumer={consumer_num_cores}"
            ],
        )

    slice_bytes = _reserve_bridge_region_bytes(
        _operand_region_bytes(iter_sizes, cons_stick, STICK_SIZE)
    )
    try:
        source_lx_base, consumer_lx_base = allocate_lx_bases(
            2,
            slice_bytes,
            region0=PRODUCER_LX_BASE,
        )
    except ValueError:
        return None, [f"input{input_idx}:lx_allocation_exceeds_capacity"]
    source_pieces = _mapped_work_slice_piece_info(
        _body(prod),
        prod_layout,
        dim_map,
        iter_sizes,
        source_lx_base,
    )
    if source_pieces is None:
        return None, [f"input{input_idx}:producer_piece_map_missing"]

    return (
        {
            "producer_index": p,
            "consumer_index": c,
            "producer": prod,
            "consumer": cons,
            "producer_idx": out_idx,
            "consumer_idx": input_idx,
            "shared_hbm_addr": addr,
            "source_hbm_start_by_core": _hbm_start_by_core(prod_dl, out_idx),
            "producer_layout": prod_layout,
            "producer_stick_dim": prod_stick,
            "consumer_layout": cons_layout,
            "source_layout": source_layout,
            "stick_dim": cons_stick,
            "producer_split": (
                prod_split_dims[0] if len(prod_split_dims) == 1 else prod_split_dims
            ),
            "mapped_split": (
                mapped_split_dims[0]
                if len(mapped_split_dims) == 1
                else mapped_split_dims
            ),
            "consumer_split": cons_split,
            "dim_map": dim_map,
            "iter_sizes": iter_sizes,
            "slice_bytes": slice_bytes,
            "source_lx_base": source_lx_base,
            "consumer_lx_base": consumer_lx_base,
            "producer_num_cores": producer_num_cores,
            "consumer_num_cores": consumer_num_cores,
            "source_pieces": source_pieces,
            "additional_consumers": additional_consumers,
        },
        [],
    )


def flash_attention_kv_repack_broadcast_rejection_reasons(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    input_idx: int,
) -> list[str]:
    """Explain why a K/V low-core-to-32-core repack descriptor cannot be planned."""
    _edge, reasons = _flash_attention_kv_repack_broadcast_edge(
        sdscs_json,
        tile_index,
        input_idx=input_idx,
    )
    return reasons


def _resolve_flash_attention_kv_repack_broadcast_edge(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    input_indices: tuple[int, ...] = (1, 2),
) -> tuple[int | None, int | None, dict | None, list[str]]:
    if tile_index != LAYOUT_XFORM_PAIR_AUTO_TILE:
        reasons: list[str] = []
        for input_idx in input_indices:
            edge, candidate_reasons = _flash_attention_kv_repack_broadcast_edge(
                sdscs_json,
                tile_index,
                input_idx=input_idx,
            )
            if edge is not None:
                return tile_index, input_idx, edge, []
            reasons.extend(candidate_reasons)
        return None, None, None, reasons

    reasons = []
    for candidate in range(_flash_value_flow_tile_count(sdscs_json)):
        for input_idx in input_indices:
            edge, candidate_reasons = _flash_attention_kv_repack_broadcast_edge(
                sdscs_json,
                candidate,
                input_idx=input_idx,
            )
            if edge is not None:
                return candidate, input_idx, edge, []
            reasons.extend(
                f"tile{candidate}:{reason}" for reason in candidate_reasons
            )
    return None, None, None, reasons or ["auto:no_candidate_tiles"]


def flash_attention_kv_repack_broadcast_pair_rejection_reasons(
    sdscs_json: list[dict],
    tile_index: int,
) -> list[str]:
    """Explain why an executable K/V repack pair probe failed closed."""
    _tile, _input_idx, _edge, reasons = (
        _resolve_flash_attention_kv_repack_broadcast_edge(
            sdscs_json,
            tile_index,
        )
    )
    return reasons


def build_flash_attention_kv_repack_broadcast_plan_artifact(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    input_idx: int,
    name_prefix: str = "flash_kv_repack_broadcast_plan",
) -> dict | None:
    """Build a non-executed descriptor artifact for low-core K/V fanout."""
    edge, reasons = _flash_attention_kv_repack_broadcast_edge(
        sdscs_json,
        tile_index,
        input_idx=input_idx,
    )
    if edge is None:
        return None
    name = f"{name_prefix}_{tile_index}_input{input_idx}"
    dataop_name = f"0_STCDPOpLx_kv_repack_broadcast_tile{tile_index}_input{input_idx}"
    dataop = _make_kv_repack_broadcast_dataop(
        dataop_name,
        edge,
    )
    dataop_body = dataop[dataop_name]
    dst_piece_count = len(dataop_body["labeledDs_"][1]["PieceInfo"])
    return {
        name: {
            "numCoresUsed_": edge["consumer_num_cores"],
            "coreIdToDscSchedule": {
                str(core_id): [[0, -1, 0, 0]]
                for core_id in range(edge["consumer_num_cores"])
            },
            "datadscs_": [dataop],
            "dscs_": [],
            "opFuncsUsed_": ["STCDPOpLx"],
            "kvRepackBroadcastPlan_": {
                "runtime_status": "not_executed",
                "blockers": [
                    "Torch-Spyre still needs an executable mixed SDSC owner "
                    "for this producer/consumer boundary",
                    "DXP one-to-many PieceInfo broadcast support is unproven",
                ],
            },
            "flashAttentionPipeline_": {
                "source": "generated-flash-prefill-kv-repack-broadcast-plan",
                "kv_repack_broadcast_plan": True,
                "kv_repack_broadcast_executable": False,
                "kv_repack_runtime_status": "not_executed",
                "kv_repack_source_sdsc": next(iter(edge["producer"])),
                "kv_repack_consumer_sdsc": next(iter(edge["consumer"])),
                "kv_repack_input_idx": edge["consumer_idx"],
                "kv_repack_producer_idx": edge["producer_idx"],
                "kv_repack_consumer_idx": edge["consumer_idx"],
                "kv_repack_shared_hbm_addr": edge["shared_hbm_addr"],
                "kv_repack_producer_cores": edge["producer_num_cores"],
                "kv_repack_consumer_cores": edge["consumer_num_cores"],
                "kv_repack_source_layout": edge["source_layout"],
                "kv_repack_consumer_layout": edge["consumer_layout"],
                "kv_repack_iter_sizes": edge["iter_sizes"],
                "kv_repack_stick_dim": edge["stick_dim"],
                "kv_repack_producer_split": edge["producer_split"],
                "kv_repack_mapped_split": edge["mapped_split"],
                "kv_repack_consumer_split": edge["consumer_split"],
                "kv_repack_source_lx_base": edge["source_lx_base"],
                "kv_repack_consumer_lx_base": edge["consumer_lx_base"],
                "kv_repack_source_piece_count": len(edge["source_pieces"]),
                "kv_repack_destination_piece_count": dst_piece_count,
                "slice_bytes": edge["slice_bytes"],
                "tile_index": tile_index,
                "rejection_reasons": reasons,
            },
        }
    }


def _kv_repack_broadcast_meta(
    edge: dict,
    *,
    source: str,
    tile_index: int,
    requested_tile_index: int,
    input_idx: int,
    executable: bool,
    runtime_forced: bool,
    skip_source_core_destinations: bool = False,
) -> dict:
    return {
        "source": source,
        "kv_repack_broadcast_plan": True,
        "kv_repack_broadcast_executable": executable,
        "kv_repack_runtime_status": "forced_probe" if runtime_forced else "not_executed",
        "kv_repack_runtime_forced": runtime_forced,
        "kv_repack_source_sdsc": next(iter(edge["producer"])),
        "kv_repack_consumer_sdsc": next(iter(edge["consumer"])),
        "kv_repack_input_idx": input_idx,
        "kv_repack_producer_idx": edge["producer_idx"],
        "kv_repack_consumer_idx": edge["consumer_idx"],
        "kv_repack_shared_hbm_addr": edge["shared_hbm_addr"],
        "kv_repack_producer_cores": edge["producer_num_cores"],
        "kv_repack_consumer_cores": edge["consumer_num_cores"],
        "kv_repack_additional_consumers": edge.get("additional_consumers", []),
        "kv_repack_source_layout": edge["source_layout"],
        "kv_repack_consumer_layout": edge["consumer_layout"],
        "kv_repack_iter_sizes": edge["iter_sizes"],
        "kv_repack_stick_dim": edge["stick_dim"],
        "kv_repack_producer_split": edge["producer_split"],
        "kv_repack_mapped_split": edge["mapped_split"],
        "kv_repack_consumer_split": edge["consumer_split"],
        "kv_repack_source_lx_base": edge["source_lx_base"],
        "kv_repack_consumer_lx_base": edge["consumer_lx_base"],
        "kv_repack_source_piece_count": len(edge["source_pieces"]),
        "kv_repack_destination_piece_count": _kv_repack_broadcast_dst_piece_count(
            edge,
            skip_source_core_destinations=skip_source_core_destinations,
        ),
        "slice_bytes": edge["slice_bytes"],
        "tile_index": tile_index,
        "requested_tile_index": requested_tile_index,
    }


def _kv_repack_consumer_core_groups(
    consumer_num_cores: int,
    group_size: int,
) -> list[list[int]]:
    if group_size <= 0 or group_size >= consumer_num_cores:
        return [list(range(consumer_num_cores))]
    return [
        list(range(start, min(start + group_size, consumer_num_cores)))
        for start in range(0, consumer_num_cores, group_size)
    ]


def build_flash_attention_kv_repack_broadcast_pair_artifacts(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    name_prefix: str = "mixed_flash_kv_repack_broadcast_pair",
    include_input_fetch_transfer: bool = True,
    stcdp_subpiece_reuse: bool = True,
    broadcast_group_size: int = 0,
    self_resident_source: bool = False,
    hbm_source: bool = False,
    hbm_direct_load: bool = False,
    hbm_staged: bool = False,
    consumer_core_state_init: bool = True,
    consumer_ds_type: str = "",
    consumer_lx_alloc_style: str = "",
    stcdp_use_unicast: int = -1,
    stcdp_force_mc_mode: int = -1,
) -> dict | None:
    """Build a default-off executable-facing K/V repack producer+consumer pair."""
    selected_tile, selected_input, edge, reasons = (
        _resolve_flash_attention_kv_repack_broadcast_edge(
            sdscs_json,
            tile_index,
        )
    )
    if edge is None or selected_tile is None or selected_input is None:
        return None
    edge = copy.deepcopy(edge)
    if hbm_staged:
        hbm_source = False
        hbm_direct_load = False
    if self_resident_source:
        edge["source_lx_base"] = edge["consumer_lx_base"]
        _move_lx_piece_start_addrs(
            edge["source_pieces"],
            edge["source_lx_base"],
        )

    prod_name = next(iter(edge["producer"]))
    cons_name = next(iter(edge["consumer"]))
    pred_sidecar = f"{name_prefix}_{selected_tile}_input{selected_input}_producer"
    cons_sidecar = f"{name_prefix}_{selected_tile}_input{selected_input}_consumer"

    producer_artifact = None
    uses_original_hbm_source = hbm_source or hbm_direct_load or hbm_staged
    if not uses_original_hbm_source:
        producer_artifact = {pred_sidecar: copy.deepcopy(_body(edge["producer"]))}
        apply_lx_flip(
            producer_artifact,
            LxFlip(
                edge["producer_idx"],
                edge["source_lx_base"],
                "kv-repack-broadcast-producer-output",
            ),
        )
        producer_artifact[pred_sidecar].setdefault("flashAttentionPipeline_", {}).update(
            {
                **_kv_repack_broadcast_meta(
                    edge,
                    source="generated-flash-prefill-kv-repack-broadcast-producer",
                    tile_index=selected_tile,
                    requested_tile_index=tile_index,
                    input_idx=selected_input,
                    executable=True,
                    runtime_forced=True,
                    skip_source_core_destinations=self_resident_source,
                ),
                "kv_repack_broadcast_role": "producer",
                "kv_repack_producer_sidecar": pred_sidecar,
                "kv_repack_consumer_sidecar": cons_sidecar,
                "kv_repack_hbm_source": False,
                "kv_repack_self_resident_source": self_resident_source,
                "kv_repack_stcdp_use_unicast": stcdp_use_unicast,
                "kv_repack_stcdp_force_mc_mode": stcdp_force_mc_mode,
                "replaces_sdsc": prod_name,
            }
        )

    compute_dsc = copy.deepcopy(_body(edge["consumer"])["dscs_"][0])
    compute_root = {cons_name: {"dscs_": [compute_dsc]}}
    compute_dl = next(iter(compute_dsc.values()))
    if not hbm_staged:
        apply_lx_flip(
            compute_root,
            LxFlip(
                edge["consumer_idx"],
                edge["consumer_lx_base"],
                "kv-repack-broadcast-consumer-input",
            ),
            core_state_init=consumer_core_state_init,
        )
        _retarget_lx_allocate_style(
            compute_root,
            edge["consumer_idx"],
            consumer_lx_alloc_style,
        )
        if consumer_ds_type:
            consumer_lds = _lds_by_idx(compute_dl, edge["consumer_idx"])
            if consumer_lds is not None:
                consumer_lds["dsType_"] = consumer_ds_type
    if include_input_fetch_transfer and not hbm_staged:
        _add_input_fetch_neighbor_transfer(compute_dl, edge["consumer_idx"])
        _add_input_fetch_neighbor_legacy_dims(
            compute_dl,
            _body(edge["consumer"]).get("numWkSlicesPerDim_", {}),
        )

    core_groups = _kv_repack_consumer_core_groups(
        edge["consumer_num_cores"],
        broadcast_group_size,
    )
    source_core_ids = _kv_repack_source_core_ids(edge)
    dataops = []
    opfuncs = []
    ordered_dataop_core_groups = []
    group_dataop_core_ids = []
    if hbm_source and not hbm_direct_load:
        for piece_idx, source_piece in enumerate(edge["source_pieces"]):
            hbm_dataop_core_ids = sorted(_piece_lx_mem_ids(source_piece))
            if not hbm_dataop_core_ids:
                return None
            hbm_dataop_name = (
                f"{len(dataops)}_STCDPOpHBM_kv_repack_broadcast_"
                f"Tensor0_idx{edge['consumer_idx']}_tile{selected_tile}"
                f"_hbm_source_piece{piece_idx}_load"
            )
            ordered_dataop_core_groups.append(hbm_dataop_core_ids)
            opfuncs.append("STCDPOpHBM")
            dataops.append(
                _make_kv_repack_hbm_roundtrip_dataop(
                    hbm_dataop_name,
                    edge,
                    source_piece=source_piece,
                    phase="load",
                    dataop_core_ids=hbm_dataop_core_ids,
                    lx_start_by_core=_piece_lx_start_by_core(source_piece),
                )
            )
    if hbm_direct_load:
        direct_load_cores = list(range(edge["consumer_num_cores"]))
        direct_lx_start_by_core = {
            core_id: edge["consumer_lx_base"] for core_id in direct_load_cores
        }
        for piece_idx, source_piece in enumerate(edge["source_pieces"]):
            hbm_dataop_name = (
                f"{len(dataops)}_STCDPOpHBM_kv_repack_broadcast_"
                f"Tensor0_idx{edge['consumer_idx']}_tile{selected_tile}"
                f"_hbm_direct_piece{piece_idx}_load"
            )
            ordered_dataop_core_groups.append(direct_load_cores)
            opfuncs.append("STCDPOpHBM")
            dataops.append(
                _make_kv_repack_hbm_roundtrip_dataop(
                    hbm_dataop_name,
                    edge,
                    source_piece=source_piece,
                    phase="load",
                    dataop_core_ids=direct_load_cores,
                    lx_start_by_core=direct_lx_start_by_core,
                )
            )
    elif not hbm_staged:
        for group_idx, core_ids in enumerate(core_groups):
            group_suffix = "" if len(core_groups) == 1 else f"_group{group_idx}"
            dataop_name = (
                f"{len(dataops)}_STCDPOpLx_kv_repack_broadcast_"
                f"Tensor0_idx{edge['consumer_idx']}_tile{selected_tile}"
                f"{group_suffix}"
            )
            dataop_core_ids = sorted(set(core_ids).union(source_core_ids))
            group_dataop_core_ids.append(dataop_core_ids)
            ordered_dataop_core_groups.append(dataop_core_ids)
            opfuncs.append("STCDPOpLx")
            dataops.append(
                _make_kv_repack_broadcast_dataop(
                    dataop_name,
                    edge,
                    include_broadcast_metadata=False,
                    stcdp_subpiece_reuse=stcdp_subpiece_reuse,
                    consumer_core_ids=core_ids,
                    dataop_core_ids=dataop_core_ids,
                    skip_source_core_destinations=self_resident_source,
                    stcdp_use_unicast=stcdp_use_unicast,
                    stcdp_force_mc_mode=stcdp_force_mc_mode,
                )
            )
    barrier_dataop_core_ids = None
    if uses_original_hbm_source:
        barrier_dataop_core_ids = list(range(edge["consumer_num_cores"]))
        barrier_dataop_name = (
            f"{len(dataops)}_nop_kv_repack_broadcast_barrier_"
            f"Tensor0_idx{edge['consumer_idx']}_tile{selected_tile}"
        )
        dataops.append(_make_nop_dataop(barrier_dataop_name, barrier_dataop_core_ids))
        opfuncs.append("nop")
    schedule = _kv_repack_copyback_schedule_for_dataops(
        edge["consumer_num_cores"],
        ordered_dataop_core_groups,
        barrier_dataop_core_ids=barrier_dataop_core_ids,
        include_compute=True,
    )
    consumer_artifact = build_flash_attention_pipeline_mixed_sdsc(
        cons_sidecar,
        dataops,
        opfuncs,
        schedule,
        [compute_dsc],
        edge["consumer_num_cores"],
    )
    consumer_root = consumer_artifact[cons_sidecar]
    cons_body = _body(edge["consumer"])
    for key in (
        "sdscFoldProps_",
        "sdscFolds_",
        "coreFoldProp_",
        "coreletFoldProp_",
        "coreIdToDsc_",
        "numWkSlicesPerDim_",
        "coreIdToWkSlice_",
    ):
        if key in cons_body:
            consumer_root[key] = copy.deepcopy(cons_body[key])
    consumer_root["flashAttentionPipeline_"].update(
        {
            **_kv_repack_broadcast_meta(
                edge,
                source="generated-flash-prefill-kv-repack-broadcast-consumer",
                tile_index=selected_tile,
                requested_tile_index=tile_index,
                input_idx=selected_input,
                executable=True,
                runtime_forced=True,
                skip_source_core_destinations=self_resident_source,
            ),
            "kv_repack_broadcast_role": "consumer",
            "kv_repack_producer_sidecar": (
                None if uses_original_hbm_source else pred_sidecar
            ),
            "kv_repack_consumer_sidecar": cons_sidecar,
            "kv_repack_input_fetch_transfer": include_input_fetch_transfer,
            "kv_repack_stcdp_subpiece_reuse": stcdp_subpiece_reuse,
            "kv_repack_broadcast_group_size": broadcast_group_size,
            "kv_repack_broadcast_group_count": (
                0 if hbm_direct_load or hbm_staged else len(core_groups)
            ),
            "kv_repack_hbm_source": uses_original_hbm_source,
            "kv_repack_hbm_direct_load": hbm_direct_load,
            "kv_repack_hbm_staged": hbm_staged,
            "kv_repack_consumer_core_state_init": consumer_core_state_init,
            "kv_repack_consumer_ds_type": consumer_ds_type,
            "kv_repack_consumer_lx_alloc_style": consumer_lx_alloc_style,
            "kv_repack_self_resident_source": self_resident_source,
            "kv_repack_stcdp_use_unicast": stcdp_use_unicast,
            "kv_repack_stcdp_force_mc_mode": stcdp_force_mc_mode,
            "replaces_sdsc": cons_name,
            "compute_tile_count": 1,
        }
    )

    artifacts = [consumer_artifact] if producer_artifact is None else [
        producer_artifact,
        consumer_artifact,
    ]
    replacements = {}
    if producer_artifact is not None:
        replacements[prod_name] = pred_sidecar
    replacements[cons_name] = cons_sidecar
    return {
        "artifacts": artifacts,
        "replacements": replacements,
        "bundle_attrs": {},
        "pointwise_lx_region0": edge["consumer_lx_base"] + edge["slice_bytes"],
        "rejection_reasons": reasons,
    }


def _flash_attention_kv_repack_hbm_staged_hoist_edges(
    sdscs_json: list[dict],
    current_tile: int,
) -> tuple[dict | None, list[str]]:
    current = _flash_value_flow_tile(sdscs_json, current_tile)
    if current is None:
        return None, ["current:tile_not_found"]
    current_consumer_index, current_consumer = current
    current_num_cores = int(_body(current_consumer).get("numCoresUsed_", 0))
    if current_num_cores <= 0:
        return None, ["current:invalid_num_cores"]

    reasons: list[str] = []
    for future_tile in range(current_tile + 1, _flash_value_flow_tile_count(sdscs_json)):
        future_reasons: list[str] = []
        for input_idx in (1, 2):
            edge, candidate_reasons = _flash_attention_kv_repack_broadcast_edge(
                sdscs_json,
                future_tile,
                input_idx=input_idx,
                allow_additional_consumers=True,
            )
            if edge is None:
                future_reasons.extend(candidate_reasons)
                continue
            future_producer_index = edge["producer_index"]
            future_consumer_index = edge["consumer_index"]
            if future_producer_index <= current_consumer_index:
                future_reasons.append(
                    f"input{input_idx}:producer_already_ready:"
                    f"producer={future_producer_index}:"
                    f"current={current_consumer_index}"
                )
                continue
            if future_consumer_index <= future_producer_index:
                future_reasons.append(
                    f"input{input_idx}:consumer_not_after_producer:"
                    f"consumer={future_consumer_index}:"
                    f"producer={future_producer_index}"
                )
                continue
            if edge["consumer_num_cores"] != current_num_cores:
                future_reasons.append(
                    f"input{input_idx}:consumer_num_cores_mismatch:"
                    f"current={current_num_cores}:"
                    f"future={edge['consumer_num_cores']}"
                )
                continue
            hoist_reasons = _hoistable_before(
                sdscs_json,
                future_producer_index,
                current_consumer_index,
            )
            if hoist_reasons:
                future_reasons.extend(
                    f"input{input_idx}:{reason}" for reason in hoist_reasons
                )
                continue
            participant_names = [
                next(iter(current_consumer)),
                next(iter(edge["producer"])),
                next(iter(edge["consumer"])),
            ]
            if len(set(participant_names)) != len(participant_names):
                future_reasons.append(
                    f"input{input_idx}:duplicate_participants:"
                    f"{','.join(participant_names)}"
                )
                continue
            source_core_ids = _kv_repack_source_core_ids(edge)
            if not source_core_ids:
                future_reasons.append(f"input{input_idx}:missing_source_cores")
                continue
            return (
                {
                    "current_tile": current_tile,
                    "future_tile": future_tile,
                    "current_consumer_index": current_consumer_index,
                    "current_consumer": current_consumer,
                    "future_edge": edge,
                    "future_input_idx": input_idx,
                    "num_cores": current_num_cores,
                    "source_core_ids": source_core_ids,
                    "slice_bytes": edge["slice_bytes"],
                },
                [],
            )
        reasons.extend(
            f"future_tile{future_tile}:{reason}" for reason in future_reasons
        )
    return None, reasons or ["hoist:no_future_kv_candidate"]


def _resolve_flash_attention_kv_repack_hbm_staged_hoist_edges(
    sdscs_json: list[dict],
    tile_index: int,
) -> tuple[dict | None, list[str]]:
    if tile_index != LAYOUT_XFORM_PAIR_AUTO_TILE:
        return _flash_attention_kv_repack_hbm_staged_hoist_edges(
            sdscs_json,
            tile_index,
        )

    reasons: list[str] = []
    for candidate in range(_flash_value_flow_tile_count(sdscs_json)):
        result, candidate_reasons = (
            _flash_attention_kv_repack_hbm_staged_hoist_edges(
                sdscs_json,
                candidate,
            )
        )
        if result is not None:
            return result, []
        reasons.extend(
            f"tile{candidate}:{reason}" for reason in candidate_reasons
        )
    return None, reasons or ["auto:no_candidate_tiles"]


def flash_attention_kv_repack_hbm_staged_hoist_rejection_reasons(
    sdscs_json: list[dict],
    tile_index: int,
) -> list[str]:
    """Explain why a future K/V HBM-staged hoist failed closed."""
    _result, reasons = _resolve_flash_attention_kv_repack_hbm_staged_hoist_edges(
        sdscs_json,
        tile_index,
    )
    return reasons


def build_flash_attention_kv_repack_hbm_staged_hoist_tile_artifacts(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    name_prefix: str = "mixed_flash_kv_repack_hbm_staged_hoist",
) -> dict | None:
    """Hoist a future low-core K/V producer while keeping HBM-staged consume."""
    hoist, reasons = _resolve_flash_attention_kv_repack_hbm_staged_hoist_edges(
        sdscs_json,
        tile_index,
    )
    if hoist is None:
        return None

    current = hoist["current_consumer"]
    edge = hoist["future_edge"]
    current_tile = hoist["current_tile"]
    future_tile = hoist["future_tile"]
    current_name = next(iter(current))
    future_prod_name = next(iter(edge["producer"]))
    producer_sidecar = f"{name_prefix}_{current_tile}_future_producer"

    producer_artifact = {producer_sidecar: copy.deepcopy(_body(edge["producer"]))}
    producer_root = producer_artifact[producer_sidecar]
    producer_root.setdefault("flashAttentionPipeline_", {}).update(
        {
            "source": "generated-flash-prefill-kv-hbm-staged-hoisted-producer",
            "kv_repack_hbm_staged_hoist_role": "future_producer",
            "kv_repack_hbm_staged_hoist_experimental": True,
            "kv_repack_hbm_staged_hoist_runtime_safe": True,
            "kv_repack_hbm_staged_hoist_runtime_forced": True,
            "kv_repack_hbm_staged_hoist_current_tile": current_tile,
            "kv_repack_hbm_staged_hoist_future_tile": future_tile,
            "kv_repack_hbm_staged_hoist_future_input_idx": hoist[
                "future_input_idx"
            ],
            "kv_repack_hbm_staged_hoist_future_predecessor_sdsc": (
                future_prod_name
            ),
            "kv_repack_hbm_staged_hoist_omitted_future_predecessor_sdsc": (
                future_prod_name
            ),
            "kv_repack_hbm_staged_hoist_inserted_before_sdsc": current_name,
            "kv_repack_hbm_staged_hoist_source_core_ids": (
                hoist["source_core_ids"]
            ),
            "replaces_sdsc": future_prod_name,
            "tile_index": current_tile,
            "requested_tile_index": tile_index,
            "compute_tile_count": 1,
        }
    )

    future_consumer = build_flash_attention_kv_repack_broadcast_pair_artifacts(
        sdscs_json,
        future_tile,
        name_prefix=f"{name_prefix}_{current_tile}_future_kv",
        include_input_fetch_transfer=False,
        hbm_staged=True,
    )
    if future_consumer is None:
        return None

    return {
        "artifacts": [producer_artifact, *future_consumer["artifacts"]],
        "replacements": {
            **future_consumer["replacements"],
        },
        "insertions_before": {current_name: [producer_sidecar]},
        "omissions": {future_prod_name},
        "bundle_attrs": {},
        "pointwise_lx_region0": future_consumer["pointwise_lx_region0"],
        "rejection_reasons": reasons,
    }


def _kv_repack_hbm_prefetch_overlap_schedule(
    num_cores: int,
    dataop_count: int,
) -> dict[str, list[list[int]]]:
    """Schedule the first future K/V HBM prefetch concurrently with compute."""
    if dataop_count <= 0:
        return {str(core_id): [[-1, 0, 0, 0]] for core_id in range(num_cores)}
    rows = [[0, 0, 0, 1]]
    rows.extend(
        [
            idx,
            -1,
            1,
            1,
        ]
        for idx in range(1, dataop_count)
    )
    return {str(core_id): [list(row) for row in rows] for core_id in range(num_cores)}


def _kv_repack_hbm_prefetch_native_load_overlap_schedule(
    num_cores: int,
    dataop_count: int,
    *,
    overlap_after_sync: bool = True,
) -> dict[str, list[list[int]]]:
    """Run native DLDSC input loads in a prologue, then overlap prefetch."""
    if dataop_count <= 1:
        return {str(core_id): [[0, -1, 0, 0]] for core_id in range(num_cores)}
    rows = [[0, -1, 0, 1], [1, 0, 1, 1 if overlap_after_sync else 0]]
    rows.extend(
        [
            idx,
            -1,
            1,
            1,
        ]
        for idx in range(2, dataop_count)
    )
    return {str(core_id): [list(row) for row in rows] for core_id in range(num_cores)}


def _kv_repack_hbm_prefetch_tail_current_schedule(
    num_cores: int,
    pre_compute_dataop_count: int,
    tail_dataop_count: int,
) -> dict[str, list[list[int]]]:
    """Run current compute before issuing the future K/V prefetch dataops."""
    rows = [
        [
            idx,
            -1,
            1 if idx > 0 else 0,
            1,
        ]
        for idx in range(pre_compute_dataop_count)
    ]
    rows.append(
        [
            -1,
            0,
            1 if pre_compute_dataop_count > 0 else 0,
            1 if tail_dataop_count > 0 else 0,
        ]
    )
    first_tail = pre_compute_dataop_count
    rows.extend(
        [
            first_tail + idx,
            -1,
            1,
            1 if idx < tail_dataop_count - 1 else 0,
        ]
        for idx in range(tail_dataop_count)
    )
    return {str(core_id): [list(row) for row in rows] for core_id in range(num_cores)}


def _kv_repack_hbm_prefetch_source_fanout_schedule(
    num_cores: int,
    pre_compute_dataop_count: int,
    source_load_core_ids: list[list[int]],
    barrier_dataop_idx: int,
    fanout_dataop_idx: int,
    fanout_core_ids: list[int] | None = None,
    post_fanout_dataop_idx: int | None = None,
    post_fanout_core_ids: list[int] | None = None,
    overlap_after_sync: bool = True,
    serialize_load_core_compute: bool = False,
) -> dict[str, list[list[int]]]:
    """Overlap loader-core HBM source loads, then fan out after compute."""
    source_load_sets = [set(core_ids) for core_ids in source_load_core_ids]
    fanout_core_set = (
        set(range(num_cores)) if fanout_core_ids is None else set(fanout_core_ids)
    )
    post_fanout_core_set = set(post_fanout_core_ids or [])
    schedule: dict[str, list[list[int]]] = {}
    for core_id in range(num_cores):
        rows = [
            [
                idx,
                -1,
                1 if idx > 0 else 0,
                1,
            ]
            for idx in range(pre_compute_dataop_count)
        ]
        local_loads = [
            pre_compute_dataop_count + idx
            for idx, core_ids in enumerate(source_load_sets)
            if core_id in core_ids
        ]
        if serialize_load_core_compute and local_loads:
            rows.extend(
                [
                    load_idx,
                    -1,
                    1 if rows or load_idx != local_loads[0] else 0,
                    1,
                ]
                for load_idx in local_loads
            )
            rows.append([-1, 0, 1, 1])
        else:
            first_load = local_loads[0] if local_loads else -1
            rows.append(
                [
                    first_load,
                    0,
                    1 if rows else 0,
                    1 if overlap_after_sync else 0,
                ]
            )
            rows.extend(
                [
                    load_idx,
                    -1,
                    1,
                    1,
                ]
                for load_idx in local_loads[1:]
            )
        runs_fanout = core_id in fanout_core_set
        runs_post_fanout = (
            post_fanout_dataop_idx is not None and core_id in post_fanout_core_set
        )
        rows.append([barrier_dataop_idx, -1, 1, 1 if runs_fanout else 0])
        if runs_fanout:
            rows.append([fanout_dataop_idx, -1, 1, 1 if runs_post_fanout else 0])
        if runs_post_fanout:
            rows.append([post_fanout_dataop_idx, -1, 1, 0])
        schedule[str(core_id)] = [list(row) for row in rows]
    return schedule


def _kv_repack_hbm_prefetch_source_fanout_tail_schedule(
    num_cores: int,
    pre_compute_dataop_count: int,
    source_load_core_ids: list[list[int]],
    barrier_dataop_idx: int,
    fanout_dataop_idx: int,
    fanout_core_ids: list[int] | None = None,
    post_fanout_dataop_idx: int | None = None,
    post_fanout_core_ids: list[int] | None = None,
) -> dict[str, list[list[int]]]:
    """Run compute before loader-core HBM source loads and final fanout."""
    source_load_sets = [set(core_ids) for core_ids in source_load_core_ids]
    fanout_core_set = (
        set(range(num_cores)) if fanout_core_ids is None else set(fanout_core_ids)
    )
    post_fanout_core_set = set(post_fanout_core_ids or [])
    schedule: dict[str, list[list[int]]] = {}
    for core_id in range(num_cores):
        rows = [
            [
                idx,
                -1,
                1 if idx > 0 else 0,
                1,
            ]
            for idx in range(pre_compute_dataop_count)
        ]
        local_loads = [
            pre_compute_dataop_count + idx
            for idx, core_ids in enumerate(source_load_sets)
            if core_id in core_ids
        ]
        rows.append(
            [
                -1,
                0,
                1 if rows else 0,
                1,
            ]
        )
        rows.extend(
            [
                load_idx,
                -1,
                1,
                1,
            ]
            for load_idx in local_loads
        )
        runs_fanout = core_id in fanout_core_set
        runs_post_fanout = (
            post_fanout_dataop_idx is not None and core_id in post_fanout_core_set
        )
        rows.append([barrier_dataop_idx, -1, 1, 1 if runs_fanout else 0])
        if runs_fanout:
            rows.append([fanout_dataop_idx, -1, 1, 1 if runs_post_fanout else 0])
        if runs_post_fanout:
            rows.append([post_fanout_dataop_idx, -1, 1, 0])
        schedule[str(core_id)] = [list(row) for row in rows]
    return schedule


def _kv_repack_hbm_prefetch_loader_copyback_schedule(
    num_cores: int,
    pre_compute_dataop_count: int,
    source_load_core_ids: list[list[int]],
    barrier_dataop_idx: int,
    copyback_dataop_idx: int,
    copyback_core_ids: list[int],
    overlap_after_sync: bool = True,
    serialize_load_core_compute: bool = False,
) -> dict[str, list[list[int]]]:
    """Overlap loader-core HBM source loads, then copy loader LX back to HBM."""
    source_load_sets = [set(core_ids) for core_ids in source_load_core_ids]
    copyback_core_set = set(copyback_core_ids)
    schedule: dict[str, list[list[int]]] = {}
    for core_id in range(num_cores):
        rows = [
            [
                idx,
                -1,
                1 if idx > 0 else 0,
                1,
            ]
            for idx in range(pre_compute_dataop_count)
        ]
        local_loads = [
            pre_compute_dataop_count + idx
            for idx, core_ids in enumerate(source_load_sets)
            if core_id in core_ids
        ]
        if serialize_load_core_compute and local_loads:
            rows.extend(
                [
                    load_idx,
                    -1,
                    1 if rows or load_idx != local_loads[0] else 0,
                    1,
                ]
                for load_idx in local_loads
            )
            rows.append([-1, 0, 1, 1])
        else:
            first_load = local_loads[0] if local_loads else -1
            rows.append(
                [
                    first_load,
                    0,
                    1 if rows else 0,
                    1 if overlap_after_sync else 0,
                ]
            )
            rows.extend(
                [
                    load_idx,
                    -1,
                    1,
                    1,
                ]
                for load_idx in local_loads[1:]
            )
        runs_copyback = core_id in copyback_core_set
        rows.append([barrier_dataop_idx, -1, 1, 1 if runs_copyback else 0])
        if runs_copyback:
            rows.append([copyback_dataop_idx, -1, 1, 0])
        schedule[str(core_id)] = [list(row) for row in rows]
    return schedule


def _kv_repack_hbm_prefetch_loader_copyback_tail_schedule(
    num_cores: int,
    pre_compute_dataop_count: int,
    source_load_core_ids: list[list[int]],
    barrier_dataop_idx: int,
    copyback_dataop_idx: int,
    copyback_core_ids: list[int],
) -> dict[str, list[list[int]]]:
    """Run compute before loader-core HBM source loads and direct copyback."""
    source_load_sets = [set(core_ids) for core_ids in source_load_core_ids]
    copyback_core_set = set(copyback_core_ids)
    schedule: dict[str, list[list[int]]] = {}
    for core_id in range(num_cores):
        rows = [
            [
                idx,
                -1,
                1 if idx > 0 else 0,
                1,
            ]
            for idx in range(pre_compute_dataop_count)
        ]
        local_loads = [
            pre_compute_dataop_count + idx
            for idx, core_ids in enumerate(source_load_sets)
            if core_id in core_ids
        ]
        rows.append(
            [
                -1,
                0,
                1 if rows else 0,
                1,
            ]
        )
        rows.extend(
            [
                load_idx,
                -1,
                1,
                1,
            ]
            for load_idx in local_loads
        )
        runs_copyback = core_id in copyback_core_set
        rows.append([barrier_dataop_idx, -1, 1, 1 if runs_copyback else 0])
        if runs_copyback:
            rows.append([copyback_dataop_idx, -1, 1, 0])
        schedule[str(core_id)] = [list(row) for row in rows]
    return schedule


def _kv_repack_hbm_prefetch_prefill_overlap_schedule(
    num_cores: int,
    prefill_dataop_count: int,
    overlap_dataop_count: int,
) -> dict[str, list[list[int]]]:
    """Serialize current-input prefill, then overlap future prefetch with compute."""
    if prefill_dataop_count <= 0:
        return _kv_repack_hbm_prefetch_overlap_schedule(
            num_cores,
            overlap_dataop_count,
        )
    if overlap_dataop_count <= 0:
        rows = [
            [
                idx,
                -1,
                1 if idx > 0 else 0,
                1,
            ]
            for idx in range(prefill_dataop_count)
        ]
        rows.append([-1, 0, 1, 0])
        return {str(core_id): [list(row) for row in rows] for core_id in range(num_cores)}

    rows = [
        [
            idx,
            -1,
            1 if idx > 0 else 0,
            1,
        ]
        for idx in range(prefill_dataop_count)
    ]
    first_overlap = prefill_dataop_count
    rows.append(
        [
            first_overlap,
            0,
            1,
            1,
        ]
    )
    rows.extend(
        [
            first_overlap + idx,
            -1,
            1,
            1,
        ]
        for idx in range(1, overlap_dataop_count)
    )
    return {str(core_id): [list(row) for row in rows] for core_id in range(num_cores)}


def flash_attention_kv_repack_hbm_prefetch_hoist_rejection_reasons(
    sdscs_json: list[dict],
    tile_index: int,
) -> list[str]:
    """Explain why a future K/V HBM prefetch hoist failed closed."""
    return flash_attention_kv_repack_hbm_staged_hoist_rejection_reasons(
        sdscs_json,
        tile_index,
    )


def build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    name_prefix: str = "mixed_flash_kv_repack_hbm_prefetch_hoist",
    prefetch_lx_base: int | None = None,
    serial_prefetch: bool = False,
    prefill_current_input: bool = False,
    redundant_future_prefetch: bool = False,
    serialize_current_prefetch: bool = False,
    external_future_prefetch: bool = False,
    overlap_after_sync: bool = True,
    tail_current_prefetch: bool = False,
    prefetch_source_fanout: bool = False,
    prefetch_loader_fanout: bool = False,
    prefetch_loader_core_id: int = 0,
    prefetch_loader_lx_base: int = -1,
    prefetch_fanout_use_unicast: int = -1,
    prefetch_fanout_use_lxsfp_lx_transfers: int = -1,
    prefetch_fanout_copyback_core: int = -2,
    prefetch_fanout_restrict_to_copyback_core: bool = False,
    prefetch_loader_copyback_without_fanout: bool = False,
    prefetch_loader_fanout_full_tile_pieces: bool = False,
    prefetch_lx_roundtrip: bool = False,
    prefetch_corelet_id: int | None = None,
    serialize_loader_core_prefetch: bool = False,
) -> dict | None:
    """Hoist a future K/V producer and prefetch its HBM result during compute."""
    hoist, reasons = _resolve_flash_attention_kv_repack_hbm_staged_hoist_edges(
        sdscs_json,
        tile_index,
    )
    if hoist is None:
        return None

    current = hoist["current_consumer"]
    edge = copy.deepcopy(hoist["future_edge"])
    current_tile = hoist["current_tile"]
    future_tile = hoist["future_tile"]
    current_name = next(iter(current))
    future_prod_name = next(iter(edge["producer"]))
    future_cons_name = next(iter(edge["consumer"]))
    producer_sidecar = f"{name_prefix}_{current_tile}_future_producer"
    current_sidecar = f"{name_prefix}_{current_tile}_current_prefetch"
    future_cons_sidecar = f"{name_prefix}_{current_tile}_future_consumer"
    original_consumer_lx_base = edge["consumer_lx_base"]
    if prefetch_lx_base is not None:
        edge["consumer_lx_base"] = prefetch_lx_base
    fanout_copyback_enabled = prefetch_fanout_copyback_core >= -1
    if prefetch_loader_copyback_without_fanout and not fanout_copyback_enabled:
        return None
    fanout_copyback_core = (
        _kv_repack_copyback_core(edge, prefetch_fanout_copyback_core)
        if fanout_copyback_enabled
        else -1
    )
    restricted_fanout_core_ids = (
        [fanout_copyback_core]
        if fanout_copyback_enabled and prefetch_fanout_restrict_to_copyback_core
        else None
    )

    current_prefill_inputs = []
    if prefill_current_input and not serial_prefetch:
        current_dl = _dl_op(current)
        current_region_bytes = edge["slice_bytes"]
        for lds_idx in _consumer_input_indices(current_dl):
            hbm_start_by_core = _hbm_start_by_core(current_dl, lds_idx)
            if not hbm_start_by_core:
                continue
            layout = _layout_for_lds(current_dl, lds_idx)
            stick_dim = _stick_dim_for_lds(current_dl, lds_idx)
            iter_sizes = (
                _local_iter_sizes_for_layout(current_dl, layout)
                if layout is not None
                else None
            )
            if layout is None or stick_dim is None or iter_sizes is None:
                continue
            current_region_bytes = max(
                current_region_bytes,
                _reserve_bridge_region_bytes(
                    _operand_region_bytes(iter_sizes, stick_dim, STICK_SIZE)
                ),
            )
            current_prefill_inputs.append(
                {
                    "lds_idx": lds_idx,
                    "layout": layout,
                    "stick_dim": stick_dim,
                    "iter_sizes": iter_sizes,
                    "hbm_start_by_core": hbm_start_by_core,
                }
            )
        if current_prefill_inputs:
            bases = allocate_lx_bases(
                len(current_prefill_inputs) + 2,
                current_region_bytes,
                region0=PRODUCER_LX_BASE,
            )
            if prefetch_lx_base is None:
                edge["consumer_lx_base"] = bases[1]
            for item, lx_base in zip(current_prefill_inputs, bases[2:]):
                item["lx_base"] = lx_base

    producer_artifact = {producer_sidecar: copy.deepcopy(_body(edge["producer"]))}
    producer_root = producer_artifact[producer_sidecar]
    producer_root.setdefault("flashAttentionPipeline_", {}).update(
        {
            "source": "generated-flash-prefill-kv-hbm-prefetch-hoisted-producer",
            "kv_repack_hbm_prefetch_hoist_role": "future_producer",
            "kv_repack_hbm_prefetch_hoist_experimental": True,
            "kv_repack_hbm_prefetch_hoist_runtime_safe": True,
            "kv_repack_hbm_prefetch_hoist_runtime_forced": True,
            "kv_repack_hbm_prefetch_hoist_current_tile": current_tile,
            "kv_repack_hbm_prefetch_hoist_future_tile": future_tile,
            "kv_repack_hbm_prefetch_hoist_future_input_idx": hoist[
                "future_input_idx"
            ],
            "kv_repack_hbm_prefetch_hoist_future_predecessor_sdsc": (
                future_prod_name
            ),
            "kv_repack_hbm_prefetch_hoist_omitted_future_predecessor_sdsc": (
                future_prod_name
            ),
            "kv_repack_hbm_prefetch_hoist_inserted_before_sdsc": current_name,
            "kv_repack_hbm_prefetch_hoist_source_core_ids": (
                hoist["source_core_ids"]
            ),
            "replaces_sdsc": future_prod_name,
            "tile_index": current_tile,
            "requested_tile_index": tile_index,
            "compute_tile_count": 1,
        }
    )

    direct_load_cores = list(range(edge["consumer_num_cores"]))
    dataops = []
    opfuncs = []
    source_fanout_load_core_ids = []
    source_fanout_load_count = 0
    fanout_copyback_dataop_idx = None
    fanout_copyback_core_ids = None
    fanout_core_ids = None
    loader_lx_base = edge["source_lx_base"]
    if prefetch_loader_fanout:
        loader_core_id = int(prefetch_loader_core_id)
        if loader_core_id < 0 or loader_core_id >= edge["consumer_num_cores"]:
            raise ValueError(
                "loader fanout core out of range: "
                f"{loader_core_id} >= {edge['consumer_num_cores']}"
            )
        if prefetch_loader_lx_base == -2:
            loader_lx_base = edge["consumer_lx_base"] + edge["slice_bytes"]
        elif prefetch_loader_lx_base >= 0:
            loader_lx_base = int(prefetch_loader_lx_base)
        elif prefetch_loader_lx_base != -1:
            raise ValueError(
                "unsupported loader fanout LX base selector: "
                f"{prefetch_loader_lx_base}"
            )
        if loader_lx_base + edge["slice_bytes"] > LX_CAPACITY_BYTES:
            raise ValueError(
                "loader fanout LX source buffer exceeds capacity: "
                f"base={loader_lx_base} bytes={edge['slice_bytes']} "
                f"capacity={LX_CAPACITY_BYTES}"
            )
        loader_edge = copy.deepcopy(edge)
        loader_edge["source_pieces"] = (
            _kv_repack_single_loader_full_tile_source_pieces(
                edge,
                loader_core_id=loader_core_id,
                lx_base=loader_lx_base,
            )
            if prefetch_loader_fanout_full_tile_pieces
            else _kv_repack_single_loader_source_pieces(
                edge,
                loader_core_id=loader_core_id,
                lx_base=loader_lx_base,
            )
        )
        prefetch_pieces = _kv_repack_consumer_hbm_prefetch_pieces(edge)
        if not prefetch_pieces:
            return None
        hbm_dataop_name = (
            f"{len(dataops)}_STCDPOpHBM_kv_repack_hbm_prefetch_"
            f"Tensor0_idx{edge['consumer_idx']}_tile{future_tile}"
            f"_loader_core{loader_core_id}_load"
        )
        dataops.append(
            _make_kv_repack_hbm_roundtrip_dataop(
                hbm_dataop_name,
                edge,
                source_piece=prefetch_pieces[0],
                phase="load",
                dataop_core_ids=[loader_core_id],
                lx_start_by_core={loader_core_id: loader_lx_base},
                hbm_piece_addr=True,
                prefer_source_core_hbm_addr=False,
                emit_lx_roundtrip=prefetch_lx_roundtrip,
                corelet_id=prefetch_corelet_id,
            )
        )
        opfuncs.append("STCDPOpHBM")
        source_fanout_load_core_ids.append([loader_core_id])
        source_fanout_load_count = len(dataops)
        barrier_dataop_name = (
            f"{len(dataops)}_nop_kv_repack_hbm_prefetch_"
            f"Tensor0_idx{edge['consumer_idx']}_tile{future_tile}"
            "_loader_fanout_barrier"
        )
        dataops.append(
            _make_nop_dataop(
                barrier_dataop_name,
                list(range(edge["consumer_num_cores"])),
            )
        )
        opfuncs.append("nop")
        if not prefetch_loader_copyback_without_fanout:
            fanout_core_ids = restricted_fanout_core_ids or list(
                range(edge["consumer_num_cores"])
            )
            fanout_dataop_name = (
                f"{len(dataops)}_STCDPOpLx_kv_repack_hbm_prefetch_"
                f"Tensor0_idx{edge['consumer_idx']}_tile{future_tile}"
                f"_loader_core{loader_core_id}_fanout"
            )
            dataops.append(
                _make_kv_repack_broadcast_dataop(
                    fanout_dataop_name,
                    loader_edge,
                    include_broadcast_metadata=False,
                    consumer_core_ids=fanout_core_ids,
                    dataop_core_ids=fanout_core_ids,
                    stcdp_use_unicast=prefetch_fanout_use_unicast,
                    stcdp_use_lxsfp_lx_transfers=(
                        prefetch_fanout_use_lxsfp_lx_transfers
                    ),
                )
            )
            opfuncs.append("STCDPOpLx")
    elif prefetch_source_fanout:
        for piece_idx, source_piece in enumerate(edge["source_pieces"]):
            hbm_dataop_core_ids = sorted(_piece_lx_mem_ids(source_piece))
            if not hbm_dataop_core_ids:
                return None
            hbm_dataop_name = (
                f"{len(dataops)}_STCDPOpHBM_kv_repack_hbm_prefetch_"
                f"Tensor0_idx{edge['consumer_idx']}_tile{future_tile}"
                f"_source_piece{piece_idx}_load"
            )
            dataops.append(
                _make_kv_repack_hbm_roundtrip_dataop(
                    hbm_dataop_name,
                    edge,
                    source_piece=source_piece,
                    phase="load",
                    dataop_core_ids=hbm_dataop_core_ids,
                    lx_start_by_core=_piece_lx_start_by_core(source_piece),
                    emit_lx_roundtrip=prefetch_lx_roundtrip,
                    corelet_id=prefetch_corelet_id,
                )
            )
            opfuncs.append("STCDPOpHBM")
            source_fanout_load_core_ids.append(hbm_dataop_core_ids)
        source_fanout_load_count = len(dataops)
        barrier_dataop_name = (
            f"{len(dataops)}_nop_kv_repack_hbm_prefetch_"
            f"Tensor0_idx{edge['consumer_idx']}_tile{future_tile}"
            "_source_fanout_barrier"
        )
        dataops.append(
            _make_nop_dataop(
                barrier_dataop_name,
                list(range(edge["consumer_num_cores"])),
            )
        )
        opfuncs.append("nop")
        fanout_core_ids = restricted_fanout_core_ids or sorted(
            set(range(edge["consumer_num_cores"])).union(hoist["source_core_ids"])
        )
        fanout_dataop_name = (
            f"{len(dataops)}_STCDPOpLx_kv_repack_hbm_prefetch_"
            f"Tensor0_idx{edge['consumer_idx']}_tile{future_tile}"
            "_source_fanout"
        )
        dataops.append(
            _make_kv_repack_broadcast_dataop(
                fanout_dataop_name,
                edge,
                include_broadcast_metadata=False,
                consumer_core_ids=fanout_core_ids,
                dataop_core_ids=fanout_core_ids,
                stcdp_use_unicast=prefetch_fanout_use_unicast,
                stcdp_use_lxsfp_lx_transfers=(
                    prefetch_fanout_use_lxsfp_lx_transfers
                ),
            )
        )
        opfuncs.append("STCDPOpLx")
    else:
        prefetch_pieces = _kv_repack_consumer_hbm_prefetch_pieces(edge)
        for piece_idx, source_piece in enumerate(prefetch_pieces):
            piece_lx_start = next(
                (
                    int(placement["startAddr"][0])
                    for placement in source_piece.get("PlacementInfo", [])
                    if placement.get("type") == "lx"
                ),
                edge["consumer_lx_base"],
            )
            piece_lx_start_by_core = {
                core_id: piece_lx_start for core_id in direct_load_cores
            }
            hbm_dataop_name = (
                f"{len(dataops)}_STCDPOpHBM_kv_repack_hbm_prefetch_"
                f"Tensor0_idx{edge['consumer_idx']}_tile{future_tile}"
                f"_piece{piece_idx}_load"
            )
            dataops.append(
                _make_kv_repack_hbm_roundtrip_dataop(
                    hbm_dataop_name,
                    edge,
                    source_piece=source_piece,
                    phase="load",
                    dataop_core_ids=direct_load_cores,
                    lx_start_by_core=piece_lx_start_by_core,
                    hbm_piece_addr=True,
                    prefer_source_core_hbm_addr=False,
                    emit_lx_roundtrip=prefetch_lx_roundtrip,
                    corelet_id=prefetch_corelet_id,
                )
            )
            opfuncs.append("STCDPOpHBM")

    if fanout_copyback_enabled and (prefetch_source_fanout or prefetch_loader_fanout):
        copyback_pieces = _kv_repack_consumer_hbm_prefetch_pieces(edge)
        if not copyback_pieces:
            return None
        copyback_lx_base = (
            loader_lx_base
            if prefetch_loader_fanout and prefetch_loader_copyback_without_fanout
            else edge["consumer_lx_base"]
        )
        copyback_name_suffix = (
            f"_loader_copyback_core{fanout_copyback_core}"
            if prefetch_loader_fanout and prefetch_loader_copyback_without_fanout
            else f"_fanout_copyback_core{fanout_copyback_core}"
        )
        copyback_dataop_name = (
            f"{len(dataops)}_STCDPOpHBM_kv_repack_hbm_prefetch_"
            f"Tensor0_idx{edge['consumer_idx']}_tile{future_tile}"
            f"{copyback_name_suffix}"
        )
        dataops.append(
            _make_kv_repack_hbm_roundtrip_dataop(
                copyback_dataop_name,
                edge,
                source_piece=copyback_pieces[0],
                phase="store",
                dataop_core_ids=[fanout_copyback_core],
                lx_start_by_core={
                    fanout_copyback_core: copyback_lx_base
                },
                hbm_piece_addr=True,
                prefer_source_core_hbm_addr=False,
                corelet_id=prefetch_corelet_id,
            )
        )
        opfuncs.append("STCDPOpHBM")
        fanout_copyback_dataop_idx = len(dataops) - 1
        fanout_copyback_core_ids = [fanout_copyback_core]

    current_artifact = None
    if not serial_prefetch:
        current_core_ids = list(range(edge["consumer_num_cores"]))
        current_prefill_dataops = []
        current_prefill_opfuncs = []
        for item in current_prefill_inputs:
            dataop_name = (
                f"{len(current_prefill_dataops)}_STCDPOpHBM_"
                f"kv_repack_hbm_prefetch_current_Tensor0_idx{item['lds_idx']}"
                f"_tile{current_tile}_load"
            )
            current_prefill_dataops.append(
                _make_hbm_direct_fill_dataop(
                    dataop_name,
                    layout=item["layout"],
                    stick_dim=item["stick_dim"],
                    iter_sizes=item["iter_sizes"],
                    hbm_start_by_core=item["hbm_start_by_core"],
                    lx_base=item["lx_base"],
                    core_ids=current_core_ids,
                )
            )
            current_prefill_opfuncs.append("STCDPOpHBM")
        use_native_load_prologue = not current_prefill_dataops
        native_load_prologue_dataops = []
        native_load_prologue_opfuncs = []
        if use_native_load_prologue:
            native_load_prologue_dataops.append(
                _make_nop_dataop(
                    (
                        "0_nop_kv_repack_hbm_prefetch_"
                        f"dldsc_native_load_prologue_tile{current_tile}"
                    ),
                    current_core_ids,
                )
            )
            native_load_prologue_opfuncs.append("nop")
        current_compute_dsc = copy.deepcopy(_body(current)["dscs_"][0])
        current_compute_root = {current_name: {"dscs_": [current_compute_dsc]}}
        for item in current_prefill_inputs:
            apply_lx_flip(
                current_compute_root,
                LxFlip(
                    item["lds_idx"],
                    item["lx_base"],
                    "kv-repack-hbm-prefetch-current-input",
                ),
            )
            current_compute_dl = next(iter(current_compute_dsc.values()))
            current_prefill_lds = _lds_by_idx(
                current_compute_dl,
                item["lds_idx"],
            )
            if current_prefill_lds is not None:
                current_prefill_lds["isFirstUse_"] = 0
        current_dataops = [
            *current_prefill_dataops,
            *native_load_prologue_dataops,
            *dataops,
        ]
        current_opfuncs = [
            *current_prefill_opfuncs,
            *native_load_prologue_opfuncs,
            *opfuncs,
        ]
        current_schedule = (
            _kv_repack_copyback_schedule_for_dataops(
                edge["consumer_num_cores"],
                [current_core_ids for _dataop in current_dataops],
                include_compute=True,
            )
            if serialize_current_prefetch
            else _kv_repack_hbm_prefetch_loader_copyback_tail_schedule(
                edge["consumer_num_cores"],
                len(current_prefill_dataops) + len(native_load_prologue_dataops),
                source_fanout_load_core_ids,
                (
                    len(current_prefill_dataops)
                    + len(native_load_prologue_dataops)
                    + source_fanout_load_count
                ),
                (
                    len(current_prefill_dataops)
                    + len(native_load_prologue_dataops)
                    + fanout_copyback_dataop_idx
                ),
                fanout_copyback_core_ids,
            )
            if prefetch_loader_copyback_without_fanout
            and prefetch_loader_fanout
            and fanout_copyback_dataop_idx is not None
            and tail_current_prefetch
            else _kv_repack_hbm_prefetch_loader_copyback_schedule(
                edge["consumer_num_cores"],
                len(current_prefill_dataops) + len(native_load_prologue_dataops),
                source_fanout_load_core_ids,
                (
                    len(current_prefill_dataops)
                    + len(native_load_prologue_dataops)
                    + source_fanout_load_count
                ),
                (
                    len(current_prefill_dataops)
                    + len(native_load_prologue_dataops)
                    + fanout_copyback_dataop_idx
                ),
                fanout_copyback_core_ids,
                overlap_after_sync=overlap_after_sync,
                serialize_load_core_compute=(
                    serialize_loader_core_prefetch and prefetch_loader_fanout
                ),
            )
            if prefetch_loader_copyback_without_fanout
            and prefetch_loader_fanout
            and fanout_copyback_dataop_idx is not None
            else _kv_repack_hbm_prefetch_source_fanout_tail_schedule(
                edge["consumer_num_cores"],
                len(current_prefill_dataops) + len(native_load_prologue_dataops),
                source_fanout_load_core_ids,
                (
                    len(current_prefill_dataops)
                    + len(native_load_prologue_dataops)
                    + source_fanout_load_count
                ),
                (
                    len(current_prefill_dataops)
                    + len(native_load_prologue_dataops)
                    + source_fanout_load_count
                    + 1
                ),
                fanout_core_ids=fanout_core_ids,
                post_fanout_dataop_idx=(
                    None
                    if fanout_copyback_dataop_idx is None
                    else len(current_prefill_dataops)
                    + len(native_load_prologue_dataops)
                    + fanout_copyback_dataop_idx
                ),
                post_fanout_core_ids=fanout_copyback_core_ids,
            )
            if (prefetch_source_fanout or prefetch_loader_fanout)
            and tail_current_prefetch
            else _kv_repack_hbm_prefetch_source_fanout_schedule(
                edge["consumer_num_cores"],
                len(current_prefill_dataops) + len(native_load_prologue_dataops),
                source_fanout_load_core_ids,
                (
                    len(current_prefill_dataops)
                    + len(native_load_prologue_dataops)
                    + source_fanout_load_count
                ),
                (
                    len(current_prefill_dataops)
                    + len(native_load_prologue_dataops)
                    + source_fanout_load_count
                    + 1
                ),
                fanout_core_ids=fanout_core_ids,
                post_fanout_dataop_idx=(
                    None
                    if fanout_copyback_dataop_idx is None
                    else len(current_prefill_dataops)
                    + len(native_load_prologue_dataops)
                    + fanout_copyback_dataop_idx
                ),
                post_fanout_core_ids=fanout_copyback_core_ids,
                overlap_after_sync=overlap_after_sync,
                serialize_load_core_compute=(
                    serialize_loader_core_prefetch and prefetch_loader_fanout
                ),
            )
            if prefetch_source_fanout or prefetch_loader_fanout
            else _kv_repack_hbm_prefetch_tail_current_schedule(
                edge["consumer_num_cores"],
                len(current_prefill_dataops) + len(native_load_prologue_dataops),
                len(dataops),
            )
            if tail_current_prefetch
            else _kv_repack_hbm_prefetch_native_load_overlap_schedule(
                edge["consumer_num_cores"],
                len(current_dataops),
                overlap_after_sync=overlap_after_sync,
            )
            if use_native_load_prologue
            else _kv_repack_hbm_prefetch_prefill_overlap_schedule(
                edge["consumer_num_cores"],
                len(current_prefill_dataops),
                len(dataops),
            )
        )
        current_artifact = build_flash_attention_pipeline_mixed_sdsc(
            current_sidecar,
            current_dataops,
            current_opfuncs,
            current_schedule,
            [current_compute_dsc],
            edge["consumer_num_cores"],
        )
        current_root = current_artifact[current_sidecar]
        current_body = _body(current)
        for key in (
            "sdscFoldProps_",
            "sdscFolds_",
            "coreFoldProp_",
            "coreletFoldProp_",
            "coreIdToDsc_",
            "numWkSlicesPerDim_",
            "coreIdToWkSlice_",
        ):
            if key in current_body:
                current_root[key] = copy.deepcopy(current_body[key])
        current_root["flashAttentionPipeline_"].update(
            {
                "source": "generated-flash-prefill-kv-hbm-prefetch-current",
                "kv_repack_hbm_prefetch_hoist_role": "current_prefetch",
                "kv_repack_hbm_prefetch_hoist_experimental": True,
                "kv_repack_hbm_prefetch_hoist_runtime_safe": False,
                "kv_repack_hbm_prefetch_hoist_runtime_forced": True,
                "kv_repack_hbm_prefetch_hoist_serial_prefetch": False,
                "kv_repack_hbm_prefetch_hoist_redundant_future_prefetch": (
                    redundant_future_prefetch
                ),
                "kv_repack_hbm_prefetch_hoist_serialize_current_prefetch": (
                    serialize_current_prefetch
                ),
                "kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch": (
                    serialize_loader_core_prefetch
                ),
                "kv_repack_hbm_prefetch_hoist_overlap_after_sync": (
                    overlap_after_sync
                ),
                "kv_repack_hbm_prefetch_hoist_tail_current_prefetch": (
                    tail_current_prefetch
                ),
                "kv_repack_hbm_prefetch_hoist_prefetch_source_fanout": (
                    prefetch_source_fanout
                ),
                "kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout": (
                    prefetch_loader_fanout
                ),
                "kv_repack_hbm_prefetch_hoist_prefetch_loader_core_id": (
                    loader_core_id if prefetch_loader_fanout else -1
                ),
                "kv_repack_hbm_prefetch_hoist_prefetch_loader_lx_base": (
                    loader_lx_base if prefetch_loader_fanout else -1
                ),
                "kv_repack_hbm_prefetch_hoist_prefetch_loader_lx_base_request": (
                    prefetch_loader_lx_base
                ),
                "kv_repack_hbm_prefetch_hoist_prefetch_fanout_use_unicast": (
                    prefetch_fanout_use_unicast
                ),
                "kv_repack_hbm_prefetch_hoist_prefetch_fanout_use_lxsfp_lx_transfers": (
                    prefetch_fanout_use_lxsfp_lx_transfers
                ),
                "kv_repack_hbm_prefetch_hoist_prefetch_fanout_copyback_core": (
                    fanout_copyback_core if fanout_copyback_enabled else -2
                ),
                "kv_repack_hbm_prefetch_hoist_prefetch_fanout_restrict_to_copyback_core": (
                    prefetch_fanout_restrict_to_copyback_core
                ),
                "kv_repack_hbm_prefetch_hoist_prefetch_loader_copyback_without_fanout": (
                    prefetch_loader_copyback_without_fanout
                ),
                "kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout_full_tile_pieces": (
                    prefetch_loader_fanout_full_tile_pieces
                ),
                "kv_repack_hbm_prefetch_hoist_prefetch_lx_roundtrip": (
                    prefetch_lx_roundtrip
                ),
                "kv_repack_hbm_prefetch_hoist_prefetch_corelet_id": (
                    prefetch_corelet_id
                ),
                "kv_repack_hbm_prefetch_hoist_current_tile": current_tile,
                "kv_repack_hbm_prefetch_hoist_future_tile": future_tile,
                "kv_repack_hbm_prefetch_hoist_future_input_idx": hoist[
                    "future_input_idx"
                ],
                "kv_repack_hbm_prefetch_hoist_future_predecessor_sdsc": (
                    future_prod_name
                ),
                "kv_repack_hbm_prefetch_hoist_future_consumer_sdsc": (
                    future_cons_name
                ),
                "kv_repack_additional_consumers": edge.get(
                    "additional_consumers", []
                ),
                "kv_repack_hbm_prefetch_hoist_future_consumer_sidecar": (
                    future_cons_sidecar
                ),
                "kv_repack_hbm_prefetch_hoist_source_hbm_addr": (
                    edge["shared_hbm_addr"]
                ),
                "kv_repack_hbm_prefetch_hoist_consumer_lx_base": (
                    edge["consumer_lx_base"]
                ),
                "kv_repack_hbm_prefetch_hoist_original_consumer_lx_base": (
                    original_consumer_lx_base
                ),
                "kv_repack_hbm_prefetch_hoist_dataop_count": len(dataops),
                "kv_repack_hbm_prefetch_hoist_native_load_prologue": (
                    use_native_load_prologue
                ),
                "kv_repack_hbm_prefetch_hoist_prefilled_current_inputs": [
                    {
                        "lds_idx": item["lds_idx"],
                        "lx_base": item["lx_base"],
                    }
                    for item in current_prefill_inputs
                ],
                "replaces_sdsc": current_name,
                "tile_index": current_tile,
                "requested_tile_index": tile_index,
                "compute_tile_count": 1,
                "overlap_candidate": True,
            }
        )

    future_compute_dsc = copy.deepcopy(_body(edge["consumer"])["dscs_"][0])
    future_compute_root = {future_cons_name: {"dscs_": [future_compute_dsc]}}
    if not fanout_copyback_enabled:
        apply_lx_flip(
            future_compute_root,
            LxFlip(
                edge["consumer_idx"],
                edge["consumer_lx_base"],
                "kv-repack-hbm-prefetch-future-consumer-input",
            ),
        )
    future_compute_dl = next(iter(future_compute_dsc.values()))
    future_prefetch_lds = _lds_by_idx(future_compute_dl, edge["consumer_idx"])
    if (
        future_prefetch_lds is not None
        and not serial_prefetch
        and not redundant_future_prefetch
        and external_future_prefetch
    ):
        future_prefetch_lds["isExternal_"] = 1
        future_prefetch_lds["isFirstUse_"] = 0

    if serial_prefetch or redundant_future_prefetch:
        if redundant_future_prefetch and not serial_prefetch:
            future_dataops = []
            for dataop in dataops:
                dataop_name, dataop_body = next(iter(dataop.items()))
                future_dataops.append(
                    {
                        f"future_redundant_{dataop_name}": copy.deepcopy(
                            dataop_body
                        )
                    }
                )
        else:
            future_dataops = dataops
        future_opfuncs = list(opfuncs)
        future_schedule = _kv_repack_copyback_schedule_for_dataops(
            edge["consumer_num_cores"],
            [direct_load_cores for _dataop in future_dataops],
            include_compute=True,
        )
    else:
        future_dataops = []
        future_opfuncs = []
        future_schedule = {
            str(core_id): [[-1, 0, 0, 0]]
            for core_id in range(edge["consumer_num_cores"])
        }
    future_consumer_artifact = build_flash_attention_pipeline_mixed_sdsc(
        future_cons_sidecar,
        future_dataops,
        future_opfuncs,
        future_schedule,
        [future_compute_dsc],
        edge["consumer_num_cores"],
    )
    future_root = future_consumer_artifact[future_cons_sidecar]
    future_cons_body = _body(edge["consumer"])
    for key in (
        "sdscFoldProps_",
        "sdscFolds_",
        "coreFoldProp_",
        "coreletFoldProp_",
        "coreIdToDsc_",
        "numWkSlicesPerDim_",
        "coreIdToWkSlice_",
    ):
        if key in future_cons_body:
            future_root[key] = copy.deepcopy(future_cons_body[key])
    future_root["flashAttentionPipeline_"].update(
        {
            **_kv_repack_broadcast_meta(
                edge,
                source="generated-flash-prefill-kv-hbm-prefetch-future-consumer",
                tile_index=future_tile,
                requested_tile_index=tile_index,
                input_idx=hoist["future_input_idx"],
                executable=True,
                runtime_forced=True,
            ),
            "kv_repack_broadcast_role": "future_consumer",
            "kv_repack_hbm_prefetch_hoist_role": "future_consumer",
            "kv_repack_hbm_prefetch_hoist_experimental": True,
            "kv_repack_hbm_prefetch_hoist_runtime_safe": False,
            "kv_repack_hbm_prefetch_hoist_runtime_forced": True,
            "kv_repack_hbm_prefetch_hoist_serial_prefetch": serial_prefetch,
            "kv_repack_hbm_prefetch_hoist_redundant_future_prefetch": (
                redundant_future_prefetch
            ),
            "kv_repack_hbm_prefetch_hoist_serialize_current_prefetch": (
                serialize_current_prefetch
            ),
            "kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch": (
                serialize_loader_core_prefetch
            ),
            "kv_repack_hbm_prefetch_hoist_external_future_prefetch": (
                external_future_prefetch
            ),
            "kv_repack_hbm_prefetch_hoist_overlap_after_sync": (
                overlap_after_sync
            ),
            "kv_repack_hbm_prefetch_hoist_tail_current_prefetch": (
                tail_current_prefetch
            ),
            "kv_repack_hbm_prefetch_hoist_prefetch_source_fanout": (
                prefetch_source_fanout
            ),
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout": (
                prefetch_loader_fanout
            ),
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_core_id": (
                loader_core_id if prefetch_loader_fanout else -1
            ),
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_lx_base": (
                loader_lx_base if prefetch_loader_fanout else -1
            ),
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_lx_base_request": (
                prefetch_loader_lx_base
            ),
            "kv_repack_hbm_prefetch_hoist_prefetch_fanout_use_unicast": (
                prefetch_fanout_use_unicast
            ),
            "kv_repack_hbm_prefetch_hoist_prefetch_fanout_use_lxsfp_lx_transfers": (
                prefetch_fanout_use_lxsfp_lx_transfers
            ),
            "kv_repack_hbm_prefetch_hoist_prefetch_fanout_copyback_core": (
                fanout_copyback_core if fanout_copyback_enabled else -2
            ),
            "kv_repack_hbm_prefetch_hoist_prefetch_fanout_restrict_to_copyback_core": (
                prefetch_fanout_restrict_to_copyback_core
            ),
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_copyback_without_fanout": (
                prefetch_loader_copyback_without_fanout
            ),
            "kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout_full_tile_pieces": (
                prefetch_loader_fanout_full_tile_pieces
            ),
            "kv_repack_hbm_prefetch_hoist_prefetch_lx_roundtrip": (
                prefetch_lx_roundtrip
            ),
            "kv_repack_hbm_prefetch_hoist_prefetch_corelet_id": (
                prefetch_corelet_id
            ),
            "kv_repack_hbm_prefetch_hoist_current_tile": current_tile,
            "kv_repack_hbm_prefetch_hoist_future_tile": future_tile,
            "kv_repack_hbm_prefetch_hoist_future_input_idx": hoist[
                "future_input_idx"
            ],
            "kv_repack_hbm_prefetch_hoist_input_lx_base": edge[
                "consumer_lx_base"
            ],
            "kv_repack_hbm_prefetch_hoist_original_input_lx_base": (
                original_consumer_lx_base
            ),
            "kv_repack_hbm_prefetch_hoist_source_hbm_addr": (
                edge["shared_hbm_addr"]
            ),
            "kv_repack_hbm_prefetch_hoist_dataop_count": (
                len(future_dataops)
                if serial_prefetch or redundant_future_prefetch
                else 0
            ),
            "replaces_sdsc": future_cons_name,
            "compute_tile_count": 1,
        }
    )

    artifacts = [producer_artifact, future_consumer_artifact]
    if current_artifact is not None:
        artifacts.insert(1, current_artifact)
    return {
        "artifacts": artifacts,
        "replacements": (
            {future_cons_name: future_cons_sidecar}
            if serial_prefetch
            else {
                current_name: current_sidecar,
                future_cons_name: future_cons_sidecar,
            }
        ),
        "insertions_before": {current_name: [producer_sidecar]},
        "omissions": {future_prod_name},
        "bundle_attrs": {},
        "pointwise_lx_region0": edge["consumer_lx_base"] + edge["slice_bytes"],
        "rejection_reasons": reasons,
    }


def build_flash_attention_kv_repack_broadcast_copyback_artifacts(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    name_prefix: str = "mixed_flash_kv_repack_broadcast_copyback",
    stcdp_subpiece_reuse: bool = True,
    broadcast_group_size: int = 0,
    self_resident_source: bool = False,
    stcdp_use_unicast: int = -1,
    stcdp_force_mc_mode: int = -1,
    readback_core: int = -1,
    direct_source: bool = False,
    hbm_roundtrip: bool = False,
    hbm_source_fanout: bool = False,
    hbm_direct_load: bool = False,
    hbm_roundtrip_load_only: bool = False,
    hbm_roundtrip_barrier_only: bool = False,
    data_only: bool = False,
    replace_consumer: bool = False,
    compute_only: bool = False,
    exact_clone: bool = False,
    preserve_consumer_name: bool = False,
) -> dict | None:
    """Build a K/V repack copyback probe that restores the original HBM input.

    The normal probe replaces the low-core ReStickify producer with an
    LX-producing sidecar. It then inserts a mixed copyback sidecar before the
    original batchmatmul: STCDPOpLx copies producer LX into consumer LX,
    STCDPOpHBM reads one consumer-core replica back to the original HBM input
    address, and the following original HBM-backed batchmatmul is left in the
    bundle as the value check behind a real SDSC boundary.
    """
    selected_tile, selected_input, edge, reasons = (
        _resolve_flash_attention_kv_repack_broadcast_edge(
            sdscs_json,
            tile_index,
        )
    )
    if edge is None or selected_tile is None or selected_input is None:
        return None
    edge = copy.deepcopy(edge)
    if self_resident_source:
        edge["source_lx_base"] = edge["consumer_lx_base"]
        _move_lx_piece_start_addrs(
            edge["source_pieces"],
            edge["source_lx_base"],
        )
    try:
        selected_readback_core = _kv_repack_copyback_core(edge, readback_core)
    except ValueError:
        return None

    prod_name = next(iter(edge["producer"]))
    cons_name = next(iter(edge["consumer"]))
    pred_sidecar = f"{name_prefix}_{selected_tile}_input{selected_input}_producer"
    copyback_sidecar = f"{name_prefix}_{selected_tile}_input{selected_input}_copyback"
    clone_sidecar = f"{name_prefix}_{selected_tile}_input{selected_input}_exact_clone"
    if preserve_consumer_name:
        copyback_sidecar = cons_name
        clone_sidecar = cons_name

    if exact_clone:
        replacements = {} if preserve_consumer_name else {cons_name: clone_sidecar}
        return {
            "artifacts": [{clone_sidecar: copy.deepcopy(_body(edge["consumer"]))}],
            "replacements": replacements,
            "insertions_before": {},
            "bundle_attrs": {},
            "pointwise_lx_region0": edge["consumer_lx_base"] + edge["slice_bytes"],
            "rejection_reasons": reasons,
        }

    producer_artifact = None
    uses_original_hbm_source = (
        hbm_roundtrip or hbm_source_fanout or hbm_direct_load
    )
    if not uses_original_hbm_source:
        producer_artifact = {pred_sidecar: copy.deepcopy(_body(edge["producer"]))}
        apply_lx_flip(
            producer_artifact,
            LxFlip(
                edge["producer_idx"],
                edge["source_lx_base"],
                "kv-repack-broadcast-copyback-producer-output",
            ),
        )
        producer_artifact[pred_sidecar].setdefault("flashAttentionPipeline_", {}).update(
            {
                **_kv_repack_broadcast_meta(
                    edge,
                    source=(
                        "generated-flash-prefill-kv-repack-broadcast-"
                        "copyback-producer"
                    ),
                    tile_index=selected_tile,
                    requested_tile_index=tile_index,
                    input_idx=selected_input,
                    executable=True,
                    runtime_forced=True,
                    skip_source_core_destinations=self_resident_source,
                ),
                "kv_repack_broadcast_role": "copyback_producer",
                "kv_repack_producer_sidecar": pred_sidecar,
                "kv_repack_copyback_sidecar": copyback_sidecar,
                "kv_repack_copyback_readback_core": selected_readback_core,
                "kv_repack_copyback_hbm_roundtrip": False,
                "kv_repack_self_resident_source": self_resident_source,
                "kv_repack_stcdp_use_unicast": stcdp_use_unicast,
                "kv_repack_stcdp_force_mc_mode": stcdp_force_mc_mode,
                "replaces_sdsc": prod_name,
            }
        )

    core_groups = _kv_repack_consumer_core_groups(
        edge["consumer_num_cores"],
        broadcast_group_size,
    )
    source_core_ids = _kv_repack_source_core_ids(edge)
    dataops = []
    opfuncs = []
    ordered_dataop_core_groups = []
    group_dataop_core_ids = []
    source_pieces_for_hbm = (
        []
        if compute_only or hbm_roundtrip_barrier_only
        else edge["source_pieces"]
    )
    if hbm_source_fanout:
        for piece_idx, source_piece in enumerate(source_pieces_for_hbm):
            hbm_dataop_core_ids = sorted(_piece_lx_mem_ids(source_piece))
            if not hbm_dataop_core_ids:
                return None
            lx_start_by_core = _piece_lx_start_by_core(source_piece)
            hbm_dataop_name = (
                f"{len(dataops)}_STCDPOpHBM_kv_repack_copyback_"
                f"Tensor0_idx{edge['consumer_idx']}_tile{selected_tile}"
                f"_hbm_source_piece{piece_idx}_load"
            )
            ordered_dataop_core_groups.append(hbm_dataop_core_ids)
            opfuncs.append("STCDPOpHBM")
            dataops.append(
                _make_kv_repack_hbm_roundtrip_dataop(
                    hbm_dataop_name,
                    edge,
                    source_piece=source_piece,
                    phase="load",
                    dataop_core_ids=hbm_dataop_core_ids,
                    lx_start_by_core=lx_start_by_core,
                )
            )
    if hbm_direct_load:
        direct_load_cores = list(range(edge["consumer_num_cores"]))
        direct_lx_start_by_core = {
            core_id: edge["consumer_lx_base"] for core_id in direct_load_cores
        }
        for piece_idx, source_piece in enumerate(source_pieces_for_hbm):
            hbm_dataop_name = (
                f"{len(dataops)}_STCDPOpHBM_kv_repack_copyback_"
                f"Tensor0_idx{edge['consumer_idx']}_tile{selected_tile}"
                f"_hbm_direct_piece{piece_idx}_load"
            )
            ordered_dataop_core_groups.append(direct_load_cores)
            opfuncs.append("STCDPOpHBM")
            dataops.append(
                _make_kv_repack_hbm_roundtrip_dataop(
                    hbm_dataop_name,
                    edge,
                    source_piece=source_piece,
                    phase="load",
                    dataop_core_ids=direct_load_cores,
                    lx_start_by_core=direct_lx_start_by_core,
                )
            )

    if (
        not compute_only
        and not direct_source
        and not hbm_roundtrip
        and not hbm_direct_load
        and not (hbm_source_fanout and hbm_roundtrip_barrier_only)
    ):
        for group_idx, core_ids in enumerate(core_groups):
            group_suffix = "" if len(core_groups) == 1 else f"_group{group_idx}"
            dataop_name = (
                f"{len(dataops)}_STCDPOpLx_kv_repack_copyback_"
                f"Tensor0_idx{edge['consumer_idx']}_tile{selected_tile}"
                f"{group_suffix}"
            )
            dataop_core_ids = sorted(set(core_ids).union(source_core_ids))
            group_dataop_core_ids.append(dataop_core_ids)
            ordered_dataop_core_groups.append(dataop_core_ids)
            opfuncs.append("STCDPOpLx")
            dataops.append(
                _make_kv_repack_broadcast_dataop(
                    dataop_name,
                    edge,
                    include_broadcast_metadata=False,
                    stcdp_subpiece_reuse=stcdp_subpiece_reuse,
                    consumer_core_ids=core_ids,
                    dataop_core_ids=dataop_core_ids,
                    skip_source_core_destinations=self_resident_source,
                    stcdp_use_unicast=stcdp_use_unicast,
                    stcdp_force_mc_mode=stcdp_force_mc_mode,
                )
            )

    for piece_idx, source_piece in enumerate(source_pieces_for_hbm):
        if hbm_source_fanout:
            hbm_dataop_core_ids = [selected_readback_core]
            lx_start_by_core = None
            hbm_suffix = f"_piece{piece_idx}_core{selected_readback_core}"
            hbm_dataop_name = (
                f"{len(dataops)}_STCDPOpHBM_kv_repack_copyback_"
                f"Tensor0_idx{edge['consumer_idx']}_tile{selected_tile}"
                f"{hbm_suffix}"
            )
            ordered_dataop_core_groups.append(hbm_dataop_core_ids)
            opfuncs.append("STCDPOpHBM")
            dataops.append(
                _make_kv_repack_copyback_hbm_dataop(
                    hbm_dataop_name,
                    edge,
                    source_piece=source_piece,
                    dataop_core_ids=hbm_dataop_core_ids,
                    lx_start_by_core=lx_start_by_core,
                )
            )
            continue
        if hbm_roundtrip:
            hbm_dataop_core_ids = sorted(_piece_lx_mem_ids(source_piece))
            if not hbm_dataop_core_ids:
                return None
            lx_start_by_core = _piece_lx_start_by_core(source_piece)
            phases = ("load",) if hbm_roundtrip_load_only else ("load", "store")
            for phase in phases:
                hbm_dataop_name = (
                    f"{len(dataops)}_STCDPOpHBM_kv_repack_copyback_"
                    f"Tensor0_idx{edge['consumer_idx']}_tile{selected_tile}"
                    f"_roundtrip_source_piece{piece_idx}_{phase}"
                )
                ordered_dataop_core_groups.append(hbm_dataop_core_ids)
                opfuncs.append("STCDPOpHBM")
                dataops.append(
                    _make_kv_repack_hbm_roundtrip_dataop(
                        hbm_dataop_name,
                        edge,
                        source_piece=source_piece,
                        phase=phase,
                        dataop_core_ids=hbm_dataop_core_ids,
                        lx_start_by_core=lx_start_by_core,
                    )
                )
            continue
        if direct_source:
            hbm_dataop_core_ids = sorted(_piece_lx_mem_ids(source_piece))
            lx_start_by_core = _piece_lx_start_by_core(source_piece)
            hbm_suffix = f"_source_piece{piece_idx}"
        else:
            hbm_dataop_core_ids = [selected_readback_core]
            lx_start_by_core = None
            hbm_suffix = f"_piece{piece_idx}_core{selected_readback_core}"
        if not hbm_dataop_core_ids:
            return None
        hbm_dataop_name = (
            f"{len(dataops)}_STCDPOpHBM_kv_repack_copyback_"
            f"Tensor0_idx{edge['consumer_idx']}_tile{selected_tile}"
            f"{hbm_suffix}"
        )
        ordered_dataop_core_groups.append(hbm_dataop_core_ids)
        opfuncs.append("STCDPOpHBM")
        dataops.append(
            _make_kv_repack_copyback_hbm_dataop(
                hbm_dataop_name,
                edge,
                source_piece=source_piece,
                dataop_core_ids=hbm_dataop_core_ids,
                lx_start_by_core=lx_start_by_core,
            )
        )
    barrier_dataop_core_ids = None
    if not compute_only:
        barrier_dataop_core_ids = list(range(edge["consumer_num_cores"]))
        barrier_dataop_name = (
            f"{len(dataops)}_nop_kv_repack_copyback_barrier_"
            f"Tensor0_idx{edge['consumer_idx']}_tile{selected_tile}"
        )
        dataops.append(_make_nop_dataop(barrier_dataop_name, barrier_dataop_core_ids))
        opfuncs.append("nop")

    cons_body = _body(edge["consumer"])
    schedule = _kv_repack_copyback_schedule_for_dataops(
        edge["consumer_num_cores"],
        ordered_dataop_core_groups,
        barrier_dataop_core_ids=barrier_dataop_core_ids,
        include_compute=not data_only,
    )
    if data_only:
        copyback_artifact = _build_flash_attention_dataop_only_sdsc(
            copyback_sidecar,
            dataops,
            opfuncs,
            schedule,
            edge["consumer_num_cores"],
        )
    else:
        compute_dsc = copy.deepcopy(cons_body["dscs_"][0])
        copyback_artifact = build_flash_attention_pipeline_mixed_sdsc(
            copyback_sidecar,
            dataops,
            opfuncs,
            schedule,
            [compute_dsc],
            edge["consumer_num_cores"],
        )
    copyback_root = copyback_artifact[copyback_sidecar]
    copyback_root["flashAttentionPipeline_"].update(
        {
            **_kv_repack_broadcast_meta(
                edge,
                source="generated-flash-prefill-kv-repack-broadcast-copyback",
                tile_index=selected_tile,
                requested_tile_index=tile_index,
                input_idx=selected_input,
                executable=True,
                runtime_forced=True,
                skip_source_core_destinations=self_resident_source,
            ),
            "kv_repack_broadcast_role": "copyback",
            "kv_repack_producer_sidecar": pred_sidecar,
            "kv_repack_copyback_sidecar": copyback_sidecar,
            "kv_repack_copyback_readback_core": selected_readback_core,
            "kv_repack_copyback_readback_op": "STCDPOpHBM",
            "kv_repack_copyback_original_consumer": cons_name,
            "kv_repack_copyback_replaces_consumer": replace_consumer,
            "kv_repack_copyback_inserts_before_consumer": not replace_consumer,
            "kv_repack_stcdp_subpiece_reuse": stcdp_subpiece_reuse,
            "kv_repack_broadcast_group_size": broadcast_group_size,
            "kv_repack_broadcast_group_count": len(group_dataop_core_ids),
            "kv_repack_self_resident_source": self_resident_source,
            "kv_repack_copyback_direct_source": direct_source,
            "kv_repack_copyback_hbm_roundtrip": hbm_roundtrip,
            "kv_repack_copyback_hbm_source_fanout": hbm_source_fanout,
            "kv_repack_copyback_hbm_direct_load": hbm_direct_load,
            "kv_repack_copyback_hbm_roundtrip_load_only": hbm_roundtrip_load_only,
            "kv_repack_copyback_hbm_roundtrip_barrier_only": (
                hbm_roundtrip_barrier_only
            ),
            "kv_repack_copyback_data_only": data_only,
            "kv_repack_copyback_compute_only": compute_only,
            "kv_repack_copyback_preserve_consumer_name": preserve_consumer_name,
            "kv_repack_stcdp_use_unicast": stcdp_use_unicast,
            "kv_repack_stcdp_force_mc_mode": stcdp_force_mc_mode,
            "replaces_sdsc": cons_name,
            "compute_tile_count": 0 if data_only else 1,
        }
    )
    if not data_only:
        for key in (
            "sdscFoldProps_",
            "sdscFolds_",
            "coreFoldProp_",
            "coreletFoldProp_",
            "coreIdToDsc_",
            "numWkSlicesPerDim_",
            "coreIdToWkSlice_",
        ):
            if key in cons_body:
                copyback_root[key] = copy.deepcopy(cons_body[key])

    artifacts = [copyback_artifact] if producer_artifact is None else [
        producer_artifact,
        copyback_artifact,
    ]
    replacements = {} if producer_artifact is None else {prod_name: pred_sidecar}
    insertions_before = {}
    if preserve_consumer_name:
        pass
    elif replace_consumer:
        replacements[cons_name] = copyback_sidecar
    else:
        insertions_before = {cons_name: [copyback_sidecar]}

    return {
        "artifacts": artifacts,
        "replacements": replacements,
        "insertions_before": insertions_before,
        "bundle_attrs": {},
        "pointwise_lx_region0": edge["consumer_lx_base"] + edge["slice_bytes"],
        "rejection_reasons": reasons,
    }


def flash_attention_kv_repack_broadcast_copyback_rejection_reasons(
    sdscs_json: list[dict],
    tile_index: int,
) -> list[str]:
    """Explain why an executable K/V copyback probe failed closed."""
    return flash_attention_kv_repack_broadcast_pair_rejection_reasons(
        sdscs_json,
        tile_index,
    )


def _flash_attention_layout_xform_lookahead_edges(
    sdscs_json: list[dict],
    current_tile: int,
) -> tuple[dict | None, list[str]]:
    current_edge, current_reasons = _flash_attention_layout_xform_pair_edge(
        sdscs_json,
        current_tile,
    )
    if current_edge is None:
        return None, [f"current:{reason}" for reason in current_reasons]

    current_consumer_index = current_edge["consumer_index"]
    current_num_cores = int(_body(current_edge["consumer"]).get("numCoresUsed_", 0))
    reasons: list[str] = []
    for future_tile in range(current_tile + 1, _flash_value_flow_tile_count(sdscs_json)):
        future_edge, future_reasons = _flash_attention_layout_xform_pair_edge(
            sdscs_json,
            future_tile,
        )
        if future_edge is None:
            reasons.extend(
                f"future_tile{future_tile}:{reason}"
                for reason in future_reasons
            )
            continue
        future_producer_index = future_edge["producer_index"]
        future_consumer_index = future_edge["consumer_index"]
        if future_consumer_index <= current_consumer_index:
            reasons.append(
                f"future_tile{future_tile}:consumer_not_after_current:"
                f"future={future_consumer_index}:current={current_consumer_index}"
            )
            continue
        if future_producer_index >= current_consumer_index:
            reasons.append(
                f"future_tile{future_tile}:producer_not_ready:"
                f"producer={future_producer_index}:current={current_consumer_index}"
            )
            continue
        future_num_cores = int(_body(future_edge["consumer"]).get("numCoresUsed_", 0))
        if future_num_cores != current_num_cores:
            reasons.append(
                f"future_tile{future_tile}:num_cores_mismatch:"
                f"future={future_num_cores}:current={current_num_cores}"
            )
            continue
        participant_names = [
            next(iter(current_edge["producer"])),
            next(iter(future_edge["producer"])),
            next(iter(current_edge["consumer"])),
            next(iter(future_edge["consumer"])),
        ]
        if len(set(participant_names)) != len(participant_names):
            reasons.append(
                f"future_tile{future_tile}:duplicate_participants:"
                f"{','.join(participant_names)}"
            )
            continue

        max_slice = max(current_edge["slice_bytes"], future_edge["slice_bytes"])
        try:
            bases = allocate_lx_bases(4, max_slice, region0=CONSUMER_LX_BASE)
        except ValueError:
            reasons.append(f"future_tile{future_tile}:lx_allocation_exceeds_capacity")
            continue

        return (
            {
                "current_tile": current_tile,
                "future_tile": future_tile,
                "current_edge": current_edge,
                "future_edge": future_edge,
                "num_cores": current_num_cores,
                "slice_bytes": max_slice,
                "current_consumer_base": bases[0],
                "current_producer_base": bases[1],
                "future_producer_base": bases[2],
                "future_consumer_base": bases[3],
            },
            [],
        )
    return None, reasons or ["lookahead:no_future_candidate"]


def _resolve_flash_attention_layout_xform_lookahead_edges(
    sdscs_json: list[dict],
    tile_index: int,
) -> tuple[dict | None, list[str]]:
    if tile_index != LAYOUT_XFORM_PAIR_AUTO_TILE:
        return _flash_attention_layout_xform_lookahead_edges(sdscs_json, tile_index)

    reasons: list[str] = []
    for candidate in range(_flash_value_flow_tile_count(sdscs_json)):
        result, candidate_reasons = _flash_attention_layout_xform_lookahead_edges(
            sdscs_json,
            candidate,
        )
        if result is not None:
            return result, []
        reasons.extend(
            f"tile{candidate}:{reason}" for reason in candidate_reasons
        )
    return None, reasons or ["auto:no_candidate_tiles"]


def flash_attention_layout_xform_lookahead_rejection_reasons(
    sdscs_json: list[dict],
    tile_index: int,
) -> list[str]:
    """Explain why a layout-transform lookahead pipeline failed closed."""
    _result, reasons = _resolve_flash_attention_layout_xform_lookahead_edges(
        sdscs_json,
        tile_index,
    )
    return reasons


def _make_layout_xform_dataop(
    name: str,
    edge: dict,
    *,
    producer_base: int,
    consumer_base: int,
    num_cores: int,
) -> dict:
    dataop = make_datadsc(
        name,
        _stcdp_op(),
        edge["dim_pool"],
        src=Endpoint(
            edge["source_layout"],
            edge["stick_dim"],
            edge["split_dim"],
            producer_base,
        ),
        dst=Endpoint(
            edge["consumer_layout"],
            edge["stick_dim"],
            edge["split_dim"],
            consumer_base,
        ),
        iter_sizes=edge["iter_sizes"],
        stick_size=STICK_SIZE,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
    )
    source_pieces = _layout_xform_source_pieces(edge, producer_base)
    if source_pieces is not None:
        next(iter(dataop.values()))["labeledDs_"][0]["PieceInfo"] = source_pieces
    return dataop


def build_flash_attention_layout_xform_lookahead_tile_artifacts(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    name_prefix: str = "mixed_flash_pipeline_tile_layout_xform_lookahead",
) -> dict | None:
    """Build a fail-closed layout-xform copy-current/prefetch-future probe."""
    lookahead, reasons = _resolve_flash_attention_layout_xform_lookahead_edges(
        sdscs_json,
        tile_index,
    )
    if lookahead is None:
        return None

    current_edge = lookahead["current_edge"]
    future_edge = lookahead["future_edge"]
    num_cores = lookahead["num_cores"]
    current_tile = lookahead["current_tile"]
    future_tile = lookahead["future_tile"]

    current_prod_name = next(iter(current_edge["producer"]))
    future_prod_name = next(iter(future_edge["producer"]))
    current_cons_name = next(iter(current_edge["consumer"]))
    future_cons_name = next(iter(future_edge["consumer"]))
    current_pred_sidecar = f"{name_prefix}_{current_tile}_current_predecessor"
    future_pred_sidecar = f"{name_prefix}_{current_tile}_future_predecessor"
    current_cons_sidecar = f"{name_prefix}_{current_tile}_current_consumer"
    future_cons_sidecar = f"{name_prefix}_{current_tile}_future_consumer"

    current_producer_artifact = {
        current_pred_sidecar: copy.deepcopy(_body(current_edge["producer"]))
    }
    apply_lx_flip(
        current_producer_artifact,
        LxFlip(
            current_edge["producer_idx"],
            lookahead["current_producer_base"],
            "layout-xform-lookahead-current-producer-output",
        ),
    )
    current_producer_artifact[current_pred_sidecar].setdefault(
        "flashAttentionPipeline_", {}
    ).update(
        {
            "source": "generated-flash-prefill-layout-xform-lookahead-producer",
            "layout_xform_mode": "lookahead_current_then_future_prefetch",
            "layout_xform_lookahead_role": "current_predecessor",
            "layout_xform_experimental": True,
            "layout_xform_runtime_safe": False,
            "layout_xform_current_tile": current_tile,
            "layout_xform_future_tile": future_tile,
            "layout_xform_predecessor_lx_base": lookahead[
                "current_producer_base"
            ],
            "replaces_sdsc": current_prod_name,
            "tile_index": current_tile,
            "requested_tile_index": tile_index,
        }
    )

    future_producer_artifact = {
        future_pred_sidecar: copy.deepcopy(_body(future_edge["producer"]))
    }
    apply_lx_flip(
        future_producer_artifact,
        LxFlip(
            future_edge["producer_idx"],
            lookahead["future_producer_base"],
            "layout-xform-lookahead-future-producer-output",
        ),
    )
    future_producer_artifact[future_pred_sidecar].setdefault(
        "flashAttentionPipeline_", {}
    ).update(
        {
            "source": "generated-flash-prefill-layout-xform-lookahead-producer",
            "layout_xform_mode": "lookahead_current_then_future_prefetch",
            "layout_xform_lookahead_role": "future_predecessor",
            "layout_xform_experimental": True,
            "layout_xform_runtime_safe": False,
            "layout_xform_current_tile": current_tile,
            "layout_xform_future_tile": future_tile,
            "layout_xform_predecessor_lx_base": lookahead[
                "future_producer_base"
            ],
            "replaces_sdsc": future_prod_name,
            "tile_index": future_tile,
            "requested_tile_index": tile_index,
        }
    )

    current_compute_dsc = copy.deepcopy(_body(current_edge["consumer"])["dscs_"][0])
    current_compute_root = {current_cons_name: {"dscs_": [current_compute_dsc]}}
    apply_lx_flip(
        current_compute_root,
        LxFlip(
            current_edge["consumer_idx"],
            lookahead["current_consumer_base"],
            "layout-xform-lookahead-current-consumer-input",
        ),
    )
    current_compute_dl = next(iter(current_compute_dsc.values()))
    _add_input_fetch_neighbor_transfer(
        current_compute_dl,
        current_edge["consumer_idx"],
    )
    _add_input_fetch_neighbor_legacy_dims(
        current_compute_dl,
        _body(current_edge["consumer"]).get("numWkSlicesPerDim_", {}),
    )
    current_copy = _make_layout_xform_dataop(
        (
            "0_STCDPOpLx_layout_xform_current_"
            f"Tensor0_idx{current_edge['consumer_idx']}_tile{current_tile}"
        ),
        current_edge,
        producer_base=lookahead["current_producer_base"],
        consumer_base=lookahead["current_consumer_base"],
        num_cores=num_cores,
    )
    future_prefetch = _make_layout_xform_dataop(
        (
            "1_STCDPOpLx_prefetch_layout_xform_future_"
            f"Tensor0_idx{future_edge['consumer_idx']}_tile{future_tile}"
        ),
        future_edge,
        producer_base=lookahead["future_producer_base"],
        consumer_base=lookahead["future_consumer_base"],
        num_cores=num_cores,
    )
    schedule = {
        str(core_id): [[0, -1, 0, 1], [1, 0, 1, 0]]
        for core_id in range(num_cores)
    }
    current_consumer_artifact = build_flash_attention_pipeline_mixed_sdsc(
        current_cons_sidecar,
        [current_copy, future_prefetch],
        ["STCDPOpLx", "STCDPOpLx"],
        schedule,
        [current_compute_dsc],
        num_cores,
    )
    current_consumer_root = current_consumer_artifact[current_cons_sidecar]
    current_cons_body = _body(current_edge["consumer"])
    for key in (
        "sdscFoldProps_",
        "sdscFolds_",
        "coreFoldProp_",
        "coreletFoldProp_",
        "coreIdToDsc_",
        "numWkSlicesPerDim_",
        "coreIdToWkSlice_",
    ):
        if key in current_cons_body:
            current_consumer_root[key] = copy.deepcopy(current_cons_body[key])
    current_consumer_root["flashAttentionPipeline_"].update(
        {
            "source": (
                "generated-flash-prefill-layout-xform-lookahead-current-consumer"
            ),
            "layout_xform_mode": "lookahead_current_then_future_prefetch",
            "layout_xform_lookahead_role": "current_consumer",
            "layout_xform_experimental": True,
            "layout_xform_runtime_safe": False,
            "layout_xform_runtime_forced": True,
            "layout_xform_current_tile": current_tile,
            "layout_xform_future_tile": future_tile,
            "layout_xform_current_predecessor_sdsc": current_prod_name,
            "layout_xform_future_predecessor_sdsc": future_prod_name,
            "layout_xform_current_consumer_sdsc": current_cons_name,
            "layout_xform_future_consumer_sdsc": future_cons_name,
            "layout_xform_current_predecessor_sidecar": current_pred_sidecar,
            "layout_xform_future_predecessor_sidecar": future_pred_sidecar,
            "layout_xform_future_consumer_sidecar": future_cons_sidecar,
            "layout_xform_attached_input_idx": current_edge["consumer_idx"],
            "layout_xform_prefetch_input_idx": future_edge["consumer_idx"],
            "layout_xform_current_producer_lx_base": lookahead[
                "current_producer_base"
            ],
            "layout_xform_current_input_lx_base": lookahead[
                "current_consumer_base"
            ],
            "layout_xform_future_producer_lx_base": lookahead[
                "future_producer_base"
            ],
            "layout_xform_future_input_lx_base": lookahead[
                "future_consumer_base"
            ],
            "slice_bytes": lookahead["slice_bytes"],
            "replaces_sdsc": current_cons_name,
            "tile_index": current_tile,
            "requested_tile_index": tile_index,
            "compute_tile_count": 1,
        }
    )

    future_consumer_artifact = {
        future_cons_sidecar: copy.deepcopy(_body(future_edge["consumer"]))
    }
    apply_lx_flip(
        future_consumer_artifact,
        LxFlip(
            future_edge["consumer_idx"],
            lookahead["future_consumer_base"],
            "layout-xform-lookahead-future-consumer-input",
        ),
    )
    future_consumer_artifact[future_cons_sidecar].setdefault(
        "flashAttentionPipeline_", {}
    ).update(
        {
            "source": (
                "generated-flash-prefill-layout-xform-lookahead-future-consumer"
            ),
            "layout_xform_mode": "lookahead_current_then_future_prefetch",
            "layout_xform_lookahead_role": "future_consumer",
            "layout_xform_experimental": True,
            "layout_xform_runtime_safe": False,
            "layout_xform_runtime_forced": True,
            "layout_xform_current_tile": current_tile,
            "layout_xform_future_tile": future_tile,
            "layout_xform_prefetch_input_idx": future_edge["consumer_idx"],
            "layout_xform_future_input_lx_base": lookahead[
                "future_consumer_base"
            ],
            "replaces_sdsc": future_cons_name,
            "tile_index": future_tile,
            "requested_tile_index": tile_index,
        }
    )

    return {
        "artifacts": [
            current_producer_artifact,
            future_producer_artifact,
            current_consumer_artifact,
            future_consumer_artifact,
        ],
        "replacements": {
            current_prod_name: current_pred_sidecar,
            future_prod_name: future_pred_sidecar,
            current_cons_name: current_cons_sidecar,
            future_cons_name: future_cons_sidecar,
        },
        "bundle_attrs": {},
        "pointwise_lx_region0": lookahead["future_consumer_base"]
        + lookahead["slice_bytes"],
        "rejection_reasons": reasons,
    }


def _hoistable_before(
    sdscs_json: list[dict],
    producer_index: int,
    before_index: int,
) -> list[str]:
    prod = sdscs_json[producer_index]
    reasons: list[str] = []
    for in_idx in _consumer_input_indices(_dl_op(prod)):
        addr = _hbm_base(_dl_op(prod), in_idx)
        if addr is None:
            reasons.append(f"producer_input{in_idx}:not_hbm_backed")
            continue
        latest = _latest_producer_of_hbm(sdscs_json, producer_index, addr)
        if latest is not None and latest[0] >= before_index:
            reasons.append(
                f"producer_input{in_idx}:dependency_not_ready:"
                f"producer={latest[0]}:before={before_index}"
            )
    return reasons


def _flash_attention_layout_xform_hoist_edges(
    sdscs_json: list[dict],
    current_tile: int,
) -> tuple[dict | None, list[str]]:
    current = _flash_value_flow_tile(sdscs_json, current_tile)
    if current is None:
        return None, ["current:tile_not_found"]
    current_consumer_index, current_consumer = current
    current_num_cores = int(_body(current_consumer).get("numCoresUsed_", 0))
    if current_num_cores <= 0:
        return None, ["current:invalid_num_cores"]

    reasons: list[str] = []
    for future_tile in range(current_tile + 1, _flash_value_flow_tile_count(sdscs_json)):
        future = _flash_value_flow_tile(sdscs_json, future_tile)
        if future is None:
            reasons.append(f"future_tile{future_tile}:tile_not_found")
            continue
        _future_consumer_index, future_consumer = future
        future_reasons: list[str] = []
        for input_idx in _consumer_input_indices(_dl_op(future_consumer)):
            candidate_edge, candidate_reasons = (
                _flash_attention_layout_xform_pair_edge(
                    sdscs_json,
                    future_tile,
                    input_idx=input_idx,
                    allow_nonzero_input=True,
                )
            )
            if candidate_edge is None:
                future_reasons.extend(candidate_reasons)
                continue
            candidate_prod_op = _op_name(candidate_edge["producer"])
            if candidate_prod_op != "ReStickifyOpHBM":
                future_reasons.append(
                    f"input{input_idx}:producer_not_restickify_hbm:"
                    f"{candidate_prod_op}"
                )
                continue

            future_producer_index = candidate_edge["producer_index"]
            future_consumer_index = candidate_edge["consumer_index"]
            if future_producer_index <= current_consumer_index:
                future_reasons.append(
                    f"input{input_idx}:producer_already_ready:"
                    f"producer={future_producer_index}:"
                    f"current={current_consumer_index}"
                )
                continue
            if future_consumer_index <= future_producer_index:
                future_reasons.append(
                    f"input{input_idx}:consumer_not_after_producer:"
                    f"consumer={future_consumer_index}:"
                    f"producer={future_producer_index}"
                )
                continue
            future_num_cores = int(
                _body(candidate_edge["consumer"]).get("numCoresUsed_", 0)
            )
            future_prod_cores = int(
                _body(candidate_edge["producer"]).get("numCoresUsed_", 0)
            )
            if (
                future_num_cores != current_num_cores
                or future_prod_cores != current_num_cores
            ):
                future_reasons.append(
                    f"input{input_idx}:num_cores_mismatch:"
                    f"current={current_num_cores}:producer={future_prod_cores}:"
                    f"future={future_num_cores}"
                )
                continue
            hoist_reasons = _hoistable_before(
                sdscs_json,
                future_producer_index,
                current_consumer_index,
            )
            if hoist_reasons:
                future_reasons.extend(
                    f"input{input_idx}:{reason}" for reason in hoist_reasons
                )
                continue
            participant_names = [
                next(iter(current_consumer)),
                next(iter(candidate_edge["producer"])),
                next(iter(candidate_edge["consumer"])),
            ]
            if len(set(participant_names)) != len(participant_names):
                future_reasons.append(
                    f"input{input_idx}:duplicate_participants:"
                    f"{','.join(participant_names)}"
                )
                continue
            try:
                bases = allocate_lx_bases(
                    2,
                    candidate_edge["slice_bytes"],
                    region0=CONSUMER_LX_BASE,
                )
            except ValueError:
                future_reasons.append(
                    f"input{input_idx}:lx_allocation_exceeds_capacity"
                )
                continue
            return (
                {
                    "current_tile": current_tile,
                    "future_tile": future_tile,
                    "current_consumer_index": current_consumer_index,
                    "current_consumer": current_consumer,
                    "future_edge": candidate_edge,
                    "num_cores": current_num_cores,
                    "future_producer_base": bases[0],
                    "future_consumer_base": bases[1],
                    "slice_bytes": candidate_edge["slice_bytes"],
                },
                [],
            )
        reasons.extend(
            f"future_tile{future_tile}:{reason}" for reason in future_reasons
        )
    return None, reasons or ["hoist:no_future_candidate"]


def _resolve_flash_attention_layout_xform_hoist_edges(
    sdscs_json: list[dict],
    tile_index: int,
) -> tuple[dict | None, list[str]]:
    if tile_index != LAYOUT_XFORM_PAIR_AUTO_TILE:
        return _flash_attention_layout_xform_hoist_edges(sdscs_json, tile_index)

    reasons: list[str] = []
    for candidate in range(_flash_value_flow_tile_count(sdscs_json)):
        result, candidate_reasons = _flash_attention_layout_xform_hoist_edges(
            sdscs_json,
            candidate,
        )
        if result is not None:
            return result, []
        reasons.extend(
            f"tile{candidate}:{reason}" for reason in candidate_reasons
        )
    return None, reasons or ["auto:no_candidate_tiles"]


def flash_attention_layout_xform_hoist_rejection_reasons(
    sdscs_json: list[dict],
    tile_index: int,
) -> list[str]:
    """Explain why a hoisted future layout-transform prefetch failed closed."""
    _result, reasons = _resolve_flash_attention_layout_xform_hoist_edges(
        sdscs_json,
        tile_index,
    )
    return reasons


def build_flash_attention_layout_xform_hoist_tile_artifacts(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    name_prefix: str = "mixed_flash_pipeline_tile_layout_xform_hoist",
) -> dict | None:
    """Hoist an independent future producer and prefetch it during current compute."""
    hoist, reasons = _resolve_flash_attention_layout_xform_hoist_edges(
        sdscs_json,
        tile_index,
    )
    if hoist is None:
        return None

    current = hoist["current_consumer"]
    future_edge = hoist["future_edge"]
    num_cores = hoist["num_cores"]
    current_tile = hoist["current_tile"]
    future_tile = hoist["future_tile"]

    current_name = next(iter(current))
    future_prod_name = next(iter(future_edge["producer"]))
    future_cons_name = next(iter(future_edge["consumer"]))
    current_sidecar = f"{name_prefix}_{current_tile}_current_consumer"
    future_cons_sidecar = f"{name_prefix}_{current_tile}_future_consumer"

    future_compute_dsc = copy.deepcopy(_body(future_edge["producer"])["dscs_"][0])
    future_compute_root = {future_prod_name: {"dscs_": [future_compute_dsc]}}
    apply_lx_flip(
        future_compute_root,
        LxFlip(
            future_edge["producer_idx"],
            hoist["future_producer_base"],
            "layout-xform-hoist-future-producer-output",
        ),
    )

    current_compute_dsc = copy.deepcopy(_body(current)["dscs_"][0])
    future_prefetch = _make_layout_xform_dataop(
        (
            "0_STCDPOpLx_prefetch_layout_xform_hoisted_future_"
            f"Tensor0_idx{future_edge['consumer_idx']}_tile{future_tile}"
        ),
        future_edge,
        producer_base=hoist["future_producer_base"],
        consumer_base=hoist["future_consumer_base"],
        num_cores=num_cores,
    )
    schedule = {
        str(core_id): [[-1, 0, 0, 1], [0, 1, 1, 0]]
        for core_id in range(num_cores)
    }
    current_artifact = build_flash_attention_pipeline_mixed_sdsc(
        current_sidecar,
        [future_prefetch],
        ["STCDPOpLx"],
        schedule,
        [future_compute_dsc, current_compute_dsc],
        num_cores,
    )
    current_root = current_artifact[current_sidecar]
    current_body = _body(current)
    for key in (
        "sdscFoldProps_",
        "sdscFolds_",
        "coreFoldProp_",
        "coreletFoldProp_",
        "coreIdToDsc_",
        "numWkSlicesPerDim_",
        "coreIdToWkSlice_",
    ):
        if key in current_body:
            current_root[key] = copy.deepcopy(current_body[key])
    current_root["flashAttentionPipeline_"].update(
        {
            "source": "generated-flash-prefill-layout-xform-hoisted-future",
            "layout_xform_mode": "hoisted_future_producer_prefetch",
            "layout_xform_hoist_role": "current_consumer",
            "layout_xform_experimental": True,
            "layout_xform_runtime_safe": False,
            "layout_xform_runtime_forced": True,
            "layout_xform_current_tile": current_tile,
            "layout_xform_future_tile": future_tile,
            "layout_xform_future_predecessor_sdsc": future_prod_name,
            "layout_xform_omitted_future_predecessor_sdsc": future_prod_name,
            "layout_xform_future_consumer_sdsc": future_cons_name,
            "layout_xform_future_consumer_sidecar": future_cons_sidecar,
            "layout_xform_prefetch_input_idx": future_edge["consumer_idx"],
            "layout_xform_future_producer_lx_base": hoist[
                "future_producer_base"
            ],
            "layout_xform_future_input_lx_base": hoist[
                "future_consumer_base"
            ],
            "slice_bytes": hoist["slice_bytes"],
            "replaces_sdsc": current_name,
            "tile_index": current_tile,
            "requested_tile_index": tile_index,
            "compute_tile_count": 2,
        }
    )

    future_consumer_artifact = {
        future_cons_sidecar: copy.deepcopy(_body(future_edge["consumer"]))
    }
    apply_lx_flip(
        future_consumer_artifact,
        LxFlip(
            future_edge["consumer_idx"],
            hoist["future_consumer_base"],
            "layout-xform-hoist-future-consumer-input",
        ),
    )
    future_consumer_artifact[future_cons_sidecar].setdefault(
        "flashAttentionPipeline_", {}
    ).update(
        {
            "source": "generated-flash-prefill-layout-xform-hoisted-future-consumer",
            "layout_xform_mode": "hoisted_future_producer_prefetch",
            "layout_xform_hoist_role": "future_consumer",
            "layout_xform_experimental": True,
            "layout_xform_runtime_safe": False,
            "layout_xform_runtime_forced": True,
            "layout_xform_current_tile": current_tile,
            "layout_xform_future_tile": future_tile,
            "layout_xform_prefetch_input_idx": future_edge["consumer_idx"],
            "layout_xform_future_input_lx_base": hoist[
                "future_consumer_base"
            ],
            "replaces_sdsc": future_cons_name,
            "tile_index": future_tile,
            "requested_tile_index": tile_index,
        }
    )

    return {
        "artifacts": [current_artifact, future_consumer_artifact],
        "replacements": {
            current_name: current_sidecar,
            future_cons_name: future_cons_sidecar,
        },
        "omissions": {future_prod_name},
        "bundle_attrs": {},
        "pointwise_lx_region0": hoist["future_consumer_base"]
        + hoist["slice_bytes"],
        "rejection_reasons": reasons,
    }


def build_flash_attention_pipeline_overlap_prefix_tile_artifact(
    tile_sdscs: list[dict],
    tile_index: int,
    *,
    name_prefix: str = "mixed_flash_pipeline_tile",
) -> dict | None:
    """Build a one-compute sidecar that overlaps next-tile prefetch with compute.

    This is the executable prefix of the full overlap schedule.  It keeps the
    current Foundation one-DL-DSC contract by copying only the first compute DSC,
    while staging two K/V prefetch tiles and scheduling the first next-tile
    prefetch row together with compute tile 0.
    """
    if len(tile_sdscs) < 2:
        return None

    first = tile_sdscs[0]
    first_body = _body(first)
    num_cores = int(first_body.get("numCoresUsed_", 0))
    if num_cores <= 0:
        return None
    first_dl = _dl_op(first)
    out_indices = _producer_output_indices(first_dl)
    if len(out_indices) != 1:
        return None
    out_idx = out_indices[0]
    layout = _layout_for_lds(first_dl, out_idx)
    stick_dim = _stick_dim_for_lds(first_dl, out_idx)
    split_dim = _single_split_dim(first_body.get("numWkSlicesPerDim_", {}))
    if layout is None or stick_dim is None or split_dim is None:
        return None
    iter_sizes = _iter_sizes_for_layout(first_dl, layout)
    if iter_sizes is None:
        return None

    second = tile_sdscs[1]
    second_body = _body(second)
    if int(second_body.get("numCoresUsed_", 0)) != num_cores:
        return None
    second_dl = _dl_op(second)
    second_out_indices = _producer_output_indices(second_dl)
    if len(second_out_indices) != 1:
        return None
    second_out_idx = second_out_indices[0]
    if (
        _layout_for_lds(second_dl, second_out_idx) != layout
        or _stick_dim_for_lds(second_dl, second_out_idx) != stick_dim
        or _single_split_dim(second_body.get("numWkSlicesPerDim_", {}))
        != split_dim
        or _iter_sizes_for_layout(second_dl, layout) != iter_sizes
    ):
        return None

    row_dim = _flash_pipeline_row_dim(layout, split_dim, iter_sizes)
    if row_dim is None:
        return None
    slice_bytes = per_core_same_stick_slice_bytes(
        iter_sizes,
        split_dim,
        stick_dim,
        STICK_SIZE,
        num_cores,
    )
    tile_bytes = _exact_tile_bytes_for_tiles(slice_bytes, 2)
    if tile_bytes is None:
        return None
    try:
        bases = allocate_flash_attention_pipeline_bases(
            num_lanes=2,
            tile_bytes=tile_bytes,
            scratch_regions=2,
            region0=PRODUCER_LX_BASE,
        )
    except ValueError:
        return None

    datadscs, opfuncs, _schedule = build_flash_attention_pipeline_bridge(
        dim_pool=layout,
        iter_sizes=iter_sizes,
        stick_size=STICK_SIZE,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
        src_bases=bases["source_bases"],
        dst_lane_bases=bases["lane_bases"],
        layout=layout,
        stick_dim=stick_dim,
        split_dim=split_dim,
        row_dim=row_dim,
        lane_names=["k", "v"],
        tile_bytes=tile_bytes,
        overlap=False,
        stcdp_corelet_id=1,
    )
    if len(datadscs) < 4:
        return None

    name = f"{name_prefix}_{tile_index}"
    compute_dsc = copy.deepcopy(first_body["dscs_"][0])

    artifact = build_flash_attention_pipeline_mixed_sdsc(
        name,
        datadscs[:4],
        opfuncs[:4],
        flash_pipeline_overlap_prefix_schedule(num_lanes=2, num_cores=num_cores),
        [compute_dsc],
        num_cores,
    )
    root = artifact[name]
    for key in (
        "sdscFoldProps_",
        "sdscFolds_",
        "coreFoldProp_",
        "coreletFoldProp_",
        "coreIdToDsc_",
        "numWkSlicesPerDim_",
        "coreIdToWkSlice_",
    ):
        if key in first_body:
            root[key] = copy.deepcopy(first_body[key])
    root["flashAttentionPipeline_"].update(
        {
            "source": "generated-flash-prefill-overlap-prefix-tile",
            "row_dim": row_dim,
            "split_dim": split_dim,
            "stick_dim": stick_dim,
            "layout": layout,
            "iter_sizes": iter_sizes,
            "tile_bytes": tile_bytes,
            "tile_index": tile_index,
            "replaces_sdsc": next(iter(first)),
            "prefetch_tile_count": 2,
            "prefetch_corelet_id": 1,
            "compute_tile_count": 1,
            "overlap_prefix": True,
        }
    )
    return artifact


def _latest_producer_of_hbm(
    sdscs_json: list[dict],
    before_index: int,
    hbm_addr: str,
):
    for p in range(before_index - 1, -1, -1):
        prod = sdscs_json[p]
        prod_dl = _dl_op(prod)
        for out_idx in _producer_output_indices(prod_dl):
            if _hbm_base(prod_dl, out_idx) == hbm_addr:
                return p, prod, out_idx
    return None


def _renumber_datadscs(datadscs: list[dict], start: int) -> list[dict]:
    renamed = []
    for offset, datadsc in enumerate(datadscs):
        old_name, body = next(iter(datadsc.items()))
        suffix = old_name.split("_", 1)[1]
        renamed.append({f"{start + offset}_{suffix}": body})
    return renamed


def _flash_value_flow_tile(sdscs_json: list[dict], tile_index: int):
    batch_seen = -1
    for c, cons in enumerate(sdscs_json):
        if _op_name(cons) != "batchmatmul":
            continue
        batch_seen += 1
        if batch_seen == tile_index:
            return c, cons
    return None


def _flash_value_flow_tile_count(sdscs_json: list[dict]) -> int:
    return sum(1 for sdsc in sdscs_json if _op_name(sdsc) == "batchmatmul")


def flash_attention_value_flow_tile_rejection_reasons(
    sdscs_json: list[dict],
    tile_index: int,
) -> list[str]:
    """Explain why a requested real value-flow flash tile fails closed."""
    tile = _flash_value_flow_tile(sdscs_json, tile_index)
    if tile is None:
        return ["tile_not_found"]

    c, cons = tile
    cons_dl = _dl_op(cons)
    cons_body = _body(cons)
    num_cores = int(cons_body.get("numCoresUsed_", 0))
    if num_cores <= 0:
        return ["invalid_num_cores"]

    reasons: list[str] = []
    for in_idx in _consumer_input_indices(cons_dl):
        prefix = f"input{in_idx}"
        addr = _hbm_base(cons_dl, in_idx)
        if addr is None:
            reasons.append(f"{prefix}:not_hbm_backed")
            continue
        producer = _latest_producer_of_hbm(sdscs_json, c, addr)
        if producer is None:
            reasons.append(f"{prefix}:no_latest_producer")
            continue

        _p, prod, out_idx = producer
        future = _future_consumers(sdscs_json, _p, addr)
        if len(future) != 1 or future[0][0] != c or future[0][2] != in_idx:
            future_names = [
                f"{next(iter(fcons))}:input{fin_idx}"
                for _fc, fcons, fin_idx in future
            ]
            reasons.append(
                f"{prefix}:not_single_consumer:{','.join(future_names)}"
            )
            continue

        prod_dl = _dl_op(prod)
        prod_layout = _layout_for_lds(prod_dl, out_idx)
        prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
        cons_layout = _layout_for_lds(cons_dl, in_idx)
        cons_stick = _stick_dim_for_lds(cons_dl, in_idx)
        split_dim = _single_split_dim(cons_body.get("numWkSlicesPerDim_", {}))
        if (
            prod_layout is None
            or prod_stick is None
            or cons_layout is None
            or cons_stick is None
            or split_dim is None
        ):
            reasons.append(f"{prefix}:missing_layout_stick_or_split")
            continue

        mismatch_reason = _layout_mismatch_reason(
            prefix,
            prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
        )
        if mismatch_reason is not None:
            reasons.append(mismatch_reason)
            continue

        iter_sizes = _iter_sizes_for_layout(cons_dl, cons_layout)
        if iter_sizes is None:
            reasons.append(f"{prefix}:missing_iter_sizes")
            continue
        if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
            reasons.append(f"{prefix}:invalid_split:{split_dim}")
            continue

        slice_bytes = _reserve_bridge_region_bytes(
            per_core_same_stick_slice_bytes(
                iter_sizes,
                split_dim,
                cons_stick,
                STICK_SIZE,
                num_cores,
            )
        )
        try:
            allocate_lx_bases(3, slice_bytes, region0=PRODUCER_LX_BASE)
        except ValueError:
            reasons.append(f"{prefix}:lx_allocation_exceeds_capacity")
            continue

        # At least one input is eligible.  The requested tile can be realized.
        return []

    return reasons or ["no_candidate_inputs"]


def build_flash_attention_value_flow_tile_artifact(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    name_prefix: str = "mixed_flash_value_flow_tile",
) -> tuple[dict, str] | None:
    """Mutate one flash tile to consume real producer LX values via STCDPOpLx."""
    tile = _flash_value_flow_tile(sdscs_json, tile_index)
    if tile is None:
        return None
    c, cons = tile

    cons_dl = _dl_op(cons)
    cons_body = _body(cons)
    num_cores = int(cons_body.get("numCoresUsed_", 0))
    if num_cores <= 0:
        return None
    edges = []
    for in_idx in _consumer_input_indices(cons_dl):
        addr = _hbm_base(cons_dl, in_idx)
        if addr is None:
            continue
        producer = _latest_producer_of_hbm(sdscs_json, c, addr)
        if producer is None:
            continue
        _p, prod, out_idx = producer
        future = _future_consumers(sdscs_json, _p, addr)
        if len(future) != 1 or future[0][0] != c or future[0][2] != in_idx:
            continue
        prod_dl = _dl_op(prod)
        prod_layout = _layout_for_lds(prod_dl, out_idx)
        prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
        cons_layout = _layout_for_lds(cons_dl, in_idx)
        cons_stick = _stick_dim_for_lds(cons_dl, in_idx)
        split_dim = _single_split_dim(cons_body.get("numWkSlicesPerDim_", {}))
        if (
            prod_layout is None
            or prod_stick is None
            or cons_layout is None
            or cons_stick is None
            or split_dim is None
        ):
            continue
        mismatch_reason = _layout_mismatch_reason(
            f"input{in_idx}",
            prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
        )
        if mismatch_reason is not None:
            continue
        iter_sizes = _iter_sizes_for_layout(cons_dl, cons_layout)
        if iter_sizes is None:
            continue
        if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
            continue
        slice_bytes = _reserve_bridge_region_bytes(
            per_core_same_stick_slice_bytes(
                iter_sizes,
                split_dim,
                cons_stick,
                STICK_SIZE,
                num_cores,
            )
        )
        edges.append(
            {
                "producer": prod,
                "producer_idx": out_idx,
                "consumer_idx": in_idx,
                "layout": cons_layout,
                "stick_dim": cons_stick,
                "split_dim": split_dim,
                "iter_sizes": iter_sizes,
                "slice_bytes": slice_bytes,
            }
        )
    if not edges:
        return None

    max_slice = max(edge["slice_bytes"] for edge in edges)
    try:
        bases = allocate_lx_bases(
            len(edges) * 3,
            max_slice,
            region0=PRODUCER_LX_BASE,
        )
    except ValueError:
        return None

    datadscs = []
    opfuncs = []
    edge_meta = []
    for edge_idx, edge in enumerate(edges):
        producer_base, scratch_base, consumer_base = bases[
            edge_idx * 3: edge_idx * 3 + 3
        ]
        apply_lx_flip(
            edge["producer"],
            LxFlip(edge["producer_idx"], producer_base, "producer-output"),
        )
        apply_lx_flip(
            cons,
            LxFlip(edge["consumer_idx"], consumer_base, "consumer-input"),
        )
        bridge_datadscs, bridge_opfuncs, _sched = build_roundtrip_bridge(
            dim_pool=edge["layout"],
            iter_sizes=edge["iter_sizes"],
            stick_size=STICK_SIZE,
            num_cores=num_cores,
            lx_size=edge["slice_bytes"],
            producer_base=producer_base,
            scratch_base=scratch_base,
            consumer_base=consumer_base,
            layout=edge["layout"],
            stick_dim=edge["stick_dim"],
            split_dim=edge["split_dim"],
        )
        datadscs.extend(_renumber_datadscs(bridge_datadscs, len(datadscs)))
        opfuncs.extend(bridge_opfuncs)
        edge_meta.append(
            {
                "producer": next(iter(edge["producer"])),
                "producer_idx": edge["producer_idx"],
                "consumer_idx": edge["consumer_idx"],
                "layout": edge["layout"],
                "stick_dim": edge["stick_dim"],
                "split_dim": edge["split_dim"],
                "slice_bytes": edge["slice_bytes"],
                "producer_base": producer_base,
                "scratch_base": scratch_base,
                "consumer_base": consumer_base,
            }
        )

    name = f"{name_prefix}_{tile_index}"
    artifact = build_flash_attention_pipeline_mixed_sdsc(
        name,
        datadscs,
        opfuncs,
        mixed_schedule(len(datadscs), num_cores),
        [copy.deepcopy(_body(cons)["dscs_"][0])],
        num_cores,
    )
    root = artifact[name]
    for key in (
        "sdscFoldProps_",
        "sdscFolds_",
        "coreFoldProp_",
        "coreletFoldProp_",
        "numWkSlicesPerDim_",
        "coreIdToDsc_",
        "coreIdToWkSlice_",
    ):
        if key in cons_body:
            root[key] = copy.deepcopy(cons_body[key])
    root["flashAttentionPipeline_"].update(
        {
            "source": "generated-flash-prefill-real-value-flow",
            "tile_index": tile_index,
            "replaces_sdsc": next(iter(cons)),
            "edges": edge_meta,
        }
    )
    return artifact, next(iter(cons))


def detect_pointwise_handoff(sdscs_json: list[dict]):
    """Find a fully legal same-layout pointwise LX handoff edge."""
    for p in range(len(sdscs_json)):
        prod = sdscs_json[p]
        if _op_name(prod) not in _POINTWISE_HANDOFF_OPS:
            continue
        prod_dl = _dl_op(prod)
        out_indices = _producer_output_indices(prod_dl)
        if len(out_indices) != 1:
            continue
        out_idx = out_indices[0]
        prod_addr = _hbm_base(prod_dl, out_idx)
        if prod_addr is None:
            continue
        prod_shard = prod[next(iter(prod))].get("numWkSlicesPerDim_")
        consumers = _future_consumers(sdscs_json, p, prod_addr)
        if len(consumers) != 1:
            continue
        _c, cons, in_idx = consumers[0]
        if _op_name(cons) not in _POINTWISE_HANDOFF_OPS:
            continue
        cons_shard = cons[next(iter(cons))].get("numWkSlicesPerDim_")
        if cons_shard != prod_shard:
            continue
        prod_layout = _layout_for_lds(prod_dl, out_idx)
        prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
        cons_dl = _dl_op(cons)
        cons_layout = _layout_for_lds(cons_dl, in_idx)
        cons_stick = _stick_dim_for_lds(cons_dl, in_idx)
        if (
            prod_layout is None
            or prod_stick is None
            or cons_layout is None
            or cons_stick is None
            or not _same_physical_stick_layout(
                prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
            )
        ):
            continue

        split_dim = _single_split_dim(cons_shard)
        if split_dim is None:
            continue
        num_cores = int(_body(cons).get("numCoresUsed_", 0))
        split_factor = _split_factor(cons_shard, split_dim)
        if num_cores <= 0 or split_factor != num_cores:
            continue
        iter_sizes = _iter_sizes_for_layout(cons_dl, cons_layout)
        if iter_sizes is None:
            continue
        if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
            continue
        slice_bytes = _reserve_bridge_region_bytes(
            per_core_same_stick_slice_bytes(
                iter_sizes,
                split_dim,
                cons_stick,
                STICK_SIZE,
                num_cores,
            )
        )
        if slice_bytes > STREAM_THRESHOLD and cons_stick != split_dim:
            continue
        return {
            "producer": prod,
            "consumer": cons,
            "producer_idx": out_idx,
            "consumer_idx": in_idx,
            "iter_sizes": iter_sizes,
            "layout": cons_layout,
            "stick_dim": cons_stick,
            "split_dim": split_dim,
            "num_cores": num_cores,
            "slice_bytes": slice_bytes,
        }
    return None


def detect_flash_score_scale_handoff(sdscs_json: list[dict]):
    """Find a legal flash score ``batchmatmul -> scalar mul`` LX handoff edge."""
    for p in range(len(sdscs_json)):
        prod = sdscs_json[p]
        if _op_name(prod) != "batchmatmul":
            continue
        prod_dl = _dl_op(prod)
        out_indices = _producer_output_indices(prod_dl)
        if len(out_indices) != 1:
            continue
        out_idx = out_indices[0]
        prod_addr = _hbm_base(prod_dl, out_idx)
        if prod_addr is None:
            continue
        consumers = _future_consumers(sdscs_json, p, prod_addr)
        if len(consumers) != 1:
            continue
        _c, cons, in_idx = consumers[0]
        if _op_name(cons) != "mul":
            continue
        prod_layout = _layout_for_lds(prod_dl, out_idx)
        prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
        cons_dl = _dl_op(cons)
        cons_layout = _layout_for_lds(cons_dl, in_idx)
        cons_stick = _stick_dim_for_lds(cons_dl, in_idx)
        if (
            prod_layout is None
            or prod_stick is None
            or cons_layout is None
            or cons_stick is None
            or not _same_physical_stick_layout(
                prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
            )
        ):
            continue

        prod_shard = _body(prod).get("numWkSlicesPerDim_", {})
        cons_shard = _body(cons).get("numWkSlicesPerDim_", {})
        if not _same_shard_on_layout(prod_shard, cons_shard, cons_layout):
            continue
        split_dim = _single_split_dim(cons_shard)
        if split_dim is None:
            continue
        num_cores = int(_body(cons).get("numCoresUsed_", 0))
        if num_cores <= 0:
            continue
        if (
            _split_factor(cons_shard, split_dim) != num_cores
            or _split_factor(prod_shard, split_dim) != num_cores
        ):
            continue
        iter_sizes = _iter_sizes_for_layout(cons_dl, cons_layout)
        if iter_sizes is None:
            continue
        if int(iter_sizes.get(cons_stick, 0)) > FLASH_SCORE_SCALE_MAX_STICK_ELEMS:
            continue
        if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
            continue
        slice_bytes = _reserve_bridge_region_bytes(
            per_core_same_stick_slice_bytes(
                iter_sizes,
                split_dim,
                cons_stick,
                STICK_SIZE,
                num_cores,
            )
        )
        if slice_bytes > STREAM_THRESHOLD and cons_stick != split_dim:
            continue
        return {
            "producer": prod,
            "consumer": cons,
            "producer_idx": out_idx,
            "consumer_idx": in_idx,
            "iter_sizes": iter_sizes,
            "layout": cons_layout,
            "stick_dim": cons_stick,
            "split_dim": split_dim,
            "num_cores": num_cores,
            "slice_bytes": slice_bytes,
        }
    return None


def detect_onchip_edge(sdscs_json: list[dict]):
    """Find an eligible same-stick same-shard producer->consumer edge.

    The original proof matched add->add only.  Keep that production-shaped narrow
    contract, but allow the same pointwise shape for attention's Inductor-level
    online-softmax graph.  We require a single future consumer to avoid fanout
    values that still need the HBM materialization.
    """
    edge = detect_pointwise_handoff(sdscs_json)
    if edge is None:
        return None
    return (
        edge["producer"],
        edge["consumer"],
        edge["producer_idx"],
        edge["consumer_idx"],
    )


def realize_pointwise_handoff(
    sdscs_json: list[dict],
    *,
    region0: int = PRODUCER_LX_BASE,
) -> bool:
    edge = detect_pointwise_handoff(sdscs_json)
    if edge is None:
        return False
    return _realize_handoff_edge(edge, region0=region0)


def _realize_handoff_edge(
    edge: dict,
    *,
    region0: int = PRODUCER_LX_BASE,
    producer_core_state_init: bool = True,
    producer_num_corelets: int | None = None,
) -> bool:
    if edge["slice_bytes"] <= STREAM_THRESHOLD:
        realization = realize_same_layout_handoff(
            iter_sizes=edge["iter_sizes"],
            layout=edge["layout"],
            stick_dim=edge["stick_dim"],
            split_dim=edge["split_dim"],
            stick_size=STICK_SIZE,
            num_cores=edge["num_cores"],
            producer_ldsidx=edge["producer_idx"],
            consumer_ldsidx=edge["consumer_idx"],
            region0=region0,
        )
    else:
        realization = realize_streamed_handoff(
            iter_sizes=edge["iter_sizes"],
            layout=edge["layout"],
            stick_dim=edge["stick_dim"],
            split_dim=edge["split_dim"],
            stick_size=STICK_SIZE,
            num_cores=edge["num_cores"],
            producer_ldsidx=edge["producer_idx"],
            consumer_ldsidx=edge["consumer_idx"],
            region0=region0,
        )
    if realization is None:
        return False
    prod = edge["producer"]
    cons = edge["consumer"]
    apply_lx_flip(
        prod,
        realization.producer_flip,
        core_state_init=producer_core_state_init,
        num_corelets=producer_num_corelets,
    )
    apply_lx_flip(cons, realization.consumer_flip)
    body = _body(cons)
    body["coreIdToDscSchedule"] = realization.schedule
    body["datadscs_"] = realization.datadscs
    body["opFuncsUsed_"] = realization.opfuncs
    _dl_op(cons)["numCoreletsUsed_DSC2_"] = 1
    return True


def realize_flash_score_scale_handoff(sdscs_json: list[dict]) -> bool:
    edge = detect_flash_score_scale_handoff(sdscs_json)
    if edge is None:
        return False
    # PT producers use the same allocator-shaped endpoint contract as the
    # value-correct first-principles PT-LX bridge: base 0, explicit corelet
    # count, and no producer-side coreStateInit_ injection.
    return _realize_handoff_edge(
        edge,
        region0=0,
        producer_core_state_init=False,
        producer_num_corelets=1,
    )


def realize_flash_attention_pointwise_handoffs(
    sdscs_json: list[dict],
    *,
    score_scale_handoff: bool = False,
    pointwise_region0: int = PRODUCER_LX_BASE,
) -> int:
    """Realize every legal same-layout flash handoff in one flash bundle."""
    count = 0
    # One realization mutates the graph by turning an HBM producer output into an
    # LX endpoint, so at most one new edge can disappear per SDSC iteration.
    for _ in range(len(sdscs_json)):
        if score_scale_handoff and realize_flash_score_scale_handoff(sdscs_json):
            count += 1
            continue
        if not realize_pointwise_handoff(sdscs_json, region0=pointwise_region0):
            break
        count += 1
    return count


def realize_onchip_handoff(
    sdscs_json: list[dict],
    *,
    attention_score_handoff: bool = False,
    static_matmul_handoff: bool = False,
    min_handoff_bytes: int = 1 << 20,
) -> bool:
    """Realize the eligible same-core handoff edge in place; fail-closed.

    When requested, stock SDPA score fanout is handled first because its
    producer feeds both max and sub.  Otherwise, or if that fails closed, detect
    the original pointwise edge, build a same-layout bridge with the same
    size-aware LX allocation as the standalone realization helpers, flip
    producer-output + consumer-input to LX, and fold the bridge into the
    consumer (mixed DL+data-op SuperDSC).
    """
    if attention_score_handoff and realize_attention_score_handoff(
        sdscs_json, min_handoff_bytes
    ):
        return True
    if static_matmul_handoff and realize_static_matmul_handoff(
        sdscs_json, min_handoff_bytes
    ):
        return True

    return realize_pointwise_handoff(sdscs_json)


# --- Bundle-level reduction reshard splice (the SwiGLU mul -> down_proj edge) --
# Mirrors realize_onchip_handoff: an in-memory mutation of the SDSC list that
# (a) flips the producer-output and consumer-input edge labeledDs to LX-resident
# at bases A/B, and (b) inserts a STANDALONE pure-data-op STCDPOpLx SDSC between
# them (dxp.cpp:255 dscs_==0 && dataOpdscs_>0 branch), avoiding the mixed-fold
# SdscTree.cpp:152 reject. The standalone SDSC is inserted into sdscs_json as its
# own {name: body} entry, so the generate_bundle write loop serializes it and the
# bundle.mlir step loop emits it in order between producer and consumer.


def _producer_out_extent_dl(dl: dict, out_idx: int) -> int | None:
    """Total ``out`` extent of producer-output ``out_idx`` from its alloc node.

    Multiplies the per-fold ``factor_`` of the ``out`` coordinate on the
    producer-output HBM allocate node to recover the full logical reduction
    extent K (= the mul output cols = the down-proj K). ``None`` if absent.
    """
    for node in dl.get("scheduleTree_", []):
        if node.get("nodeType_") != "allocate" or node.get("ldsIdx_") != out_idx:
            continue
        coord = node.get("coordinates_", {}).get("coordInfo", {}).get("out")
        if not coord:
            return None
        extent = 1
        for fold in coord["folds"]["dim_prop_attr"]:
            extent *= int(fold["factor_"])
        return extent
    return None


def realize_reduction_reshard_bundle(
    sdscs_json: list[dict],
    *,
    m_rows: int,
    expected_k: int,
    m_split: int,
    n_split: int,
    num_cores: int,
    perband: bool = False,
    sdsc_name: str = "1b_STCDP_reshard",
) -> bool:
    """Realize the SwiGLU mul -> down_proj reduction reshard in the SDSC list.

    Walks producer SDSCs; for each producer-output HBM tensor whose logical
    ``out`` extent matches ``expected_k`` (the reduction dim the down-proj
    reduces over), finds the future consumer SDSC reading that same HBM base
    (``_future_consumers``). Builds the 2-D reshard via
    :func:`realize_reduction_reshard`, flips producer-out and consumer-in to LX
    (:func:`apply_lx_flip`), and inserts the standalone pure-data-op STCDP SDSC
    into ``sdscs_json`` immediately before the consumer. Returns True if any edge
    was realized. Fail-closed: a producer/consumer that does not match the pinned
    geometry is left untouched.
    """
    realized = False
    p = 0
    while p < len(sdscs_json):
        prod = sdscs_json[p]
        try:
            prod_dl = _dl_op(prod)
        except (KeyError, IndexError, StopIteration):
            p += 1
            continue
        edge = None
        for out_idx in _producer_output_indices(prod_dl):
            if _producer_out_extent_dl(prod_dl, out_idx) != expected_k:
                continue
            hbm_addr = _hbm_base(prod_dl, out_idx)
            if hbm_addr is None:
                continue
            consumers = _future_consumers(sdscs_json, p, hbm_addr)
            if len(consumers) == 1:
                edge = (out_idx, consumers[0])
                break
        if edge is None:
            p += 1
            continue
        producer_out_idx, (cons_pos, consumer_sdsc, consumer_in_idx) = edge

        realization = realize_reduction_reshard(
            iter_sizes={"mb_": m_rows, "out_": expected_k},
            layout=["mb_", "out_"],
            row_dim="mb_",
            stick_dim="out_",
            producer_splits={"mb_": m_split, "out_": n_split},
            consumer_splits={"mb_": num_cores, "out_": 1},
            stick_size=STICK_SIZE,
            num_cores=num_cores,
            producer_ldsidx=producer_out_idx,
            consumer_ldsidx=consumer_in_idx,
            perband=perband,
        )
        if realization is None:
            p += 1
            continue

        apply_lx_flip(prod, realization.producer_flip)
        apply_lx_flip(consumer_sdsc, realization.consumer_flip)
        standalone = build_standalone_reduction_reshard_sdsc(realization, sdsc_name)
        sdscs_json.insert(cons_pos, standalone)
        realized = True
        # Resume scanning after the inserted SDSC (now at cons_pos).
        p = cons_pos + 1
    return realized


def build_standalone_reduction_reshard_sdsc(
    realization: "ReductionReshardRealization",
    sdsc_name: str = "1b_STCDP_reshard",
) -> dict:
    """Return the standalone pure-data-op SDSC for a realized reduction reshard.

    Re-keys the realization's ``standalone_sdsc`` under ``sdsc_name`` so multiple
    edges in one bundle get distinct step names.
    """
    body = next(iter(realization.standalone_sdsc.values()))
    return {sdsc_name: body}

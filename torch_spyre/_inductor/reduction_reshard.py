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

"""Self-contained reduction-reshard substrate for the SwiGLU mul -> down_proj edge.

Distilled from flash-ws (``onchip_handoff`` + ``onchip_realize``) onto the main
base, dropping the flash-ws SDPA scaffolding. Two halves share the
:mod:`~torch_spyre._inductor.reshard` device-authoring package:

  DETECTION (:func:`plan_reduction_reshard_edges`) -- a pre-scheduling pass that
  walks every in-graph producer -> consumer activation edge and records the
  genuine ``mul -> down_proj`` reduction-input edge (a ``{mb, out}`` co-split
  producer feeding a ``K``-reduction batchmatmul that does NOT split ``K``) as a
  ``(producer_name, consumer_name)`` pair in a module-level store. The
  scheduler's ``can_fuse_vertical`` queries :func:`reduction_reshard_edges` to
  CO-BUNDLE the edge into one device program -- LX does not persist across
  separate device programs, so the intra-bundle ring reshard only applies once
  both ops land in one bundle.

  REALIZE (:func:`realize_reduction_reshard_bundle`) -- an in-memory SDSC-list
  mutator (called from the config-gated bundle hook) that finds the producer-out
  HBM tensor whose ``out`` extent matches the reduction ``K``, gates STRICTLY on
  the down-proj batchmatmul consumer (:func:`_is_reduction_consumer`), and folds
  the 2-D ``STCDPOpLx`` reshard into the consumer SDSC as a MIXED SuperDSC
  (:func:`reshard.splice_reshard` + :func:`reshard.substrate.mixed_schedule`).
  The mixed fold is the dxp-accepted shape; the per-bundle element-wise misfire
  is what the ``_is_reduction_consumer`` gate guards against.

Imports ONLY from :mod:`torch._inductor`, main inductor modules, and the
self-contained :mod:`~torch_spyre._inductor.reshard` package -- never flash-ws
``onchip_handoff`` / ``onchip_realize`` / ``restickify_ring`` (absent on main).
"""

from __future__ import annotations

import dataclasses

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Pointwise, Reduction

from . import config
from .constants import BATCH_MATMUL_OP
from .logging_utils import get_inductor_logger
from .pass_utils import (
    apply_splits_from_index_coeff,
    concretize_expr,
    find_reduction_var,
    get_mem_deps_from_rw,
    identify_matmul_inputs,
    iteration_space_from_op,
)
from .work_division import (
    _iter_computed_buffers,
    apply_splits,
    collect_tensor_deps,
)
from .reshard import LxFlip, splice_reshard
from .reshard.substrate import (
    DATAOP_LX_SIZE,
    LX_CAPACITY_BYTES,
    WORD_LENGTH,
    mixed_schedule,
)

logger = get_inductor_logger("reduction_reshard")

# A stick is 128 bytes = 64 fp16 elements (AIU 1.0). Source of truth: CLAUDE.md
# "Spyre Hardware Basics" + reshard.pieces.STICK_ELEMS.
STICK_SIZE = 64


# ============================================================================
# DETECTION (pre-scheduling pass)
# ============================================================================
#
# Co-bundle store: the reduction-reshard edges (producer_name, consumer_name)
# the planner found this compile. The scheduler's can_fuse_vertical queries this
# to fuse mul -> down_proj into ONE FusedSchedulerNode -> one device program,
# because LX does NOT persist across separate device programs (see
# reshard/substrate.py). Cleared per planner run; read by
# torch_spyre._inductor.scheduler.can_fuse_vertical.
_REDUCTION_RESHARD_EDGES: set[tuple[str, str]] = set()


def reduction_reshard_edges() -> set[tuple[str, str]]:
    """(producer_name, consumer_name) reduction-reshard edges from the last run."""
    return _REDUCTION_RESHARD_EDGES


def _decode_op_splits(op: ComputedBuffer) -> dict[str, int]:
    """Decode coeff-keyed ``op_it_space_splits`` into scheduler-symbol splits.

    Reimplements flash-ws ``restickify_ring.decode_op_splits`` over main's
    canonical ``apply_splits_from_index_coeff``. ``apply_splits`` only sets
    ``op_it_space_splits`` on ops the planner actually split (prod(splits) > 1),
    so a missing attribute means every dim defaults to split 1.
    """
    it_space = iteration_space_from_op(op)
    encoded = getattr(op, "op_it_space_splits", None)
    if encoded is None:
        return {str(sym): 1 for sym in it_space}
    rw = op.get_read_writes()
    write_index = next(iter(rw.writes)).index
    read_index = next(
        (dep.index for dep in rw.reads if isinstance(dep, MemoryDep)),
        write_index,
    )
    splits = apply_splits_from_index_coeff(
        encoded, write_index, read_index, it_space
    )
    return {str(sym): int(splits.get(sym, 1)) for sym in it_space}


def _is_reduction_input_edge(
    producer: ComputedBuffer,
    consumer: ComputedBuffer,
    consumer_read: MemoryDep,
) -> bool:
    """True for the genuine non-co-assignable reduction-input reshard edge.

    The SwiGLU ``mul -> down_proj`` edge: the consumer is a batchmatmul that
    reduces over ``K`` (the activation's stick/reduction dim), the producer
    co-splits the dim that feeds ``K`` (its ``out`` co-split, factor >= 2), and
    the consumer does NOT split ``K`` (it stays resident whole). This is exactly
    the edge flash-ws fail-closes on -- a 2-D co-split producer feeding a
    K-reduction consumer -- so the activation must be gathered LX -> RIU ring ->
    LX instead of round-tripping HBM.

    Distilled from flash-ws ``_is_reduction_input_edge`` +
    ``_consumer_reduction_symbols`` + ``_reduction_layout``. On main the genuine
    consumer is ALWAYS a batchmatmul, so ``identify_matmul_inputs`` +
    ``find_reduction_var`` give the activation read and ``K`` directly and
    robustly (handles M=1 constant-folding) rather than stride-diffing the
    write/read indices. We need ONE stride-coefficient match -- the K coeff in
    the consumer read equals a producer-write symbol coeff -- to confirm the
    producer co-split dim feeds K.
    """
    # (1) Consumer is the down_proj: a batchmatmul Reduction.
    if not isinstance(getattr(consumer, "data", None), Reduction):
        return False
    if consumer.data.reduction_type != BATCH_MATMUL_OP:
        return False

    # (2) cons_read must BE the matmul activation (x) input, and K = the dim it
    # reduces over.
    rw = consumer.get_read_writes()
    reads = [d for d in rw.reads if isinstance(d, MemoryDep)]
    if len(reads) != 2:
        return False
    out_dep = next(iter(rw.writes))
    x_dep, _y_dep = identify_matmul_inputs(reads, out_dep)
    if x_dep is None or x_dep.name != consumer_read.name:
        return False
    k_var = find_reduction_var(x_dep, out_dep)

    # (3) The consumer does NOT split the dim it reduces over (K resident whole).
    cons_splits = _decode_op_splits(consumer)
    if int(cons_splits.get(str(k_var), 1)) != 1:
        return False

    # (4) The producer co-splits the dim that maps onto K. The K coeff in the
    # consumer read is the buffer stride of K; the producer-write symbol with the
    # same stride is the producer dim that feeds K (inlined symbol correspondence:
    # the activation IS the producer's output buffer, so equal coeff == same dim).
    prod_write = next(iter(producer.get_read_writes().writes))
    k_coeff = consumer_read.index.coeff(k_var)
    if k_coeff == 0:
        return False
    prod_sym = next(
        (
            s
            for s in prod_write.index.free_symbols
            if prod_write.index.coeff(s) == k_coeff
        ),
        None,
    )
    if prod_sym is None:
        return False
    prod_splits = _decode_op_splits(producer)
    if int(prod_splits.get(str(prod_sym), 1)) < 2:
        return False

    # The edge must be a genuine CO-split (mb x out), not a pure-N or pure-M
    # split: at least two producer dims carry a non-unity split factor.
    return sum(1 for v in prod_splits.values() if v > 1) >= 2


def plan_reduction_reshard_edges(graph: GraphLowering) -> None:
    """Record the SwiGLU mul -> down_proj reduction-reshard edges for co-bundling.

    Pure observer pre-scheduling pass: walks every in-graph producer -> consumer
    activation edge and records the genuine reduction-input edge into
    ``_REDUCTION_RESHARD_EDGES``. Runs AFTER work division (so
    ``op.op_it_space_splits`` is committed) and BEFORE the Scheduler is built (so
    ``can_fuse_vertical`` can query the store). Gated on
    ``config.onchip_reduction_reshard``; cleared each run. One bad edge cannot
    crash compilation.

    Distilled from flash-ws ``plan_onchip_handoffs`` + ``run_onchip_handoff_planner``.
    """
    _REDUCTION_RESHARD_EDGES.clear()
    if not config.onchip_reduction_reshard:
        return

    ops = graph.operations
    name_to_op = {
        op.get_name(): op for op in ops if isinstance(op, ComputedBuffer)
    }
    for consumer in ops:
        if not isinstance(consumer, ComputedBuffer):
            continue
        for read_dep in consumer.get_read_writes().reads:
            if not isinstance(read_dep, MemoryDep):
                continue
            producer = name_to_op.get(read_dep.name)
            if producer is None:
                # Graph input / weight / extern -- no in-graph producer split.
                continue
            try:
                if _is_reduction_input_edge(producer, consumer, read_dep):
                    _REDUCTION_RESHARD_EDGES.add(
                        (producer.get_name(), consumer.get_name())
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "reduction_reshard skipping edge %s -> %s: %s: %s",
                    producer.get_name(),
                    consumer.get_name(),
                    type(exc).__name__,
                    exc,
                )
                continue

    if _REDUCTION_RESHARD_EDGES:
        logger.info(
            "reduction_reshard detected %d reduction-reshard edge(s): %s",
            len(_REDUCTION_RESHARD_EDGES),
            sorted(_REDUCTION_RESHARD_EDGES),
        )


# ============================================================================
# REALIZE (bundle SDSC mutator)
# ============================================================================
#
# Self-contained SDSC-JSON dict walkers (reimplemented verbatim from flash-ws
# onchip_realize; no flash-ws dependency). The SDSC body is a single-key dict
# whose value carries dscs_/datadscs_/coreIdToDscSchedule/opFuncsUsed_.


def _body(sdsc_json: dict) -> dict:
    return sdsc_json[next(iter(sdsc_json))]


def _dl_op(sdsc_json: dict) -> dict:
    """Return the single DL op dict of an SDSC body's first dsc."""
    dsc = _body(sdsc_json)["dscs_"][0]
    return dsc[next(iter(dsc))]


def _hbm_base(dl: dict, lds_idx: int) -> str | None:
    """Per-core[0] HBM base for the labeledDs allocate node, else None."""
    for node in dl["scheduleTree_"]:
        if node.get("nodeType_") == "allocate" and node.get("ldsIdx_") == lds_idx:
            if node.get("component_") != "hbm":
                return None
            return next(
                iter(node["startAddressCoreCorelet_"]["data_"].values()), None
            )
    return None


def _label_indices(labels: list[str]) -> list[int]:
    return [int(lbl.rsplit("-idx", 1)[1]) for lbl in labels]


def _producer_output_indices(dl: dict) -> list[int]:
    return _label_indices(dl["computeOp_"][0]["outputLabeledDs"])


def _consumer_input_indices(dl: dict) -> list[int]:
    return _label_indices(dl["computeOp_"][0]["inputLabeledDs"])


def _future_consumers(sdscs_json: list[dict], start: int, hbm_addr: str):
    """SDSCs after ``start`` reading ``hbm_addr`` with no later producer between.

    Returns ``[(pos, consumer_sdsc, consumer_in_idx), ...]``. Scratch HBM
    addresses are reused, so an input belongs to this producer only when no
    later producer between ``start`` and the consumer wrote the same address.
    """
    consumers = []
    for c in range(start + 1, len(sdscs_json)):
        cons = sdscs_json[c]
        cons_dl = _dl_op(cons)
        for in_idx in _consumer_input_indices(cons_dl):
            if _hbm_base(cons_dl, in_idx) != hbm_addr:
                continue
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


def _is_reduction_consumer(cons_sdsc: dict, expected_k: int) -> bool:
    """True iff ``cons_sdsc`` reduces over the producer's split dim ``expected_k``.

    The reshard handoff is valid ONLY for the genuine reduction consumer (the
    down-proj batchmatmul that reduces over the mul output's K dim). This gate
    rejects every co-assignable element-wise consumer in the SwiGLU chain
    (neg/exp/add/realdiv/mul: ``N_`` has no ``in_`` key) and the gate/up matmuls
    (batchmatmul, but ``N_.in_`` == 4096 != K -- they *produce* K, not reduce
    it). Without this gate the producer-only ``out`` extent == K test misfires on
    the within-bundle element-wise edges (device misfire: max_err 1.918).
    """
    try:
        dl = _dl_op(cons_sdsc)
    except (KeyError, IndexError, StopIteration):
        return False
    if dl.get("computeOp_", [{}])[0].get("opFuncName") != "batchmatmul":
        return False
    return int(dl.get("N_", {}).get("in_", -1)) == int(expected_k)


@dataclasses.dataclass(frozen=True)
class ReductionReshardRealization:
    """A realized 2-D core-to-core reduction reshard (the genuine ring move).

    A producer that co-splits two dims (``{mb, out}``) feeds a consumer that
    reduces over the producer's stick/split dim (``K`` not split): the SwiGLU
    ``mul -> down_proj`` edge. The datadscs/schedule come from the
    :mod:`~torch_spyre._inductor.reshard` substrate (the CPU-proven 2-D
    ``STCDPOpLx`` authoring).
    """

    producer_base: int
    consumer_base: int
    slice_bytes: int
    producer_flip: LxFlip
    consumer_flip: LxFlip
    datadscs: list
    opfuncs: list[str]
    schedule: dict
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

    The producer co-splits ``{mb:m_split, out:n_split}`` (its stick/``out`` dim
    IS the consumer's reduction ``K``); the consumer mb-bands ``{mb:num_cores}``
    and reduces over the full ``K`` (not split). The move is LX -> RIU ring ->
    LX. Delegates the device-program synthesis to the
    :mod:`~torch_spyre._inductor.reshard` substrate. Fail-closes (``None``) on any
    shape it cannot map exactly: a non-2-D producer co-split, a consumer that is
    not a single ``mb``-banded reduction, an uneven band split, or a per-core LX
    footprint that overflows ``capacity``.
    """
    from .reshard import (
        Band,
        Piece,
        allocate_lx_bases,
        build_asymmetric_reshard_bridge,
        build_consumer_pieces,
        build_perband_reshard_bridge,
        build_producer_pieces,
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

    return ReductionReshardRealization(
        producer_base=producer_base,
        consumer_base=consumer_base,
        slice_bytes=slice_bytes,
        producer_flip=LxFlip(producer_ldsidx, producer_base, "producer-output"),
        consumer_flip=LxFlip(consumer_ldsidx, consumer_base, "consumer-input"),
        datadscs=datadscs,
        opfuncs=opfuncs,
        schedule=schedule,
        perband=perband,
    )


def realize_reduction_reshard_bundle(
    sdscs_json: list[dict],
    *,
    m_rows: int,
    expected_k: int,
    m_split: int,
    n_split: int,
    num_cores: int,
    perband: bool = False,
    region0: int = 0,
) -> bool:
    """Realize the SwiGLU mul -> down_proj reduction reshard in the SDSC list.

    Walks producer SDSCs; for each producer-output HBM tensor whose logical
    ``out`` extent matches ``expected_k`` (the reduction dim the down-proj
    reduces over), finds the future consumer SDSC reading that same HBM base
    (:func:`_future_consumers`) and gates STRICTLY on the reduction consumer
    (:func:`_is_reduction_consumer`). Builds the 2-D reshard via
    :func:`realize_reduction_reshard`, then MIXED-folds the ``STCDPOpLx`` into the
    consumer (down_proj) SDSC via :func:`reshard.splice_reshard` (flips both
    endpoints to LX, attaches the STCDP datadscs + the mixed schedule). Returns
    True if any edge was realized. Fail-closed: a producer/consumer that does not
    match the pinned geometry is left untouched.

    Distilled from flash-ws ``realize_reduction_reshard_bundle``. The mixed fold
    is the dxp-accepted shape (a standalone pure-data-op SDSC aborts at
    ``dxp.cpp:479``); the ``_is_reduction_consumer`` gate stops the per-bundle
    hook misfiring on the within-bundle element-wise edges.
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
            # Gate STRICTLY to the reduction consumer: the down-proj batchmatmul
            # reducing over the producer's K dim. Without this, the producer-only
            # ``out`` extent == K test matches the within-bundle 12800-wide
            # element-wise edges (neg/exp/add/realdiv/mul) and the gate/up
            # matmuls, which are co-assignable and must NEVER be reshard-moved.
            if len(consumers) == 1 and _is_reduction_consumer(
                consumers[0][1], expected_k
            ):
                edge = (out_idx, consumers[0])
                break
        if edge is None:
            p += 1
            continue
        producer_out_idx, (_cons_pos, consumer_sdsc, consumer_in_idx) = edge

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
            region0=region0,
        )
        if realization is None:
            p += 1
            continue

        # MIXED fold into the consumer (down_proj) SDSC -- NOT a standalone step.
        # The patched dxp admits only the mixed shape (dataOpdscs_ + dscs_ +
        # coreIdToDscSchedule); splice_reshard flips both endpoints to LX,
        # attaches the STCDP datadscs + the MIXED schedule (data-ops then the
        # consumer DL row), and marks the consumer body mixed.
        splice_reshard(
            producer_sdsc=prod,
            consumer_sdsc=consumer_sdsc,
            producer_out_idx=producer_out_idx,
            consumer_in_idx=consumer_in_idx,
            producer_base=realization.producer_base,
            consumer_base=realization.consumer_base,
            datadscs=realization.datadscs,
            opfuncs=realization.opfuncs,
            schedule=mixed_schedule(len(realization.datadscs), num_cores),
        )
        realized = True
        p += 1
    return realized


# --- Co-assignment: the prerequisite that creates the reduction-reshard edge ----
#
# cost_model_matmul_division co-splits the MATMULS ({mb:4,out:8}) but the
# element-wise SwiGLU tail (neg/exp/add/realdiv/mul) is Pointwise -> the default
# work_distribution gives it pure-M {mb:32}. With the mul pure-M, the mul->down_proj
# edge is NOT a producer-co-split-on-the-reduction-dim edge, so the reduction
# detection cannot fire. Co-assignment propagates each matmul's {mb:4,out:8} split
# onto its split-agnostic Pointwise consumers (same-core, no move, value-correct),
# so the mul becomes {mb:4,out:8} and the genuine mul->down_proj reshard edge exists.
# Distilled from the core-to-core ab/coassign monkeypatch; runs as a pass inside
# _distribute_work (after cost_model_matmul_division, before work_distribution).


def _op_buf_names(op, kind: str) -> set:
    return {d.name for d in getattr(op.get_read_writes(), kind)}


def _recover_committed_split(op):
    """Reconstruct {symbol: count>1} from op.op_it_space_splits + its iter space."""
    it = iteration_space_from_op(op)
    rw = op.get_read_writes()
    wi = next(iter(rw.writes)).index
    ri = next((d.index for d in rw.reads), wi)
    split = apply_splits_from_index_coeff(op.op_it_space_splits, wi, ri, it)
    return {s: v for s, v in split.items() if v > 1}, it


def _map_split_by_extent(src_split, src_it, dst_it) -> dict:
    """Map a producer split onto a consumer iter space by matching dim extents."""
    out: dict = {}
    used: set = set()
    for ssym, cnt in src_split.items():
        ext = concretize_expr(src_it[ssym])
        for dsym, dext in dst_it.items():
            if dsym in used:
                continue
            if concretize_expr(dext) == ext:
                out[dsym] = cnt
                used.add(dsym)
                break
    return out


def _commit_coassign_split(op, split) -> None:
    rw = op.get_read_writes()
    args = get_mem_deps_from_rw(rw)
    _, output_td = collect_tensor_deps(op, args)
    apply_splits(op, split, output_td)


def coassign_elementwise(graph: GraphLowering, mm_ops: list) -> list:
    """Propagate each matmul's split onto its Pointwise consumer chain.

    Returns the newly co-assigned ops so ``_distribute_work`` adds them to
    ``work_distribution``'s preassigned set (they are then divided here, not by the
    default pure-M path). Value-correct: Pointwise ops are split-agnostic and the
    propagated split is mapped by matching dim extents, so each consumer core reads
    exactly the tile its own core produced (same-core, no data movement).
    """
    ops = list(_iter_computed_buffers(graph.operations))
    seen = {id(m) for m in mm_ops}
    extra: list = []
    frontier = [
        (m, _recover_committed_split(m))
        for m in mm_ops
        if getattr(m, "op_it_space_splits", None)
    ]
    for prod, (psplit, pit) in frontier:
        if not psplit:
            continue
        pbufs = _op_buf_names(prod, "writes")
        for op in ops:
            if id(op) in seen or not isinstance(op.data, Pointwise):
                continue
            if not (_op_buf_names(op, "reads") & pbufs):
                continue
            dit = iteration_space_from_op(op)
            csplit = _map_split_by_extent(psplit, pit, dit)
            if not csplit:
                continue
            _commit_coassign_split(op, csplit)
            seen.add(id(op))
            extra.append(op)
            frontier.append((op, (csplit, dit)))
            logger.info("coassign %s <- %s", op.get_name(), csplit)
    return extra

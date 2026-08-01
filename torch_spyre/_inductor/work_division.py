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


import dataclasses
import math
import itertools
from sympy import Expr, Integer, Symbol, divisors
from .ir import (
    SpyreConstantFallback,
    SpyreEmptyFallback,
    BroadcastAsyncFallback,
    WaitWorkFallback,
)
from torch._inductor.ir import (
    ComputedBuffer,
    DeviceCopy,
    ExternKernel,
    FallbackKernel,
    MultiOutput,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
)

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch_spyre._C import ElementArrangement

from .errors import Unsupported
from .constants import (
    BATCH_MATMUL_FP8_OP,
    BATCH_MATMUL_FP8MB_OP,
    DEVICE_NAME,
    MATMUL_REDUCTION_OPS,
    TOPK_OPS,
)
from .ir import FixedTiledLayout
from .pass_utils import (
    SchedNodeArg,
    finite_upper_or_none,
    compute_granularity,
    compute_max_size,
    concretize_expr,
    get_mem_deps_from_rw,
    device_coordinates,
    iteration_space_from_op,
    splits_by_index_coeff,
    apply_splits_from_index_coeff,
    op_read_writes,
)
from .propagate_hints import get_op_hints
from collections.abc import Iterable
from typing import Callable

from .logging_utils import get_inductor_logger
from . import config
import logging

logger = get_inductor_logger("work_division")

# Maximum memory-access span per core.
#
# MVLOC supports a maximum value of 65535, with each entry representing
# 4096 bytes. Therefore, the maximum addressable offset is:
# 65535 * 4096 = 268431360 bytes (255.996 MiB).
MAX_SPAN_BYTES = 65535 * 4096


@dataclasses.dataclass
class TensorDep:
    """Bundles a MemoryDep with its FixedTiledLayout and pre-computes device coordinates."""

    dep: MemoryDep
    layout: FixedTiledLayout
    device_coords: list[Expr] = dataclasses.field(init=False)

    def __post_init__(self):
        self.device_coords = device_coordinates(
            self.layout.device_layout, self.dep, None
        )


# Per-symbol (max_size, granularity) bucket metadata for symbolic iteration vars.
# Concrete iteration vars are absent from the dict — lookups default to the
# concrete ``concretize_expr`` path via ``_effective_size`` / ``_valid_divisor_basis``.
SymbolMeta = dict[Symbol, tuple[int, int]]


def _collect_symbol_metadata(it_space: dict[Symbol, Expr]) -> SymbolMeta:
    """Build ``{symbol: (max_size, granularity)}`` for opted-in symbolic dims.

    An iteration var is "opted in" iff the user passed
    ``mark_dynamic(max=...)`` -- that's exactly when ShapeEnv records a
    finite upper bound. Auto-dynamic symbols (Dynamo promoting an int on
    retrace when a Python loop varies it) have no finite max, so we skip
    them here and let them fall through to the existing
    ``concretize_expr`` + ``optimization_hint`` path.

    Concrete dims (no free symbols) are also omitted, so callers can use
    ``v in meta`` to detect both cases.
    """
    meta: SymbolMeta = {}
    for sym, expr in it_space.items():
        if not (hasattr(expr, "free_symbols") and expr.free_symbols):
            continue
        if finite_upper_or_none(expr) is None:
            logger.debug(
                f"[work_division/symbolic] skipping auto-dynamic symbol "
                f"{sym}; use mark_dynamic(max=...) to enable symbolic planning"
            )
            continue
        max_size = compute_max_size(expr)
        granularity = compute_granularity(expr, max_size)
        meta[sym] = (max_size, granularity)
    if meta:
        logger.info(
            "[work_division/symbolic] collected symbol_meta: "
            + ", ".join(f"{sym}=(max={ms}, gran={g})" for sym, (ms, g) in meta.items())
        )
    return meta


def _effective_size(v: Symbol, it_space: dict[Symbol, Expr], meta: SymbolMeta) -> int:
    """Return the canonical size of ``v`` for ranking and the span check.

    For symbolic dims, this is ``max_size`` — the worst-case runtime footprint
    that the compiled plan must remain legal against. For concrete dims, it
    is the concretized integer range.
    """
    if v in meta:
        return meta[v][0]
    return concretize_expr(it_space.get(v, 1))


def _valid_divisor_basis(
    v: Symbol, it_space: dict[Symbol, Expr], meta: SymbolMeta
) -> int:
    """Return the integer whose divisors are valid split counts for ``v``.

    For symbolic dims, this is ``granularity`` — the divisibility invariant
    ``n | granularity`` ensures ``R / n`` stays integer for every admissible
    runtime value ``R = granularity * k``. For concrete dims, it is just the
    concretized size.

    Absent dims (e.g. pool reduction dims ki/kj stripped from the
    work-division iteration space) return 1 — no valid split beyond 1,
    matching the hardware constraint that pool window dims are never split.
    """
    if v in meta:
        return meta[v][1]
    return concretize_expr(it_space.get(v, 1))


def core_split(size: int, max_cores: int) -> int:
    """
    Find the largest divisor of size that doesn't exceed max_cores.
    Args:
        size: The dimension size to split
        max_cores: Maximum number of cores to use for this dimension

    Returns:
        Number of cores to use (always divides size evenly)
    """
    for i in range(max_cores, 0, -1):
        if size % i == 0:
            return i
    return 1


def _most_splittable_dim(
    dims: list[Symbol],
    iteration_space: dict[Symbol, Expr],
    n_cores: int,
    symbol_meta: SymbolMeta,
) -> tuple[Symbol, int] | None:
    """Return (dim, split) for the dim in dims that maximises core_split(size, n_cores).

    Returns None if no dim yields a split > 1. ``symbol_meta`` is required —
    pass an empty dict for fully-concrete iteration spaces.
    """
    best_dim, best_split = None, 0
    for d in dims:
        s = core_split(_valid_divisor_basis(d, iteration_space, symbol_meta), n_cores)
        if s > best_split:
            best_dim, best_split = d, s
    return (best_dim, best_split) if best_split > 1 else None


def coordinate_mask_blocked_vars(
    reduction_vars: Iterable[Symbol],
    stick_vars: dict[Symbol, int],
    it_space: dict[Symbol, Expr],
) -> set[Symbol]:
    """Return reduction stick vars that cannot be split across cores.

    The backend compiler cannot apply coordinate masking to a dimension spread
    over cores, and masking is applied to a dim that is padded, reduced, and the
    stick dim (mirrors ``_get_coordinate_mask`` in codegen/superdsc.py). A stick
    var is guaranteed non-symbolic, so its element count concretizes; it is
    padded iff not stick-aligned.

    ``it_space`` must be the element-valued iteration space (not the
    stick-adjusted copy), since padding is defined on element counts.
    ``stick_vars`` maps each stick var to its elems_per_stick (as returned by
    ``adjust_it_space_for_sticks``).

    Both work-division paths consult this: the greedy path drops these from its
    reduction candidates, and ``enumerate_work_division_candidates`` rejects any
    split that divides one of them.
    """
    return {
        v
        for v in reduction_vars
        if v in stick_vars and concretize_expr(it_space[v]) % stick_vars[v] != 0
    }


def multi_dim_iteration_space_split(
    iteration_space: dict[Symbol, Expr],
    max_cores: int,
    output_dims: list[Symbol],
    reduction_dims: list[Symbol],
    min_splits: dict[Symbol, int] | None = None,
    symbol_meta: SymbolMeta | None = None,
) -> dict[Symbol, int]:
    """Distribute max_cores across the iteration space.

    Three-pass algorithm:
      1. Satisfy min_splits (span-reduction commitments).
      2. Distribute remaining cores to output_dims in priority order.
      3. If this is a reduction op, pick the single most-splittable reduction dim
         for any remaining cores.

    ``symbol_meta`` carries ``(max_size, granularity)`` for any symbolic dim;
    when a dim is symbolic, ``core_split`` is fed ``granularity`` instead of
    the concretised size so the chosen split divides every admissible runtime
    bucket evenly.

    The product of all splits will be <= max_cores.
    """
    symbol_meta = symbol_meta or {}
    is_reduction_included = bool(reduction_dims)

    splits = {v: 1 for v in iteration_space.keys()}
    n_cores_remaining = max_cores

    if min_splits:
        # Sanity check: making sure that reduction_dims list is cleared up if
        #               any reduction dim is already selected during span reduction
        assert (
            not is_reduction_included  # empty
            or not any(v in min_splits for v in reduction_dims)  # no overlap
        )

        for var, min_split in min_splits.items():
            assert var not in output_dims and var not in reduction_dims

            if n_cores_remaining // min_split <= 0:
                logger.critical(
                    f"Cannot satisfy minimum split requirement for {var}: "
                    f"need {min_split} splits but only {n_cores_remaining} cores remaining. "
                    f"Skipping this constraint - hardware span limit may be violated."
                )
                continue
            splits[var] = min_split
            n_cores_remaining = n_cores_remaining // min_split

    for v in output_dims:
        if n_cores_remaining <= 1:
            break
        # Symbolic dims use granularity (divisibility invariant); concrete
        # dims use the concretised size. _valid_divisor_basis picks per dim.
        # TODO(issue#1372): remaining concrete sites use concretize_expr; once
        #                   symbolic work division is end-to-end, this comment
        #                   can be dropped.
        basis = _valid_divisor_basis(v, iteration_space, symbol_meta)
        best_split = core_split(basis, n_cores_remaining)
        if v in symbol_meta:
            logger.info(
                f"[work_division/symbolic] dim {v} (symbolic, max="
                f"{symbol_meta[v][0]}, gran={symbol_meta[v][1]}): "
                f"core_split(basis={basis}, n_cores={n_cores_remaining}) = "
                f"{best_split}"
            )
        if best_split > 1:
            splits[v] = best_split
            n_cores_remaining = n_cores_remaining // best_split

    if is_reduction_included and n_cores_remaining > 1:
        result = _most_splittable_dim(
            reduction_dims, iteration_space, n_cores_remaining, symbol_meta
        )
        if result is not None:
            best_dim, best_split = result
            splits[best_dim] = best_split

    return splits


_FP8_COMPOUND_EAS = (ElementArrangement.QFP8MB, ElementArrangement.QFP8WT)


def _has_fp8_compound_tensor(tds: list[TensorDep]) -> bool:
    """Check if any tensor uses a DD2 compound FP8 stick."""
    return any(
        hasattr(td.layout.device_layout, "element_arrangement")
        and td.layout.device_layout.element_arrangement in _FP8_COMPOUND_EAS
        for td in tds
    )


def _is_qfp8wt_tensor(td: TensorDep) -> bool:
    """Check if a specific tensor has QFP8WT element arrangement."""
    return (
        hasattr(td.layout.device_layout, "element_arrangement")
        and td.layout.device_layout.element_arrangement == ElementArrangement.QFP8WT
    )


def _physical_output_split_is_legal(
    logical_elements: int,
    split: int,
    physical_group_elements: int,
) -> bool:
    """Whether every output slice starts on a physical storage-group boundary.

    A QFP8WT stick contains the documented logical ``[K:2, N:64]`` factors,
    but Torch-Spyre stores that 128-element compound stick as one indivisible
    terminal device group.  A logical N split can therefore divide the 64-wide
    output-stick count and still be impossible to express as a QFP8WT base
    address.  This is the exact case for Granite gate/up N=12800 with N split 8:
    200 FP16 output sticks divide by 8, while 100 physical FP8 weight groups do
    not.  Codegen would otherwise gcd-clamp the requested split after planning.
    """
    if min(logical_elements, split, physical_group_elements) < 1:
        return False
    physical_groups = (
        logical_elements + physical_group_elements - 1
    ) // physical_group_elements
    return physical_groups % split == 0


def _qfp8wt_n_split_is_legal(
    n_elements: int,
    n_split: int,
    n_dim: Symbol,
    input_tds: list[TensorDep],
) -> bool:
    """Check the RHS QFP8WT storage boundary for a candidate N split."""
    for td in input_tds:
        if not _is_qfp8wt_tensor(td):
            continue
        if not any(n_dim in coord.free_symbols for coord in td.device_coords):
            continue
        physical_group_elements = int(td.layout.device_layout.device_size[-1])
        if not _physical_output_split_is_legal(
            n_elements, n_split, physical_group_elements
        ):
            return False
    return True


def _compound_stick_granularities(td: TensorDep) -> dict[Symbol, int]:
    """Return logical atomic sizes encoded by one compound FP8 stick."""
    ea = td.layout.device_layout.element_arrangement
    if ea == ElementArrangement.QFP8MB:
        coords_and_factors = zip(td.device_coords[-3:], (8, 2, 8))
    elif ea == ElementArrangement.QFP8WT:
        coords_and_factors = zip(td.device_coords[-2:], (2, 64))
    else:
        return {}

    granularities: dict[Symbol, int] = {}
    for coord, factor in coords_and_factors:
        if len(coord.free_symbols) != 1:
            continue
        var = next(iter(coord.free_symbols))
        granularities[var] = granularities.get(var, 1) * factor
    return granularities


def _get_fp8_compound_split_constraints(
    input_tds: list[TensorDep],
    output_td: TensorDep,
) -> dict[Symbol, int]:
    """Prevent cross-core K splitting for DD2 compound FP8 operands.

    QFP8WT's non-terminal K:2 factor and QFP8MB's repeated K:8 factors
    identify the reduction dimension. M remains splittable at granularity 2.
    """
    constraints: dict[Symbol, int] = {}
    all_tds = input_tds + [output_td]
    if not _has_fp8_compound_tensor(all_tds):
        return constraints

    for td in all_tds:
        ea = td.layout.device_layout.element_arrangement
        if ea == ElementArrangement.QFP8WT:
            for var in td.device_coords[-2].free_symbols:
                constraints[var] = 1
        elif ea == ElementArrangement.QFP8MB:
            occurrences: dict[Symbol, int] = {}
            for coord in td.device_coords[-3:]:
                for var in coord.free_symbols:
                    occurrences[var] = occurrences.get(var, 0) + 1
            for var, count in occurrences.items():
                if count > 1:
                    constraints[var] = 1

    return constraints


def adjust_it_space_for_sticks(
    it_space: dict[Symbol, Expr],
    tensor_deps: list[TensorDep],
    symbol_meta: SymbolMeta | None = None,
) -> tuple[dict[Symbol, Expr], dict[Symbol, int]]:
    """
    Return a copy of it_space with stick variables converted from elements to
    sticks, plus a dict mapping each stick variable to its max element per stick
    value.

    For each tensor, find the variable that indexes its stick dimension and
    convert its size in it_space from elements to sticks. This ensures work
    division treats sticks as atomic units.

    For DD2 compound FP8 layouts, each logical dimension uses its physical
    atomic size: QFP8MB M=2/K=64 and QFP8WT K=2/N=64.

    When tensors of different dtypes share a stick variable (e.g. a float16
    input and an int64 argmax output), the largest elems_per_stick is used
    so the adjustment is conservative (fewer sticks → smaller adjusted size →
    fewer cores assigned to the stick dimension).

    TODO: As of now, the stick dim cannot be symbolic. Granularity
    on a symbolic stick var would have to additionally be a multiple of
    ``elems_per_stick`` for the stick-count conversion to stay coherent; that
    is out of scope here. Raises ``Unsupported`` if any tensor's stick dim
    maps to a symbolic iteration variable.

    The original it_space is not mutated.
    """
    symbol_meta = symbol_meta or {}

    # Pass 1: find the largest elems_per_stick per stick variable.
    adjusted_space = dict(it_space)
    max_elems: dict[Symbol, int] = {}
    for td in tensor_deps:
        # Handle DD2 compound FP8 sticks. A logical dimension can occur more
        # than once in the physical stick (QFP8MB K:8 * K:8 = K:64).
        if (
            hasattr(td.layout.device_layout, "element_arrangement")
            and td.layout.device_layout.element_arrangement in _FP8_COMPOUND_EAS
        ):
            for stick_var, atomic_elems in _compound_stick_granularities(td).items():
                if stick_var not in adjusted_space:
                    continue
                if stick_var in symbol_meta:
                    raise Unsupported(
                        f"symbolic compound-stick dim {stick_var} is not "
                        f"supported yet (tensor {td.dep.name})"
                    )
                max_elems[stick_var] = max(max_elems.get(stick_var, 1), atomic_elems)
            continue

        stick_expr = td.device_coords[-1]
        if len(stick_expr.free_symbols) != 1:
            continue
        stick_var = next(iter(stick_expr.free_symbols))
        if stick_var not in adjusted_space:
            continue
        if stick_var in symbol_meta:
            logger.info(
                f"[work_division/symbolic] stick-dim guard raised: "
                f"stick_var={stick_var} on tensor {td.dep.name} is symbolic"
            )
            raise Unsupported(
                f"symbolic stick dim {stick_var} is not supported yet "
                f"(tensor {td.dep.name}); symbolic dims must be non-stick "
                f"(e.g. the leading batch dim)."
            )
        elems_per_stick = td.layout.device_layout.elems_per_stick()
        if stick_var not in max_elems or elems_per_stick > max_elems[stick_var]:
            max_elems[stick_var] = elems_per_stick

    # Pass 2: adjust each variable once using the maximum.
    for stick_var, elems_per_stick in max_elems.items():
        # FIXME: here we assume padding to a full stick. It may not always be
        #        the case and we should use a more robust way of computing the
        #        number of sticks
        adjusted_space[stick_var] = (
            adjusted_space[stick_var] + elems_per_stick - 1
        ) // elems_per_stick

    return adjusted_space, max_elems


def get_per_core_span(
    td: TensorDep,
    splits: dict[Symbol, int],
    it_space_orig: dict[Symbol, Expr],
    symbol_meta: SymbolMeta,
) -> int:
    """Compute per-core memory span in bytes for a tensor under the given splits.

    coordinate expressions from compute_coordinates() in views.py are sums of
    independent single-variable terms, so max of the full expression equals the
    sum of per-variable maxima obtained by zeroing out all other variables.
    min is always 0 since all variables start at 0. If this invariant in
    compute_coordinates() ever changes, this logic must be revisited.

    it_space_orig must be the original element-valued ranges, not the
    stick-adjusted copy, because device coordinate expressions are written in
    terms of element indices.

    For symbolic dims, the per-dim range ``R`` is the ``max_size`` from
    ``symbol_meta`` divided by the dim's split count — the worst-case runtime
    footprint that any compiled plan must remain legal against.
    """
    device_size = td.layout.device_layout.device_size
    itemsize = td.layout.dtype.itemsize
    for d, coord in enumerate(td.device_coords[:-1]):
        if not coord.free_symbols:
            continue
        per_core_max = 0
        per_core_min = 0
        for v in coord.free_symbols:
            term = coord.subs({u: 0 for u in coord.free_symbols - {v}})
            # Per-core span is a hardware-bound quantity that must be checked
            # against MAX_SPAN_BYTES. For symbolic dims we use ``max_size``
            # (the worst-case footprint, also the HBM allocation footprint).
            R = _effective_size(v, it_space_orig, symbol_meta) // splits.get(v, 1)
            per_core_max += int(term.subs(v, R - 1))
            per_core_min += int(term.subs(v, 0))
        per_core_size = per_core_max - per_core_min + 1
        if per_core_size > 1:
            stride_elems = math.prod(device_size[d + 1 :])
            return per_core_size * stride_elems * itemsize
    return itemsize


def warn_if_per_core_overflow(
    tensor_deps: list[TensorDep],
    it_space_orig: dict[Symbol, Expr],
    splits: dict[Symbol, int],
    op_name: str,
    symbol_meta: SymbolMeta,
) -> None:
    """Log CRITICAL if any tensor's per-core memory span exceeds MAX_SPAN_BYTES."""
    for td in tensor_deps:
        per_core_span = get_per_core_span(td, splits, it_space_orig, symbol_meta)
        if per_core_span > MAX_SPAN_BYTES:
            dl = td.layout.device_layout
            logger.critical(
                f"{op_name}: per-core tensor span "
                f"{per_core_span / (1024 * 1024):.3f} MB "
                f"(shape={list(td.layout.size)}, dtype={td.layout.dtype}, "
                f"device_size={list(dl.device_size)}, splits={splits}) "
                f"exceeds hardware limit of {MAX_SPAN_BYTES / (1024 * 1024):.2f} MB"
            )


def must_split_vars(
    tensor_deps: list[TensorDep],
    it_space_orig: dict[Symbol, Expr],
    it_space_adjusted: dict[Symbol, Expr],
    stick_vars: dict[Symbol, int],
    max_cores: int,
    symbol_meta: SymbolMeta,
) -> dict[Symbol, int]:
    """Return the minimum splits per iteration variable to keep each tensor's
    memory span within MAX_SPAN_BYTES.

    Processes tensors one at a time, carrying accumulated_splits forward so
    splits committed for one tensor reduce the search space for subsequent ones.
    For each violating tensor, iterates device dimensions outer to inner and
    searches for the joint split combination (Cartesian product over contributing
    variables) that brings the span closest to (but not exceeding) MAX_SPAN_BYTES.
    If no combo satisfies the limit, picks the one that minimizes the span.
    Gives up on a dimension when the committed splits still leave it evaluating
    to > 1, meaning inner dimensions cannot reduce the span further.

    For symbolic dims, ``symbol_meta`` supplies ``(max_size, granularity)``.
    The Cartesian search enumerates ``divisors(granularity)`` (not
    ``divisors(max_size)``) so every chosen split divides every admissible
    runtime bucket evenly. The span check itself uses ``max_size`` as the
    worst-case footprint.

    Args:
        tensor_deps: List of tensor dependencies to check
        it_space_orig: Original iteration space (element-valued)
        it_space_adjusted: Adjusted iteration space (stick-valued for stick vars)
        stick_vars: Mapping of stick variables to elements per stick
        max_cores: Maximum number of cores available
        symbol_meta: Per-symbol (max_size, granularity) for symbolic dims

    Returns a dict mapping Symbol -> number of slices.
    """
    # TODO: use compute_max_size(...) / compute_granularity(...) from pass_utils.py
    # for symbolic path. Refer to #2287 for details.
    accumulated_splits: dict[Symbol, int] = {}

    for td in tensor_deps:
        if (
            get_per_core_span(td, accumulated_splits, it_space_orig, symbol_meta)
            <= MAX_SPAN_BYTES
        ):
            continue

        for coord in td.device_coords[:-1]:
            # Concretize for the ``> 1`` comparison: with symbolic ranges,
            # ``s0 > 1`` returns a sympy Relational whose truth value is
            # undefined.  Span filtering here is a structural decision that
            # needs a concrete answer.
            # TODO(issue#1372): Symbolic work division will keep this symbolic.
            split_vars = [
                v
                for v in coord.free_symbols
                if _effective_size(v, it_space_orig, symbol_meta) > 1
            ]
            if not split_vars:
                continue

            def valid_splits(v: Symbol) -> list[int]:
                current_min = accumulated_splits.get(v, 1)
                if v in symbol_meta:
                    # Symbolic dim: split must divide granularity for the
                    # n | granularity divisibility invariant to hold across
                    # every admissible runtime bucket.
                    basis = symbol_meta[v][1]
                    return [s for s in divisors(basis) if s >= current_min]
                if v in stick_vars:
                    stick_count = concretize_expr(it_space_adjusted[v])
                    return [s for s in divisors(stick_count) if s >= current_min]
                return [
                    s
                    for s in divisors(concretize_expr(it_space_orig[v]))
                    if s >= current_min
                ]

            var_divisors = [valid_splits(v) for v in split_vars]

            for v, candidates in zip(split_vars, var_divisors):
                if not candidates:
                    raise Unsupported(
                        f"No valid split for variable {v} "
                        f"(orig_size={_effective_size(v, it_space_orig, symbol_meta)}, "
                        f"min_required={accumulated_splits.get(v, 1)}) "
                        f"for tensor {td.dep.name}."
                    )

            # NOTE: Exhaustive search of all combinations. It's probably ok
            #       assuming the search space is small. Can revisit if this
            #       becomes a bottleneck.
            #
            # Two-tier selection by span value:
            #   - Within-limit combos: prefer largest span (= fewest cores used)
            #   - Above-limit combos: prefer smallest span (= most progress)
            best_within: tuple[int, tuple] | None = None  # (span, combo)
            best_above: tuple[int, tuple] | None = None  # (span, combo)

            for combo in itertools.product(*var_divisors):
                trial = dict(accumulated_splits)
                for v, s in zip(split_vars, combo):
                    trial[v] = s

                if math.prod(trial.values()) > max_cores:
                    continue

                span = get_per_core_span(td, trial, it_space_orig, symbol_meta)

                if span <= MAX_SPAN_BYTES:
                    if best_within is None or span > best_within[0]:
                        best_within = (span, combo)
                else:
                    if best_above is None or span < best_above[0]:
                        best_above = (span, combo)

            # Prefer within-limit; fall back to best partial progress
            best = best_within or best_above

            if best is None:
                logger.warning(
                    f"No valid split combo found for tensor {td.dep.name} "
                    f"coord={coord} under accumulated_splits={accumulated_splits}. "
                    f"Skipping."
                )
                break

            best_span, best_combo = best
            for v, s in zip(split_vars, best_combo):
                accumulated_splits[v] = s

            if best_span <= MAX_SPAN_BYTES:
                break

            # Still above the limit. If this coord still evaluates to > 1 under
            # the committed splits, inner dimensions cannot reduce the span further.
            # Use _effective_size so symbolic dims substitute their max_size
            # rather than a misleading optimization_hint.
            per_core_coord_size = (
                max(
                    int(
                        coord.subs(
                            {
                                v: _effective_size(v, it_space_orig, symbol_meta)
                                // accumulated_splits.get(v, 1)
                                - 1
                                for v in coord.free_symbols
                            }
                        )
                    ),
                    0,
                )
                + 1
            )
            if per_core_coord_size > 1:
                logger.warning(
                    f"Cannot satisfy span limit for tensor {td.dep.name}: "
                    f"coord={coord} still evaluates to {per_core_coord_size} after splits. "
                    f"Inner dimensions cannot reduce span further. "
                    f"Best span={best_span}, limit={MAX_SPAN_BYTES}."
                )
                break

    return accumulated_splits


def prioritize_dimensions(
    output: TensorDep,
    it_space_adjusted: dict[Symbol, Expr],
    symbol_meta: SymbolMeta | None = None,
) -> tuple[list[Symbol], list[Symbol]]:
    """Partition iteration variables into output dims and reduction dims.

    Output dims are those whose symbols appear in the output tensor's device
    coordinate expressions (excluding the stick coordinate). Reduction dims are
    the remainder. Both lists are sorted by decreasing size — for symbolic
    dims the canonical size is ``max_size`` from ``symbol_meta``, preserving
    the existing "largest-output-dim-first" policy under the extension that a
    symbolic dim's size is its bucket upper bound.

    Variables already committed as min_splits should be filtered out of
    it_space_adjusted before calling this function.
    """
    symbol_meta = symbol_meta or {}
    coord_vars = {v for e in output.device_coords[:-1] for v in e.free_symbols}

    output_pairs: list[tuple[Symbol, Expr]] = []
    reduction_pairs: list[tuple[Symbol, Expr]] = []
    for s, e in it_space_adjusted.items():
        (output_pairs if s in coord_vars else reduction_pairs).append((s, e))

    # Sort by decreasing size (concrete for static dims, max_size for symbolic).
    def _size_key(t: tuple[Symbol, Expr]) -> int:
        sym, _ = t
        return _effective_size(sym, it_space_adjusted, symbol_meta)

    output_pairs.sort(key=_size_key, reverse=True)
    reduction_pairs.sort(key=_size_key, reverse=True)

    return [t[0] for t in output_pairs], [t[0] for t in reduction_pairs]


def _resolve_layout(op: ComputedBuffer) -> "FixedTiledLayout":
    """Return the FixedTiledLayout for op, unwrapping MutationLayoutSHOULDREMOVE.

    Mutation ops keep MutationLayoutSHOULDREMOVE at pre-scheduler time so the
    scheduler can identify them as in-place writes.  Their target buffer already
    has a FixedTiledLayout assigned by propagate_spyre_tensor_layouts, so
    real_layout() gives us the correct device layout for work division.
    """
    layout = op.get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        layout = layout.real_layout()
    assert isinstance(layout, FixedTiledLayout), (
        f"Expected FixedTiledLayout for {op.get_name()}, got {type(layout)}"
    )
    return layout


def collect_tensor_deps(
    op: ComputedBuffer, args: list[SchedNodeArg]
) -> tuple[list[TensorDep], TensorDep]:
    """Build TensorDep lists for inputs and the output of op."""
    input_tds = [TensorDep(a.dep, a.layout) for a in args]
    rw = op_read_writes(op)
    output_td = TensorDep(next(iter(rw.writes)), _resolve_layout(op))
    return input_tds, output_td


def apply_splits(
    op: ComputedBuffer,
    splits: dict,
    output_td: TensorDep,
) -> None:
    """Commit splits to op.

    Does nothing when the product of splits is 1 (no parallelism).
    """
    cores_used = math.prod(splits.values())
    if cores_used <= 1:
        return

    rw = op_read_writes(op)
    write_index = output_td.dep.index
    first_read = next(iter(rw.reads), None)
    read_index = first_read.index if first_read is not None else write_index
    op.op_it_space_splits = splits_by_index_coeff(splits, write_index, read_index)


def enumerate_work_division_candidates(
    op: ComputedBuffer,
    max_cores: int,
) -> list[dict[Symbol, int]]:
    """Return every permissible core-division split for ``op``.

    A split (``dict[Symbol, int]``, same form as :func:`apply_splits`) is
    permissible iff: each per-dim factor divides its dim's size,
    ``prod(factors) <= max_cores``, every tensor's per-core span
    ``<= MAX_SPAN_BYTES``, and at most one reduction (K) dim is split. A factor
    of ``1`` means the dim is unsplit; the all-ones single-core split is
    included when it is itself permissible.

    Only ``Pointwise`` / ``Reduction`` ops have a divisible iteration space;
    ``TOPK`` reductions run single-core, so they yield only the unsplit split.
    """
    # TODO: Enumerate compute bound ops and for seeds or compute optimized
    # work division where HBM bandwidth can saturate compute.

    it_space = iteration_space_from_op(op)

    # TOPK reductions run single-core (see divide_reduction_op): the only
    # permissible "division" is the unsplit one.
    if isinstance(op.data, Reduction) and op.data.reduction_type in TOPK_OPS:
        return [{v: 1 for v in it_space}]

    input_tds, output_td = collect_tensor_deps(
        op, get_mem_deps_from_rw(op_read_writes(op))
    )
    all_tds = input_tds + [output_td]

    symbol_meta = _collect_symbol_metadata(it_space)
    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(
        it_space, all_tds, symbol_meta
    )

    # Reduction (K) dims are the iteration vars absent from the output tensor's
    # device coordinates (mirrors prioritize_dimensions / splits_by_index_coeff).
    coord_vars = {v for e in output_td.device_coords[:-1] for v in e.free_symbols}
    reduction_vars = [v for v in it_space_adjusted if v not in coord_vars]
    mask_blocked = coordinate_mask_blocked_vars(reduction_vars, stick_vars, it_space)

    # Per-dim candidate factors, mirroring must_split_vars.valid_splits but with
    # no ``>= current_min`` floor (we want the full set, including 1).
    def factors(v: Symbol) -> list[int]:
        if v in symbol_meta:
            basis = symbol_meta[v][1]  # granularity
        elif v in stick_vars:
            basis = concretize_expr(it_space_adjusted[v])  # stick count
        else:
            basis = concretize_expr(it_space[v])  # element count
        return [int(s) for s in divisors(basis)]

    def create_splits(axis_vars, combo):
        return dict(zip(axis_vars, combo))

    def valid_split(splits):
        if math.prod(splits.values()) > max_cores:  # core budget
            return False
        if sum(1 for v in reduction_vars if splits[v] > 1) > 1:  # <= 1 K-split
            return False
        if any(  # per-core span <= MAX_SPAN_BYTES, on element-valued it_space
            get_per_core_span(td, splits, it_space, symbol_meta) > MAX_SPAN_BYTES
            for td in all_tds
        ):
            return False
        if any(  # a coordinate-masked dim cannot be split across cores
            splits[v] > 1 for v in mask_blocked
        ):
            return False
        return True

    vars_ = list(it_space_adjusted.keys())
    candidates = [
        splits
        for combo in itertools.product(*(factors(v) for v in vars_))
        if valid_split(splits := create_splits(vars_, combo))
    ]

    return candidates


def _work_div_hint_by_name(op: ComputedBuffer) -> dict[str, int]:
    dim_to_split: dict[str, int] = {}
    for _, hint_dict in sorted(get_op_hints(op).items()):
        dim_to_split.update(hint_dict.get("work_div") or {})
    return dim_to_split


def _has_work_div_hint(op: ComputedBuffer) -> bool:
    return any(hint_dict.get("work_div") for hint_dict in get_op_hints(op).values())


def _resolve_work_div_hint(
    op: ComputedBuffer,
    it_space: dict[Symbol, Expr],
) -> dict[Symbol, int] | None:
    dim_to_split = _work_div_hint_by_name(op)
    if not dim_to_split:
        return None

    loop_var_dims = getattr(op, "work_div_loop_info", {})
    splits: dict[Symbol, int] = {}
    for name, split in dim_to_split.items():
        for sym in it_space:
            if sym in splits:
                continue
            if name in loop_var_dims.get(sym, []):
                splits[sym] = split
                break
    return splits if splits else None


def _apply_user_hint(
    op: ComputedBuffer,
    user_splits: dict[Symbol, int],
    it_space_adjusted: dict[Symbol, Expr],
    output_td: TensorDep,
    max_cores: int,
) -> dict[Symbol, int]:
    """Apply splits in insertion order, pruning lower-priority overflows."""
    op_name = op.get_name()

    splits: dict[Symbol, int] = {}
    cores_used = 1
    loop_var_dims = getattr(op, "work_div_loop_info", {})
    for sym, split_val in user_splits.items():
        # bool is an int subclass in Python, but it is not a meaningful split.
        if isinstance(split_val, bool) or not isinstance(split_val, (int, Integer)):
            raise Unsupported(
                f"work_division_hint: {op_name} split value {split_val!r} "
                f"for dim {sym} must be an integer."
            )
        split = int(split_val)
        if split < 1:
            raise Unsupported(
                f"work_division_hint: {op_name} split value {split!r} "
                f"for dim {sym} must be positive."
            )
        if sym not in it_space_adjusted:
            raise Unsupported(
                f"work_division_hint: {op_name} dim {sym} is not in the "
                f"work-division iteration space."
            )

        next_cores = cores_used * split
        if next_cores > max_cores:
            logger.info(
                "work_division_hint: %s skipping named dim(s) %s (split=%s) "
                "because cores would be %s, exceeding SENCORES=%s",
                op_name,
                loop_var_dims.get(sym, []),
                split,
                next_cores,
                max_cores,
            )
            continue

        dim_size = concretize_expr(it_space_adjusted[sym])
        if dim_size % split != 0:
            raise Unsupported(
                f"work_division_hint: {op_name} dim {sym} size={dim_size} "
                f"is not evenly divisible by split={split}."
            )

        splits[sym] = split
        cores_used = next_cores

    coord_vars = {v for e in output_td.device_coords[:-1] for v in e.free_symbols}
    reduction_vars_to_split = {
        sym for sym, split in splits.items() if split > 1 and sym not in coord_vars
    }
    if len(reduction_vars_to_split) > 1:
        raise Unsupported(
            f"work_division_hint: {op_name} splits "
            f"{len(reduction_vars_to_split)} reduction dimensions "
            f"({reduction_vars_to_split}), but the backend supports at most 1."
        )

    return splits


def _commit_user_splits(
    op: ComputedBuffer,
    splits: dict[Symbol, int],
    output_td: TensorDep,
) -> None:
    if math.prod(splits.values()) <= 1:
        if hasattr(op, "op_it_space_splits"):
            delattr(op, "op_it_space_splits")
        return
    apply_splits(op, splits, output_td)


def span_reduction_pass(
    op: ComputedBuffer,
    args: list[SchedNodeArg],
    max_cores: int,
) -> None:
    """Mandatory per-op pass: compute minimum splits to satisfy the MAX_SPAN_BYTES.

    Writes results to op.op_it_space_splits. If no span violation exists,
    op.op_it_space_splits is left unset (apply_splits is a no-op for splits <= 1).
    """
    # Pre-scheduling may be run more than once on the same IR while layouts are
    # refined. A split left by the previous work-distribution pass is not a
    # mandatory span split for this invocation. Clear it before recomputing,
    # otherwise the cost-model pass mistakes the stale split for a hard bound.
    if hasattr(op, "op_it_space_splits"):
        delattr(op, "op_it_space_splits")

    it_space = iteration_space_from_op(op)
    input_tds, output_td = collect_tensor_deps(op, args)
    all_tds = input_tds + [output_td]

    # Symbolic-dim bucket metadata for the iteration vars in this op. Built
    # before stick adjustment so the stick-dim guard inside
    # adjust_it_space_for_sticks can raise on symbolic stick dims.
    symbol_meta = _collect_symbol_metadata(it_space)

    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(
        it_space, all_tds, symbol_meta
    )
    min_splits = must_split_vars(
        all_tds, it_space, it_space_adjusted, stick_vars, max_cores, symbol_meta
    )

    # DD2 compound FP8 reductions cannot split K across cores.
    if (
        isinstance(op.data, Reduction)
        and op.data.reduction_type in MATMUL_REDUCTION_OPS
    ):
        qfp8_constraints = _get_fp8_compound_split_constraints(input_tds, output_td)
        min_splits.update(qfp8_constraints)

    coord_vars = {v for e in output_td.device_coords[:-1] for v in e.free_symbols}
    reduction_vars_to_split = set(min_splits) - coord_vars
    # Each entry in Reduction.reduction_ranges maps to at most one Symbol via
    # index_vars_squeeze (size-1 entries are squeezed away). So len > 1 means
    # genuinely distinct reduction dimensions, not multiple symbols from one dim.
    if len(reduction_vars_to_split) > 1:
        raise Unsupported(
            f"Cannot satisfy hardware memory span limit "
            f"({MAX_SPAN_BYTES / (1024**2):.3f}MB) without splitting "
            f"{len(reduction_vars_to_split)} reduction dimension(s) "
            f"({reduction_vars_to_split}), but the backend supports at most 1."
        )

    apply_splits(op, min_splits, output_td)

    if symbol_meta and math.prod(min_splits.values()) > 1:
        logger.info(
            f"[work_division/symbolic] span_reduction {op.get_name()}: "
            f"committed min_splits={ {str(k): v for k, v in min_splits.items()} }, "
            f"cores={math.prod(min_splits.values())}"
        )
    if logger.isEnabledFor(logging.DEBUG) and math.prod(min_splits.values()) > 1:
        logger.debug(
            f"span_reduction work_division {op.get_name()}: cores={math.prod(min_splits.values())}, "
            f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
            f"symbol_meta={symbol_meta}, "
            f"priorities=[], min_splits={min_splits}, "
            f"op_it_space_splits={op.op_it_space_splits}"
        )


def _default_split(
    it_space_adjusted: dict[Symbol, Expr],
    output_td: TensorDep,
    committed_splits: dict[Symbol, int],
    max_cores: int,
    symbol_meta: SymbolMeta,
    blocked: set[Symbol],
) -> tuple[dict[Symbol, int], list[Symbol], list[Symbol]]:
    """Distribute max_cores by priority on top of span_reduction's commits.

    Returns the chosen splits and the (output, reduction) priority dims the
    caller logs. Shared by work_distribution_pass and cost_model_matmul_division.
    """
    # TODO: The final dim committed by span_reduction_pass holds the minimum
    #       split that gets the span under the limit, so it may have headroom
    #       for additional parallelism (outer dims committed before it are
    #       already maximally split and have no headroom). Excluding it here
    #       leaves that parallelism on the table when other dims can't absorb
    #       the remaining cores.
    it_space_remaining = {
        s: e for s, e in it_space_adjusted.items() if s not in committed_splits
    }
    if len(output_td.device_coords) >= 2 and _is_qfp8wt_tensor(output_td):
        stick_coord_vars = set(output_td.device_coords[-2].free_symbols)
        it_space_remaining = {
            s: e for s, e in it_space_remaining.items() if s not in stick_coord_vars
        }
    output_dims, reduction_dims = prioritize_dimensions(
        output_td, it_space_remaining, symbol_meta
    )

    # If span_reduction_pass already committed a reduction split, suppress further
    # reduction splitting so the final result never exceeds one reduction dim split.
    coord_vars = {v for e in output_td.device_coords[:-1] for v in e.free_symbols}
    if any(v not in coord_vars for v in committed_splits):
        reduction_dims = []

    # Drop reduction dims the backend compiler can't split across cores before
    # the greedy distributor commits them.
    reduction_dims = [v for v in reduction_dims if v not in blocked]

    # Pass max_cores, not remaining_cores: multi_dim_iteration_space_split
    # accounts for committed_splits in its first pass, consuming those cores
    # itself before distributing the rest by priority.
    splits = multi_dim_iteration_space_split(
        it_space_adjusted,
        max_cores,
        output_dims,
        reduction_dims,
        committed_splits,
        symbol_meta,
    )
    return splits, output_dims, reduction_dims


def work_distribution_pass(
    op: ComputedBuffer,
    args: list[SchedNodeArg],
    max_cores: int,
) -> None:
    """Optional per-op pass: distribute remaining cores to maximize parallelism.

    Reads op.op_it_space_splits written by span_reduction_pass (if any) to
    recover the already-committed splits, then fills remaining cores by priority.
    """
    it_space = iteration_space_from_op(op)
    input_tds, output_td = collect_tensor_deps(op, args)
    all_tds = input_tds + [output_td]

    symbol_meta = _collect_symbol_metadata(it_space)

    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(
        it_space, all_tds, symbol_meta
    )

    # Recover splits committed by span_reduction_pass using the same
    # coeff-keyed encoding that codegen uses — stable across passes.
    if hasattr(op, "op_it_space_splits"):
        rw = op_read_writes(op)
        write_index = next(iter(rw.writes)).index
        read_index = next((d.index for d in rw.reads), write_index)
        min_splits = apply_splits_from_index_coeff(
            op.op_it_space_splits, write_index, read_index, it_space
        )
    else:
        min_splits = {}

    # apply_splits_from_index_coeff returns 1 for every unsplit dim; keep only
    # dims with actual committed splits so they don't overlap with priorities.
    committed_splits = {s: v for s, v in min_splits.items() if v > 1}

    # DD2 compound FP8 reductions cannot split K across cores.
    qfp8_constraints = _get_fp8_compound_split_constraints(input_tds, output_td)
    for var, split_val in qfp8_constraints.items():
        if var in committed_splits:
            del committed_splits[var]
        min_splits[var] = split_val

    if not config.ignore_work_division_hints:
        user_splits = _resolve_work_div_hint(op, it_space_adjusted)
        if user_splits is not None:
            user_splits = _apply_user_hint(
                op, user_splits, it_space_adjusted, output_td, max_cores
            )
            dropped = {
                s: v for s, v in committed_splits.items() if user_splits.get(s, 1) < v
            }
            if dropped:
                logger.warning(
                    f"work_division_hint: {op.get_name()} user hint reduces "
                    f"splits committed by span_reduction for dims {list(dropped)}. "
                    f"Applying strict user hint; this may violate the hardware "
                    f"{MAX_SPAN_BYTES / (1024**2):.3f} MB span limit."
                )
            _commit_user_splits(op, user_splits, output_td)

            if logger.isEnabledFor(logging.DEBUG):
                op_splits: tuple[dict, dict] = getattr(
                    op, "op_it_space_splits", ({}, {})
                )
                logger.debug(
                    f"work_distribution(user-hint) work_division {op.get_name()}: "
                    f"cores={math.prod(user_splits.values())}, "
                    f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
                    f"min_splits={committed_splits}, user_splits={user_splits}, "
                    f"op_it_space_splits={op_splits}"
                )
            warn_if_per_core_overflow(
                all_tds, it_space, user_splits, op.get_name(), symbol_meta
            )
            return

    coord_vars = {v for e in output_td.device_coords[:-1] for v in e.free_symbols}
    reduction_vars = [v for v in it_space_adjusted if v not in coord_vars]
    blocked = coordinate_mask_blocked_vars(reduction_vars, stick_vars, it_space)
    splits, output_dims, reduction_dims = _default_split(
        it_space_adjusted, output_td, committed_splits, max_cores, symbol_meta, blocked
    )

    # Enforce compound-stick split constraints on the final split.
    qfp8_constraints = _get_fp8_compound_split_constraints(input_tds, output_td)
    splits.update(qfp8_constraints)

    apply_splits(op, splits, output_td)

    if symbol_meta and math.prod(splits.values()) > 1:
        logger.info(
            f"[work_division/symbolic] work_distribution {op.get_name()}: "
            f"final splits={ {str(k): v for k, v in splits.items()} }, "
            f"cores={math.prod(splits.values())}, "
            f"priorities={[str(d) for d in output_dims + reduction_dims]}"
        )
    if logger.isEnabledFor(logging.DEBUG) and math.prod(splits.values()) > 1:
        logger.debug(
            f"work_distribution work_division {op.get_name()}: cores={math.prod(splits.values())}, "
            f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
            f"symbol_meta={symbol_meta}, "
            f"priorities={output_dims + reduction_dims}, min_splits={committed_splits}, "
            f"op_it_space_splits={op.op_it_space_splits}"
        )

    warn_if_per_core_overflow(all_tds, it_space, splits, op.get_name(), symbol_meta)


def _is_fp8_scale_op(op: ComputedBuffer) -> bool:
    """Return whether ``op`` is the specialized scaled-matmul scale pass.

    Keep this check local to the custom op.  General pointwise operators may
    legitimately want a different division from their producer, so producer
    continuity must not silently become the default work-distribution policy.
    """
    origins = getattr(op.data, "origins", ())
    return any(
        str(getattr(origin, "target", "")) == "spyre.apply_fp8_scale.default"
        for origin in origins
    )


def _map_producer_output_splits(
    producer_output_splits: dict[Expr, int],
    consumer_read_index: Expr,
    consumer_it_space: dict[Symbol, Expr],
) -> dict[Symbol, int] | None:
    """Map a producer's coefficient-keyed output division onto a consumer read.

    ``op_it_space_splits`` deliberately keys output dimensions by their flat
    tensor-index coefficient so a split survives scheduler variable renaming.
    The consumer's read of that tensor supplies the inverse mapping here.
    Return ``None`` rather than inheriting a partial division: losing one axis
    would change ownership and defeat the purpose of this optimization.
    """
    mapped: dict[Symbol, int] = {}
    matched_coeffs: set[Expr] = set()
    for symbol in consumer_it_space:
        coeff = consumer_read_index.coeff(symbol)
        if coeff in producer_output_splits:
            mapped[symbol] = producer_output_splits[coeff]
            matched_coeffs.add(coeff)

    if matched_coeffs != set(producer_output_splits):
        return None
    return mapped


def _inherit_fp8_scale_work_division(
    graph: GraphLowering,
    op: ComputedBuffer,
    args: list[SchedNodeArg],
    max_cores: int,
) -> bool:
    """Keep a scaled-matmul scale pass on its full-size producer's core grid.

    The FP8 matmul cost model partitions the MxN output deliberately.  Applying
    a row or column scale is elementwise over that same MxN tensor; assigning a
    new grid makes cores exchange ownership between the reduction and each
    scale pass.  Inherit only from a same-shaped computed input and re-check all
    local legality conditions before committing the split.
    """
    if not _is_fp8_scale_op(op):
        return False
    if not config.ignore_work_division_hints and _has_work_div_hint(op):
        return False

    op_size = list(op.get_size())
    it_space = iteration_space_from_op(op)
    _, output_td = collect_tensor_deps(op, args)

    for arg in args:
        producer = graph.get_buffer(arg.dep.name)
        if not isinstance(producer, ComputedBuffer):
            continue
        if list(producer.get_size()) != op_size:
            continue
        producer_coeff_splits = getattr(producer, "op_it_space_splits", None)
        if not producer_coeff_splits:
            continue

        producer_output_splits, producer_reduction_splits = producer_coeff_splits
        if not producer_output_splits:
            continue
        # A scale pass has no reduction axis and therefore cannot preserve a
        # producer K split.  DD2 compound FP8 matmul forbids K splitting, but
        # make that required invariant explicit instead of dropping it.
        if producer_reduction_splits:
            continue
        inherited = _map_producer_output_splits(
            producer_output_splits, arg.dep.index, it_space
        )
        if inherited is None:
            continue

        # Unmentioned consumer dimensions stay unsplit.
        splits = {symbol: inherited.get(symbol, 1) for symbol in it_space}
        cores = math.prod(splits.values())
        if cores <= 1 or cores > max_cores:
            continue

        # Reuse the canonical candidate enumerator so stick-count divisibility,
        # coordinate masks, reduction constraints, and address-span legality
        # stay in one place.  Logical divisibility alone is not sufficient.
        legal_candidates = enumerate_work_division_candidates(op, max_cores)
        if splits not in legal_candidates:
            continue

        apply_splits(op, splits, output_td)
        logger.debug(
            "fp8_scale_continuity work_division %s: producer=%s cores=%s "
            "splits=%s op_it_space_splits=%s",
            op.get_name(),
            producer.get_name(),
            cores,
            splits,
            op.op_it_space_splits,
        )
        return True

    return False


_PT_ROWS = 8  # PT block rows per corelet


@dataclasses.dataclass(frozen=True)
class _MatmulCostProfile:
    """Precision-specific inputs to the analytic matmul cost model.

    The remaining coefficients below describe DD2 scheduling effects shared by
    both precisions. These fields are the quantities that change mechanically
    when FMA becomes FMA8 and the two input tensors become one byte per value.
    The two tile targets are device-fit coefficients expressed in logical
    elements, so they must also be precision-specific.
    """

    peak_macs_us_core: float
    lhs_bytes: int
    rhs_bytes: int
    output_bytes: int
    target_pt_passes: int
    target_m_tie_passes: int
    target_n_tile_elems: int


# Constants for the matmul cost model (_matmul_split_cost). Each is either an
# AIU hardware limit or a coefficient fit to measured device kernel times.
_PT_EFFICIENCY_EXPONENT = 0.25
_M_MIN = _PT_ROWS // 2  # below half a PT pass an m-split buys nothing
_DL16_PEAK_MACS_US_CORE = (98.304e12 / 2 / 32) / 1e6
_DL16_MATMUL_COST_PROFILE = _MatmulCostProfile(
    peak_macs_us_core=_DL16_PEAK_MACS_US_CORE,
    lhs_bytes=2,
    rhs_bytes=2,
    output_bytes=2,
    target_pt_passes=5,
    target_m_tie_passes=4,
    target_n_tile_elems=512,
)
_FP8_MATMUL_COST_PROFILE = _MatmulCostProfile(
    # FMA8 performs two FP8 products per 16-bit PT lane. Accumulation and the
    # exposed output remain 16-bit, while both streamed operands are one byte.
    peak_macs_us_core=2 * _DL16_PEAK_MACS_US_CORE,
    lhs_bytes=1,
    rhs_bytes=1,
    output_bytes=2,
    target_pt_passes=5,
    # DD2 Q/O oracle fit: retain four M cohorts through M=512, then expose
    # eight at M>=1024. Keep this explicitly profile-local until it is
    # validated on projection families beyond K=N=4096.
    target_m_tie_passes=16,
    # A 1024-element FP8 weight tile occupies the same bytes as the calibrated
    # 512-element DL16 tile.
    target_n_tile_elems=1024,
)
_HBM_BW_GBS = 204.8  # LPDDR5 aggregate peak bandwidth
_PSUM_PER_CORE_ELEM_US = 1.0e-3
_BMM_PSUM_PER_CORE_ELEM_US = 1.0e-4
_COHORT_LIMIT = 8  # cores sharing a broadcast before it contends for bandwidth
_COHORT_PENALTY_EXPONENT = 0.75
_M_LANE_UNDERUSE_PENALTY_US = 10.0  # tie-break when too few M lanes are used
_M_TILE_UNDERFILL_TARGET = 16  # rows/core below this pay PT startup overhead
_M_TILE_UNDERFILL_PENALTY_US = 30.0
_WIDE_N_TILE_PENALTY_US = 25.0  # per log2 step over the profile's N target
_CORE_UNDERUSE_PENALTY_US = (
    150.0  # soft replacement for the old hard full-core fallback
)
_BMM_BATCH_SPLIT_PENALTY_US = 10.0  # true-BMM batch split cost per log2 step
_LARGE_M_TILE_SHAPE_PENALTY_US = 20.0
_SHARED_DOWN_N_SPLIT_PENALTY_US = 10.0


def _matmul_split_cost(
    b_axis: tuple[int, int],
    m_axis: tuple[int, int],
    n_axis: tuple[int, int],
    k_axis: tuple[int, int],
    max_cores: int,
    shared_weight: bool = False,
    profile: _MatmulCostProfile = _DL16_MATMUL_COST_PROFILE,
) -> float:
    """Estimated kernel time in microseconds for ``[B,M,K]@[B,K,N]`` run with
    the given core split. Each axis is a ``(size, split)`` pair so a dim's size
    cannot be paired with another dim's split. Lower is better; inf if infeasible.
    """
    (B, b), (M, m), (N, n), (K, k) = b_axis, m_axis, n_axis, k_axis
    cores_used = b * m * n * k
    if cores_used == 0 or cores_used > max_cores:
        return float("inf")

    # Compute: per-core MACs over peak, derated when the per-core M tile is too
    # short to fill the PT pipeline. The PT array streams M in passes of
    # _PT_ROWS; below the profile's target number of passes its startup/drain
    # overhead is amortised over too little work and grows sub-linearly.
    m_t = M // m if m else 1
    pt_passes = max(1.0, m_t / _PT_ROWS)
    pt_eff = min(
        1.0,
        (pt_passes / profile.target_pt_passes) ** _PT_EFFICIENCY_EXPONENT,
    )
    compute_us = (B * M * N * K / cores_used) / (profile.peak_macs_us_core * pt_eff)

    # HBM: every input operand is broadcast to the cohort of cores splitting the
    # orthogonal dim. Past _COHORT_LIMIT the broadcasts contend for the shared
    # link, so effective bandwidth falls off linearly with cohort size.
    weight_batches = 1 if shared_weight else B
    bytes_total = (
        B * M * K * profile.lhs_bytes
        + weight_batches * K * N * profile.rhs_bytes
        + B * M * N * profile.output_bytes
    )
    fanout_split = max(m, n) if shared_weight else n
    cohort_penalty = max(
        1.0, (fanout_split / _COHORT_LIMIT) ** _COHORT_PENALTY_EXPONENT
    )
    hbm_us = bytes_total / (_HBM_BW_GBS * 1000) * cohort_penalty

    # PSUM: a K-split spreads the reduction over k cores, costing (k-1)
    # partial-sum hops. Charge each core's output tile rather than the whole
    # output, so useful K-splits are not over-penalized.
    psum_coeff = _PSUM_PER_CORE_ELEM_US if shared_weight else _BMM_PSUM_PER_CORE_ELEM_US
    output_elems_per_core = (B * M * N) / max(1, b * m * n)
    psum_us = max(0, k - 1) * output_elems_per_core * psum_coeff

    # Tie-break: among compute-equivalent splits prefer exposing enough M lanes
    # to stream work over the stationary weight tile. PT efficiency above handles
    # the opposite case where an M split makes each per-core tile too short.
    target_m = max(
        _M_MIN,
        min(
            max_cores // 2,
            max(1, M // (profile.target_m_tie_passes * _PT_ROWS)),
        ),
    )
    m_lane_underuse_us = (
        max(0.0, math.log2(target_m / max(1, m))) * _M_LANE_UNDERUSE_PENALTY_US
    )
    m_tile_underfill_us = (
        max(0.0, math.log2(_M_TILE_UNDERFILL_TARGET / max(1, m_t)))
        * _M_TILE_UNDERFILL_PENALTY_US
    )

    # Very wide output tiles lose schedule efficiency in the generated kernel.
    # Charge only the over-wide side so short-M and small/moderate-N choices
    # are not pulled away from PT-friendly M tiles.
    n_t = N // n if n else N
    wide_n_us = (
        max(0.0, math.log2(max(1, n_t) / profile.target_n_tile_elems))
        * _WIDE_N_TILE_PENALTY_US
    )

    # Once M is large enough to feed the PT, prefer tile shapes that avoid
    # unnecessary layout/fusion fallout. For true BMMs this means avoiding
    # splitting a tiny output dimension when the reduction dimension is much
    # larger (value-matmul geometry: K >> N). For shared-weight projections this
    # means avoiding very wide per-core N tiles when the whole projection is
    # narrow enough that more N lanes are available. Both effects are expressed
    # as ratios rather than op names or workload-specific shapes.
    filled_m_tile_factor = 1.0 if m_t >= _M_TILE_UNDERFILL_TARGET else 0.0
    true_bmm_value_split_us = (
        0.0
        if shared_weight or n <= 1
        else filled_m_tile_factor
        * max(0.0, math.log2(max(1, K) / max(1, N)))
        * math.log2(n)
        * _LARGE_M_TILE_SHAPE_PENALTY_US
    )
    shared_narrow_tile_us = (
        0.0
        if not shared_weight
        else filled_m_tile_factor
        * max(
            0.0,
            math.log2((profile.target_n_tile_elems * _COHORT_LIMIT) / max(1, N)),
        )
        * max(
            0.0,
            math.log2(max(1, n_t) / (profile.target_n_tile_elems // 4)),
        )
        * (_LARGE_M_TILE_SHAPE_PENALTY_US / 4)
    )
    shared_down_n_split_us = (
        0.0
        if not shared_weight or n <= 1
        else max(0.0, math.log2(max(1, K) / max(1, N)))
        * math.log2(n)
        * _SHARED_DOWN_N_SPLIT_PENALTY_US
    )
    large_m_tile_shape_us = (
        true_bmm_value_split_us + shared_narrow_tile_us + shared_down_n_split_us
    )

    # Prefer using the full core budget, but keep this soft so measured-good
    # lower-core candidates can still win.
    core_underuse_us = (
        max(0.0, math.log2(max_cores / cores_used)) * _CORE_UNDERUSE_PENALTY_US
    )

    # True BMMs often need batch parallelism to avoid tiny-M underfill. Charge a
    # small additive split overhead instead of multiplying the whole estimate.
    batch_split_us = (
        0.0 if shared_weight else math.log2(max(1, b)) * _BMM_BATCH_SPLIT_PENALTY_US
    )

    return (
        compute_us
        + hbm_us
        + psum_us
        + m_lane_underuse_us
        + m_tile_underfill_us
        + wide_n_us
        + large_m_tile_shape_us
        + core_underuse_us
        + batch_split_us
    )


def _single_input_row_dims(
    row_dims: list[Symbol],
    input_tds: list[TensorDep],
) -> list[Symbol]:
    def _appears_in_one_input(dim: Symbol) -> bool:
        hits = sum(
            dim in {v for e in td.device_coords for v in e.free_symbols}
            for td in input_tds
        )
        return hits == 1

    return [d for d in row_dims if _appears_in_one_input(d)]


def _pick_innermost_output_dim(
    dims: list[Symbol],
    output_index: Expr,
) -> Symbol | None:
    """Pick the row dim nearest the output's contiguous/stick dimension.

    Shared 2D weights are represented as broadcast views. Their stride-0 batch
    dimensions disappear from device coordinates, so both batch dims and the
    true M dim can look like "LHS-only" row dims. In the output's flat host
    index, the true M dim is the innermost row dimension: it has the smallest
    non-zero coefficient, while batch dims have coefficients multiplied by M
    and/or outer batch extents.
    """

    candidates: list[tuple[int, Symbol]] = []
    for dim in dims:
        coeff = output_index.coeff(dim)
        if coeff == 0:
            continue
        candidates.append((abs(concretize_expr(coeff)), dim))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _cost_model_matmul_planner(
    op: ComputedBuffer,
    splits: dict[Symbol, int],
    it_space_adjusted: dict[Symbol, Expr],
    output_td: TensorDep,
    stick_vars: dict[Symbol, int],
    committed_splits: dict[Symbol, int],
    max_cores: int,
    input_tds: list[TensorDep],
) -> dict[Symbol, int]:
    """Override the default split for a matmul / bmm with the lowest-cost
    feasible (b, m, n, k) per _matmul_split_cost.

    Returns ``splits`` unchanged for anything this planner does not model:
    non-matmuls, ops with a span-committed split already in place, multi-K
    matmuls, or a chosen split that would use fewer cores than the default.
    """
    if not isinstance(op.data, Reduction):
        return splits
    if op.data.reduction_type not in MATMUL_REDUCTION_OPS:
        return splits
    force_fp8_lx_grid = op.data.reduction_type in (
        BATCH_MATMUL_FP8_OP,
        BATCH_MATMUL_FP8MB_OP,
    ) and (config.fp8_lx_poc_m_split or config.fp8_lx_poc_n_split)
    if committed_splits and not force_fp8_lx_grid:
        return splits

    # Classify the output coord dims from the output's own terminal stick
    # coordinate. Using the union-wide ``stick_vars`` here is incorrect for
    # QFP8MB: its activation stick makes M atomic in pairs even though M is
    # still an output row dimension. The old test therefore saw both M and N
    # as stick dims, declined to model the op, and left it to the generic
    # one-dimensional distributor.
    output_coord_vars = {
        v for e in output_td.device_coords[:-1] for v in e.free_symbols
    }
    ordered_output_coord_vars = [d for d in it_space_adjusted if d in output_coord_vars]
    output_stick_vars = output_td.device_coords[-1].free_symbols
    n_dims = [d for d in ordered_output_coord_vars if d in output_stick_vars]
    row_dims = [d for d in ordered_output_coord_vars if d not in output_stick_vars]
    if len(n_dims) != 1 or not row_dims:
        return splits
    n_dim = n_dims[0]

    m_candidates = _single_input_row_dims(row_dims, input_tds)
    rhs_loaded_once = False
    if len(m_candidates) == 1:
        m_dim = m_candidates[0]
    elif len(m_candidates) > 1:
        m_dim = _pick_innermost_output_dim(m_candidates, output_td.dep.index)
        if m_dim is None:
            return splits
        rhs_loaded_once = True
    else:
        return splits
    batch_dims = [d for d in row_dims if d != m_dim]
    # Folded projection matmuls have no batch dims left by this stage, but the
    # RHS is still an unbatched weight that is loaded once. Treat them like the
    # broadcast/shared-weight path for cost purposes while keeping true BMMs
    # where RHS depends on batch dims on the non-shared path.
    if not batch_dims:
        rhs_loaded_once = True

    # K is the lone reduction dim (anything else this planner does not model).
    reduction = [d for d in it_space_adjusted if d not in output_coord_vars]
    if len(reduction) != 1:
        return splits
    k_dim = reduction[0]

    # Any dimension touched by a physical stick is measured in atomic sticks
    # in ``it_space_adjusted``. Restore logical element counts for the byte and
    # MAC model, while continuing to enumerate split factors from the adjusted
    # counts so a core never divides a QFP8MB row pair or another stick atom.
    M_e = concretize_expr(it_space_adjusted[m_dim]) * stick_vars.get(m_dim, 1)
    n_sticks = concretize_expr(it_space_adjusted[n_dim])
    k_sticks = concretize_expr(it_space_adjusted[k_dim])
    N_e = n_sticks * stick_vars.get(n_dim, 1)
    K_e = k_sticks * stick_vars.get(k_dim, 1)

    batch_sizes = [concretize_expr(it_space_adjusted[bd]) for bd in batch_dims]
    B_total = math.prod(batch_sizes)

    b_combos = (
        list(itertools.product(*([int(d) for d in divisors(s)] for s in batch_sizes)))
        if batch_dims
        else [()]
    )
    m_divs = [int(d) for d in divisors(concretize_expr(it_space_adjusted[m_dim]))]
    n_divs = [int(d) for d in divisors(n_sticks)]
    k_divs = [int(d) for d in divisors(k_sticks)]

    # DD2 compound FP8 matmul cannot split K across cores.
    if _has_fp8_compound_tensor(input_tds + [output_td]):
        k_divs = [1]

    profile = (
        _FP8_MATMUL_COST_PROFILE
        if op.data.reduction_type in (BATCH_MATMUL_FP8_OP, BATCH_MATMUL_FP8MB_OP)
        else _DL16_MATMUL_COST_PROFILE
    )

    if force_fp8_lx_grid:
        m_split = config.fp8_lx_poc_m_split
        n_split = config.fp8_lx_poc_n_split
        if min(m_split, n_split) < 1:
            raise Unsupported(
                "TORCH_SPYRE_FP8_LX_POC_M_SPLIT and "
                "TORCH_SPYRE_FP8_LX_POC_N_SPLIT must both be positive"
            )
        if batch_dims:
            raise Unsupported(
                "the private FP8/LX work-division override currently supports "
                "only a 2D shared-weight matmul"
            )
        if m_split not in m_divs or n_split not in n_divs:
            raise Unsupported(
                f"FP8/LX PoC grid {m_split}x{n_split} does not divide "
                f"M={M_e}, N={N_e} at the DD2 stick granularity"
            )
        if m_split * n_split > max_cores:
            raise Unsupported(
                f"FP8/LX PoC grid {m_split}x{n_split} uses more than "
                f"SENCORES={max_cores}"
            )
        if not _qfp8wt_n_split_is_legal(N_e, n_split, n_dim, input_tds):
            raise Unsupported(
                f"FP8/LX PoC N split {n_split} cuts a QFP8WT physical group"
            )

        forced = dict(splits)
        forced[m_dim] = m_split
        forced[n_dim] = n_split
        forced[k_dim] = 1
        logger.info(
            "fp8_lx_poc work_division %s: overriding span split %s with M=%s N=%s K=1",
            op.get_name(),
            committed_splits,
            m_split,
            n_split,
        )
        return forced

    best = None
    best_cost = float("inf")
    for b_combo in b_combos:
        b_prod = math.prod(b_combo)
        for mm in m_divs:
            for nn in n_divs:
                if not _qfp8wt_n_split_is_legal(N_e, nn, n_dim, input_tds):
                    continue
                for kk in k_divs:
                    if b_prod * mm * nn * kk > max_cores:
                        continue
                    c = _matmul_split_cost(
                        (B_total, b_prod),
                        (M_e, mm),
                        (N_e, nn),
                        (K_e, kk),
                        max_cores,
                        shared_weight=rhs_loaded_once,
                        profile=profile,
                    )
                    if c < best_cost:
                        best_cost = c
                        best = (b_combo, mm, nn, kk)

    if best is None:
        return splits

    b_combo, m_s, n_s, k_s = best
    new_splits = dict(splits)
    for bd, bs in zip(batch_dims, b_combo):
        new_splits[bd] = int(bs)
    new_splits[m_dim] = m_s
    new_splits[n_dim] = n_s
    new_splits[k_dim] = k_s

    # Never trade down to fewer cores than the default distributor already found.
    if math.prod(new_splits.values()) < math.prod(splits.values()):
        if not _has_fp8_compound_tensor(input_tds + [output_td]):
            return splits
        # For compound FP8, force k_dim = 1 regardless of core count.
        new_splits[k_dim] = 1

    logger.debug(
        f"cost_model work_division {op.get_name()}: "
        f"b={b_combo} m={m_s} n={n_s} k={k_s} rhs_loaded_once={rhs_loaded_once} "
        f"cost={best_cost:.1f}us "
        f"[B={B_total} M={M_e} K={K_e} N={N_e}]"
    )
    return new_splits


def divide_pointwise_op(
    op: ComputedBuffer,
    args: list[SchedNodeArg],
    max_cores: int,
    pass_fn: Callable,
) -> None:
    pass_fn(op, args, max_cores)


def divide_reduction_op(
    op: ComputedBuffer,
    args: list[SchedNodeArg],
    max_cores: int,
    pass_fn: Callable,
) -> None:
    red: Reduction = op.data

    # Currently we support Topk for k<=4, which can be handled efficiently on single core
    # TODO: Modification will be required to enable Topk for k>4
    if red.reduction_type in TOPK_OPS:
        if not config.ignore_work_division_hints and _has_work_div_hint(op):
            logger.warning(
                f"work_division_hint: {op.get_name()} ignores work_div hint "
                f"because TOPK reductions run single-core."
            )
        return

    pass_fn(op, args, max_cores)


def _validate_max_cores() -> int:
    max_cores = config.sencores
    if max_cores > 32 or max_cores < 1:
        raise Unsupported(f"invalid SENCORES value {max_cores}")
    return max_cores


def _iter_computed_buffers(operations: list[Operation]):
    """Yield ComputedBuffer ops, handling FallbackKernel/ExternKernel dispatch."""
    for op in operations:
        if op.is_no_op():
            pass
        elif isinstance(op, ComputedBuffer):
            layout = op.maybe_get_layout()
            if layout is None or layout.device.type != DEVICE_NAME:
                continue
            yield op
        elif isinstance(op, FallbackKernel):
            # FallbackKernel produces 0..N trailing MultiOutputs
            # (see torch_spyre/_inductor/propagate_layouts.py).
            # Work division is not supported on either; the MultiOutputs
            # are skipped in their own branch below.
            pass
        elif isinstance(op, MultiOutput):
            pass
        elif isinstance(op, ExternKernel):
            if isinstance(op, (SpyreConstantFallback, SpyreEmptyFallback, DeviceCopy)):
                # Work division not supported on allocation/constant kernels, nor
                # on DeviceCopy.
                pass
            elif isinstance(op, (BroadcastAsyncFallback, WaitWorkFallback)):
                # Work division not supported on broadcast kernels
                pass
            else:
                logger.warning(f"unhandled node type {type(op)}")
        else:
            logger.warning(f"unhandled operation type {type(op)}")


def _apply_input_layout_overrides(
    op: ComputedBuffer, args: list[SchedNodeArg]
) -> list[SchedNodeArg]:
    """Apply per-op input layout overrides stored in op._input_layout_overrides.

    insert_post_mutation_restickify uses this to make work division treat an
    input buffer with an override layout instead of its committed layout.

    The same tag is also used by SpyreKernel.create_tensor_arg, so work
    division and codegen agree on the input layout.
    """
    overrides: dict[str, FixedTiledLayout] = getattr(op, "_input_layout_overrides", {})
    if not overrides:
        return args
    return [SchedNodeArg(a.dep, overrides.get(a.dep.name, a.layout)) for a in args]


def span_reduction(graph: GraphLowering) -> None:
    """Pass 1: compute minimum per-op splits required by MAX_SPAN_BYTES."""
    operations = graph.operations
    max_cores = _validate_max_cores()
    for op in _iter_computed_buffers(operations):
        rw = op_read_writes(op)
        args = _apply_input_layout_overrides(op, get_mem_deps_from_rw(rw))
        if isinstance(op.data, Pointwise):
            divide_pointwise_op(op, args, max_cores, span_reduction_pass)
        elif isinstance(op.data, Reduction):
            divide_reduction_op(op, args, max_cores, span_reduction_pass)


def work_distribution(
    graph: GraphLowering, preassigned_ops: list[Operation] | None = None
) -> None:
    """Pass 3: distribute remaining cores across ops to maximize parallelism.

    Ops in `preassigned_ops` were already divided by cost_model_matmul_division;
    they are left untouched so every op is divided by exactly one pass.
    """
    operations = graph.operations
    preassigned_ops = preassigned_ops or []
    max_cores = _validate_max_cores()
    for op in _iter_computed_buffers(operations):
        if op in preassigned_ops:
            continue
        rw = op_read_writes(op)
        args = _apply_input_layout_overrides(op, get_mem_deps_from_rw(rw))
        if isinstance(op.data, Pointwise):
            if _inherit_fp8_scale_work_division(graph, op, args, max_cores):
                continue
            divide_pointwise_op(op, args, max_cores, work_distribution_pass)
        elif isinstance(op.data, Reduction):
            divide_reduction_op(op, args, max_cores, work_distribution_pass)


def _cost_model_divide_op(op: ComputedBuffer, max_cores: int) -> bool:
    """Re-price one matmul's split with the analytic cost model.

    Runs between span_reduction and work_distribution, so op.op_it_space_splits
    still holds only span_reduction's commits. Computes the split
    work_distribution would pick, hands it to the cost model, and commits the
    cost model's choice when it differs — returning True so the caller excludes
    the op from work_distribution (every op is divided by exactly one pass).
    """
    if not isinstance(op.data, Reduction):
        return False
    if op.data.reduction_type not in MATMUL_REDUCTION_OPS:
        return False
    if not config.ignore_work_division_hints and _has_work_div_hint(op):
        # User hints take ownership of the split decision; do not override them.
        return False

    rw = op_read_writes(op)
    args = get_mem_deps_from_rw(rw)
    input_tds, output_td = collect_tensor_deps(op, args)
    all_tds = input_tds + [output_td]

    it_space = iteration_space_from_op(op)

    symbol_meta = _collect_symbol_metadata(it_space)

    # Phase 1 covers Pointwise (and incidentally non-matmul Reduction) only;
    # symbolic batchmatmul needs symmetric changes inside the cost model
    # (_matmul_split_cost concretises M, N, K) and is tracked as a follow-up.
    # Raise loudly so users do not silently get a plan based on
    # the warmup optimization_hint.
    if symbol_meta:
        raise Unsupported(
            f"symbolic dim(s) {sorted(map(str, symbol_meta))} on batchmatmul "
            f"op {op.get_name()} are not supported yet; symbolic work "
            f"division currently covers pointwise (and non-matmul reduction) "
            f"ops only."
        )

    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(it_space, all_tds)

    # op.op_it_space_splits holds span_reduction's commits here: span_reduction
    # runs before this pass, and work_distribution — which would overwrite it —
    # runs after and skips the ops this pass claims.
    if hasattr(op, "op_it_space_splits"):
        write_index = next(iter(rw.writes)).index
        read_index = next((d.index for d in rw.reads), write_index)
        span_splits = apply_splits_from_index_coeff(
            op.op_it_space_splits, write_index, read_index, it_space
        )
        committed_splits = {s: v for s, v in span_splits.items() if v > 1}
    else:
        committed_splits = {}

    coord_vars = {v for e in output_td.device_coords[:-1] for v in e.free_symbols}
    reduction_vars = [v for v in it_space_adjusted if v not in coord_vars]
    blocked = coordinate_mask_blocked_vars(reduction_vars, stick_vars, it_space)
    default_splits, _, _ = _default_split(
        it_space_adjusted, output_td, committed_splits, max_cores, symbol_meta, blocked
    )
    splits = _cost_model_matmul_planner(
        op,
        default_splits,
        it_space_adjusted,
        output_td,
        stick_vars,
        committed_splits,
        max_cores,
        input_tds,
    )
    if splits == default_splits:
        return False

    apply_splits(op, splits, output_td)
    warn_if_per_core_overflow(all_tds, it_space, splits, op.get_name(), symbol_meta)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"cost_model_matmul_division work_division {op.get_name()}: "
            f"cores={math.prod(splits.values())}, "
            f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
            f"min_splits={committed_splits}, "
            f"op_it_space_splits={op.op_it_space_splits}"
        )
    return True


def cost_model_matmul_division(graph: GraphLowering) -> list[Operation]:
    """Pass 2: re-price matmul/bmm splits with the analytic hardware cost model.

    Runs after span_reduction and before work_distribution. Returns the ops it
    re-split so passes.py can exclude them from work_distribution — every op is
    divided by exactly one pass.
    """
    operations = graph.operations
    max_cores = _validate_max_cores()
    cost_model_ops: list[Operation] = []
    for op in _iter_computed_buffers(operations):
        if _cost_model_divide_op(op, max_cores):
            cost_model_ops.append(op)
    return cost_model_ops

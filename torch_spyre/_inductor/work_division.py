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
import enum
import math
import itertools
from sympy import Expr, Symbol, divisors
from .ir import SpyreConstantFallback, SpyreEmptyFallback

import torch
from torch._inductor.ir import (
    ComputedBuffer,
    ExternKernel,
    FallbackKernel,
    MultiOutput,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
)

from torch._inductor.dependencies import MemoryDep

from .errors import Unsupported
from .constants import BATCH_MATMUL_OP, TOPK_OPS
from .ir import FixedTiledLayout
from .pass_utils import (
    SchedNodeArg,
    concretize_expr,
    get_mem_deps_from_rw,
    device_coordinates,
    iteration_space_from_op,
    splits_by_index_coeff,
    apply_splits_from_index_coeff,
)
from typing import Callable, Iterator

from .logging_utils import get_inductor_logger
from . import config
import logging

logger = get_inductor_logger("work_division")

# Maximum memory access span per core: 256MB hardware limit
MAX_SPAN_BYTES = 256 * 1024 * 1024

aten = torch.ops.aten
spyreop = torch.ops.spyre


@dataclasses.dataclass
class TensorDep:
    """Bundles a MemoryDep with its FixedTiledLayout and pre-computes device coordinates."""

    dep: MemoryDep
    layout: FixedTiledLayout
    device_coords: list[Expr] = dataclasses.field(init=False)

    def __post_init__(self):
        self.device_coords = device_coordinates(self.layout.device_layout, self.dep)


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
) -> tuple[Symbol, int] | None:
    """Return (dim, split) for the dim in dims that maximises core_split(size, n_cores).

    Returns None if no dim yields a split > 1.
    """
    best_dim, best_split = None, 0
    for d in dims:
        s = core_split(concretize_expr(iteration_space[d]), n_cores)
        if s > best_split:
            best_dim, best_split = d, s
    return (best_dim, best_split) if best_split > 1 else None


def multi_dim_iteration_space_split(
    iteration_space: dict[Symbol, Expr],
    max_cores: int,
    output_dims: list[Symbol],
    reduction_dims: list[Symbol],
    min_splits: dict[Symbol, int] | None = None,
) -> dict[Symbol, int]:
    """Distribute max_cores across the iteration space.

    Three-pass algorithm:
      1. Satisfy min_splits (span-reduction commitments).
      2. Distribute remaining cores to output_dims in priority order.
      3. If this is a reduction op, pick the single most-splittable reduction dim
         for any remaining cores.

    The product of all splits will be <= max_cores.
    """
    is_reduction_included = bool(reduction_dims)

    splits = {v: 1 for v in iteration_space.keys()}
    n_cores_remaining = max_cores

    if min_splits:
        # Sanity check: making sure that reduction_dims list is cleared up if
        #               any reduction dim is already selected during span reduction
        assert (
            not is_reduction_included  # not empty
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
        # TODO(issue#1372): with symbolic work division, concretize_expr
        #                   for core_split will not be needed.
        best_split = core_split(concretize_expr(iteration_space[v]), n_cores_remaining)
        if best_split > 1:
            splits[v] = best_split
            n_cores_remaining = n_cores_remaining // best_split

    if is_reduction_included and n_cores_remaining > 1:
        result = _most_splittable_dim(
            reduction_dims, iteration_space, n_cores_remaining
        )
        if result is not None:
            best_dim, best_split = result
            splits[best_dim] = best_split

    return splits


def adjust_it_space_for_sticks(
    it_space: dict[Symbol, Expr],
    tensor_deps: list[TensorDep],
) -> tuple[dict[Symbol, Expr], dict[Symbol, int]]:
    """
    Return a copy of it_space with stick variables converted from elements to
    sticks, plus a dict mapping each stick variable to its max element per stick
    value.

    For each tensor, find the variable that indexes its stick dimension and
    convert its size in it_space from elements to sticks. This ensures work
    division treats sticks as atomic units.

    When tensors of different dtypes share a stick variable (e.g. a float16
    input and an int64 argmax output), the largest elems_per_stick is used
    so the adjustment is conservative (fewer sticks → smaller adjusted size →
    fewer cores assigned to the stick dimension).

    The original it_space is not mutated.
    """
    # Pass 1: find the largest elems_per_stick per stick variable.
    adjusted_space = dict(it_space)
    max_elems: dict[Symbol, int] = {}
    for td in tensor_deps:
        stick_expr = td.device_coords[-1]
        if len(stick_expr.free_symbols) != 1:
            continue
        stick_var = next(iter(stick_expr.free_symbols))
        if stick_var not in adjusted_space:
            continue
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
            # Concretize the iteration-space size so R (and therefore the
            # ``int(term.subs(...))`` cast below) is a Python int.  Per-core
            # span is a hardware-bound quantity that must be compared against
            # MAX_SPAN_BYTES, so concretization here is the right boundary.
            # TODO(issue#1372): Symbolic work division will keep this symbolic.
            R = concretize_expr(it_space_orig[v]) // splits.get(v, 1)
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
) -> None:
    """Log CRITICAL if any tensor's per-core memory span exceeds MAX_SPAN_BYTES."""
    for td in tensor_deps:
        per_core_span = get_per_core_span(td, splits, it_space_orig)
        if per_core_span > MAX_SPAN_BYTES:
            dl = td.layout.device_layout
            logger.critical(
                f"{op_name}: per-core tensor span "
                f"{per_core_span / (1024 * 1024):.2f} MB "
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

    Args:
        tensor_deps: List of tensor dependencies to check
        it_space_orig: Original iteration space (element-valued)
        it_space_adjusted: Adjusted iteration space (stick-valued for stick vars)
        stick_vars: Mapping of stick variables to elements per stick
        max_cores: Maximum number of cores available

    Returns a dict mapping Symbol -> number of slices.
    """
    accumulated_splits: dict[Symbol, int] = {}

    for td in tensor_deps:
        if get_per_core_span(td, accumulated_splits, it_space_orig) <= MAX_SPAN_BYTES:
            continue

        for coord in td.device_coords[:-1]:
            # Concretize for the ``> 1`` comparison: with symbolic ranges,
            # ``s0 > 1`` returns a sympy Relational whose truth value is
            # undefined.  Span filtering here is a structural decision that
            # needs a concrete answer.
            # TODO(issue#1372): Symbolic work division will keep this symbolic.
            vars = [
                v
                for v in coord.free_symbols
                if concretize_expr(it_space_orig.get(v, 1)) > 1
            ]
            if not vars:
                continue

            def valid_splits(v: Symbol) -> list[int]:
                current_min = accumulated_splits.get(v, 1)
                if v in stick_vars:
                    stick_count = concretize_expr(it_space_adjusted[v])
                    return [s for s in divisors(stick_count) if s >= current_min]
                return [
                    s
                    for s in divisors(concretize_expr(it_space_orig[v]))
                    if s >= current_min
                ]

            var_divisors = [valid_splits(v) for v in vars]

            for v, candidates in zip(vars, var_divisors):
                if not candidates:
                    raise Unsupported(
                        f"No valid split for variable {v} "
                        f"(orig_size={concretize_expr(it_space_orig[v])}, "
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
                for v, s in zip(vars, combo):
                    trial[v] = s

                if math.prod(trial.values()) > max_cores:
                    continue

                span = get_per_core_span(td, trial, it_space_orig)

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
            for v, s in zip(vars, best_combo):
                accumulated_splits[v] = s

            if best_span <= MAX_SPAN_BYTES:
                break

            # Still above the limit. If this coord still evaluates to > 1 under
            # the committed splits, inner dimensions cannot reduce the span further.
            # Concretize it_space_orig[v] so the ``int(coord.subs(...))`` cast
            # below succeeds with symbolic ranges.
            # TODO(issue#1372): Symbolic work division will keep this symbolic.
            per_core_coord_size = (
                max(
                    int(
                        coord.subs(
                            {
                                v: concretize_expr(it_space_orig[v])
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
) -> tuple[list[Symbol], list[Symbol]]:
    """Partition iteration variables into output dims and reduction dims.

    Output dims are those whose symbols appear in the output tensor's device
    coordinate expressions (excluding the stick coordinate). Reduction dims are
    the remainder. Both lists are sorted by decreasing concrete size.

    Variables already committed as min_splits should be filtered out of
    it_space_adjusted before calling this function.
    """
    coord_vars = {v for e in output.device_coords[:-1] for v in e.free_symbols}

    output_pairs: list[tuple[Symbol, Expr]] = []
    reduction_pairs: list[tuple[Symbol, Expr]] = []
    for s, e in it_space_adjusted.items():
        (output_pairs if s in coord_vars else reduction_pairs).append((s, e))

    # Concretize sort keys: comparing two sympy Symbols returns a Relational
    # whose truth value is undefined and would raise inside Python's sort.
    # The priority order is a structural decision (largest dim first) that
    # needs a concrete numeric ordering.
    # TODO(issue#1372): Symbolic work division will keep this symbolic.
    output_pairs.sort(key=lambda t: concretize_expr(t[1]), reverse=True)
    reduction_pairs.sort(key=lambda t: concretize_expr(t[1]), reverse=True)

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
    rw = op.get_read_writes()
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

    rw = op.get_read_writes()
    write_index = output_td.dep.index
    first_read = next(iter(rw.reads), None)
    read_index = first_read.index if first_read is not None else write_index
    op.op_it_space_splits = splits_by_index_coeff(splits, write_index, read_index)


def span_reduction_pass(
    op: ComputedBuffer,
    args: list[SchedNodeArg],
    max_cores: int,
) -> None:
    """Mandatory per-op pass: compute minimum splits to satisfy the 256MB span limit.

    Writes results to op.op_it_space_splits. If no span violation exists,
    op.op_it_space_splits is left unset (apply_splits is a no-op for splits <= 1).
    """
    it_space = iteration_space_from_op(op)
    input_tds, output_td = collect_tensor_deps(op, args)
    all_tds = input_tds + [output_td]

    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(it_space, all_tds)
    min_splits = must_split_vars(
        all_tds, it_space, it_space_adjusted, stick_vars, max_cores
    )

    coord_vars = {v for e in output_td.device_coords[:-1] for v in e.free_symbols}
    reduction_vars_to_split = set(min_splits) - coord_vars
    # Each entry in Reduction.reduction_ranges maps to at most one Symbol via
    # index_vars_squeeze (size-1 entries are squeezed away). So len > 1 means
    # genuinely distinct reduction dimensions, not multiple symbols from one dim.
    if len(reduction_vars_to_split) > 1:
        raise Unsupported(
            f"Cannot satisfy hardware memory span limit "
            f"({MAX_SPAN_BYTES // (1024 * 1024)}MB) without splitting "
            f"{len(reduction_vars_to_split)} reduction dimension(s) "
            f"({reduction_vars_to_split}), but the backend supports at most 1."
        )

    apply_splits(op, min_splits, output_td)

    if logger.isEnabledFor(logging.DEBUG) and math.prod(min_splits.values()) > 1:
        logger.debug(
            f"span_reduction work_division {op.get_name()}: cores={math.prod(min_splits.values())}, "
            f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
            f"priorities=[], min_splits={min_splits}, "
            f"op_it_space_splits={op.op_it_space_splits}"
        )


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

    it_space_adjusted, stick_vars = adjust_it_space_for_sticks(it_space, all_tds)

    # Recover splits committed by span_reduction_pass using the same
    # coeff-keyed encoding that codegen uses — stable across passes.
    if hasattr(op, "op_it_space_splits"):
        rw = op.get_read_writes()
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

    # TODO: The final dim committed by span_reduction_pass holds the minimum
    #       split that gets the span under the limit, so it may have headroom
    #       for additional parallelism (outer dims committed before it are
    #       already maximally split and have no headroom). Excluding it here
    #       leaves that parallelism on the table when other dims can't absorb
    #       the remaining cores.
    it_space_remaining = {
        s: e for s, e in it_space_adjusted.items() if s not in committed_splits
    }
    output_dims, reduction_dims = prioritize_dimensions(output_td, it_space_remaining)

    # If span_reduction_pass already committed a reduction split, suppress further
    # reduction splitting so the final result never exceeds one reduction dim split.
    coord_vars = {v for e in output_td.device_coords[:-1] for v in e.free_symbols}
    if any(v not in coord_vars for v in committed_splits):
        reduction_dims = []

    # Pass max_cores, not remaining_cores: multi_dim_iteration_space_split
    # accounts for committed_splits in its first pass, consuming those cores
    # itself before distributing the rest by priority.
    splits = multi_dim_iteration_space_split(
        it_space_adjusted,
        max_cores,
        output_dims,
        reduction_dims,
        committed_splits,
    )
    if config.cost_model_planner:
        splits = _cost_model_planner(
            op,
            splits,
            it_space_adjusted,
            output_td,
            stick_vars,
            committed_splits,
            max_cores,
            input_tds,
        )

    apply_splits(op, splits, output_td)

    if logger.isEnabledFor(logging.DEBUG) and math.prod(splits.values()) > 1:
        logger.debug(
            f"work_distribution work_division {op.get_name()}: cores={math.prod(splits.values())}, "
            f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
            f"priorities={output_dims + reduction_dims}, min_splits={committed_splits}, "
            f"op_it_space_splits={op.op_it_space_splits}"
        )

    warn_if_per_core_overflow(all_tds, it_space, splits, op.get_name())


_PT_ROWS = 8  # PT block rows per corelet

# Constants for the matmul cost model (_matmul_split_cost). Each is either an
# AIU hardware limit or a coefficient fit to measured device kernel times.
_TARGET_PT_PASSES = 8  # per-core M that keeps the PT pipeline full = this * _PT_ROWS
_M_MIN = _PT_ROWS // 2  # below half a PT pass an m-split buys nothing
_PEAK_MACS_US_CORE = (98.304e12 / 2 / 32) / 1e6  # DL16 peak / 32 cores, MACs/us/core
_HBM_BW_GBS = 204.8  # LPDDR5 aggregate peak bandwidth
_DTYPE_BYTES = 2  # fp16
_PSUM_PER_ELEM_US = 1.4e-4  # per output element, per K-split ring reduction hop
_COHORT_LIMIT = 8  # cores sharing a broadcast before it contends for bandwidth
_BATCH_SPLIT_EXPONENT = 1.4  # batch-split cost grows ~ b ** this (fit to bmm sweeps)
_TARGET_M_PENALTY_US = 50.0  # tie-break weight, per log2 step off the target m-split

# Constants for the pointwise cost model (_pointwise_split_cost).
_COST_PEAK_ELEMENTS_US_CORE = 1.76e3  # per-core SFP elementwise rate (silu sweep)
_COST_STICK_FRAG_US_PER_BYTE = 4.5e-7  # per-byte tax for splitting the stick dim

# Constants for the reduction cost model (_reduction_split_cost). Calibrated
# from forced-split sum reductions on [1,32,512,4096] and [1,512,4096] across
# cores in {1,2,4}: compute_us = elems_in / cores / K fits K = 1.2e4 (R^2 0.97).
_COST_REDUCE_ELEM_PER_US_CORE = 1.2e4  # per-core simple-reduction throughput

# Reduction types the reduction planner models. Excludes matmul (its own
# planner), topk (single-core path), and ops unimplemented in spyre_kernel.
_SIMPLE_REDUCE_TYPES = {"exx2", "mean", "sum", "max", "amax", "min", "amin"}


def _matmul_split_cost(
    B: int,
    M: int,
    K: int,
    N: int,
    b: int,
    m: int,
    n: int,
    k: int,
    max_cores: int,
) -> float:
    """Estimated kernel time in microseconds for ``[B,M,K]@[B,K,N]`` run with
    the given (b, m, n, k) core split. Lower is better; infeasible -> inf.
    """
    cores_used = b * m * n * k
    if cores_used == 0 or cores_used > max_cores:
        return float("inf")

    # Compute: per-core MACs over peak, derated when the per-core M tile is too
    # short to fill the PT pipeline. The PT array streams M in passes of
    # _PT_ROWS; below _TARGET_PT_PASSES passes its startup/drain overhead is
    # amortised over too little work, and that overhead grows sub-linearly,
    # hence the sqrt.
    m_t = M // m if m else 1
    pt_passes = max(1.0, m_t / _PT_ROWS)
    pt_eff = min(1.0, (pt_passes / _TARGET_PT_PASSES) ** 0.5)
    compute_us = (B * M * N * K / cores_used) / (_PEAK_MACS_US_CORE * pt_eff)

    # HBM: every input operand is broadcast to the cohort of cores splitting the
    # orthogonal dim. Past _COHORT_LIMIT the broadcasts contend for the shared
    # link, so effective bandwidth falls off linearly with cohort size.
    bytes_total = (B * M * K + B * K * N + B * M * N) * _DTYPE_BYTES
    cohort_penalty = max(1.0, max(m, n) / _COHORT_LIMIT)
    hbm_us = bytes_total / (_HBM_BW_GBS * 1000) * cohort_penalty

    # PSUM: a K-split spreads the reduction over k cores, costing (k-1) partial-
    # sum hops around the ring, each touching every output element.
    psum_us = max(0, k - 1) * (B * M * N) * _PSUM_PER_ELEM_US

    # Tie-break: among compute-equivalent splits prefer the m-split that lands
    # per-core M near the PT sweet spot, penalising log2-distance from it.
    target_m = max(
        _M_MIN, min(max_cores // 2, max(1, M // (_TARGET_PT_PASSES * _PT_ROWS)))
    )
    target_m_us = abs(math.log2(max(1, m) / target_m)) * _TARGET_M_PENALTY_US

    # Splitting batch across cores is strictly worse than tiling it in time on
    # one core (each item is independent), so charge a super-linear b penalty.
    batch_penalty = b**_BATCH_SPLIT_EXPONENT

    return (compute_us + hbm_us + psum_us + target_m_us) * batch_penalty


def _pointwise_split_cost(
    input_sizes: list[int],
    input_fanouts: list[int],
    out_size: int,
    cores_used: int,
    stick_split: int,
    batch_split: int,
) -> float:
    """Estimated kernel time in microseconds for a pointwise op under a given
    per-dim core split. Lower is better; infeasible -> inf.

    A pointwise kernel is HBM-bound once the per-core slice is large and
    compute-bound once it shrinks, so compute and HBM are combined as a
    roofline (max), not summed.
    """
    if cores_used <= 0:
        return float("inf")

    # HBM: each input is charged numel * fanout bytes. fanout is the product
    # of splits over dims that input lacks: == 1 for a partitioned input (its
    # per-core slices tile the whole tensor), > 1 for a broadcast input, where
    # the (fanout - 1) extra reads are the cohort tax. The output is always
    # partitioned, so it is charged once.
    bytes_total = (
        sum(s * f for s, f in zip(input_sizes, input_fanouts)) + out_size
    ) * _DTYPE_BYTES
    hbm_us = bytes_total / (_HBM_BW_GBS * 1000)

    # Compute: per-core element work over the SFP elementwise rate. Bounds the
    # cost from below so tiny per-core slices don't appear free.
    compute_us = (out_size / cores_used) / _COST_PEAK_ELEMENTS_US_CORE

    # Stick fragmentation: splitting the innermost (stick) dim leaves each core
    # with partial sticks, costing per-byte HBM bank conflicts. Without this
    # the roofline is flat for non-broadcast inputs and the planner would
    # arbitrarily pick a stick-dim split that regresses on hardware.
    if stick_split > 1:
        stick_us = (
            _COST_STICK_FRAG_US_PER_BYTE * (stick_split - 1) * out_size * _DTYPE_BYTES
        )
    else:
        stick_us = 0.0

    # Splitting a batch-like outer dim across cores is strictly worse than
    # tiling it in time on one core (each unit is independent), so charge the
    # same super-linear b penalty as the matmul planner.
    batch_penalty = batch_split**_BATCH_SPLIT_EXPONENT

    return (max(compute_us, hbm_us) + stick_us) * batch_penalty


def _reduction_split_cost(
    elems_in: int,
    elems_out: int,
    d_split: int,
    r_split: int,
    max_cores: int,
) -> float:
    """Estimated kernel time in microseconds for a simple reduction run with
    the given output-dim split d_split and reduction-dim split r_split. Lower
    is better; infeasible -> inf.
    """
    cores_used = d_split * r_split
    if cores_used == 0 or cores_used > max_cores:
        return float("inf")

    # Compute: per-core input elements over the simple-reduction throughput.
    compute_us = (elems_in / cores_used) / _COST_REDUCE_ELEM_PER_US_CORE

    # HBM: input plus output bytes over bandwidth. A pure reduction has no
    # broadcast — each core reads its 1/cores slice and writes its output
    # slice — so there is no cohort term (unlike the matmul cost model, where
    # the cohort captures weight-broadcast contention).
    hbm_us = (elems_in + elems_out) * _DTYPE_BYTES / (_HBM_BW_GBS * 1000)

    # PSUM: an r-split spreads the reduction over r cores, costing (r - 1)
    # partial-sum ring hops that each touch every output element. Added on top
    # of the roofline because the ring reduce is sequenced after local compute.
    psum_us = max(0, r_split - 1) * elems_out * _PSUM_PER_ELEM_US

    return max(compute_us, hbm_us) + psum_us


# --- Op classification and per-class dim identification -------------------

OpClass = enum.Enum("OpClass", ["MATMUL", "POINTWISE", "REDUCTION"])


def _classify_op(op: ComputedBuffer) -> "OpClass | None":
    """Return the cost-model OpClass for ``op``, or None to leave it to the
    default distributor. A Reduction is a matmul (reduction_type batchmatmul)
    or one of the simple reductions; everything else is pointwise.
    """
    if isinstance(op.data, Reduction):
        if op.data.reduction_type == BATCH_MATMUL_OP:
            return OpClass.MATMUL
        if op.data.reduction_type in _SIMPLE_REDUCE_TYPES:
            return OpClass.REDUCTION
        return None
    if isinstance(op.data, Pointwise):
        return OpClass.POINTWISE
    return None


@dataclasses.dataclass
class _MatmulDims:
    m_dim: Symbol
    n_dim: Symbol
    k_dim: Symbol
    batch_dims: list[Symbol]
    M_e: int
    N_e: int
    K_e: int
    B_total: int


def _matmul_dims(
    output_td: TensorDep,
    stick_vars: dict[Symbol, int],
    committed_splits: dict[Symbol, int],
    input_tds: list[TensorDep],
    it_space_adjusted: dict[Symbol, Expr],
) -> _MatmulDims | None:
    """Identify the M, N, K, batch dims of a matmul / bmm with concrete sizes.

    Returns None when the dim structure does not fit the (one M, one N, one K,
    zero-or-more batch) shape the cost model expects, or a span-committed split
    is already in place.
    """
    if committed_splits:
        return None

    # The stickified output coord dim is N; the rest index rows. Of those row
    # dims, M is the one appearing in a single input (the LHS); batch dims
    # appear in both.
    output_coord_vars = {
        v for e in output_td.device_coords[:-1] for v in e.free_symbols
    }
    n_dims = [d for d in output_coord_vars if d in stick_vars]
    row_dims = [d for d in output_coord_vars if d not in stick_vars]
    if len(n_dims) != 1 or not row_dims:
        return None
    n_dim = n_dims[0]

    def _appears_in_one_input(dim: Symbol) -> bool:
        hits = sum(
            dim in {v for e in td.device_coords for v in e.free_symbols}
            for td in input_tds
        )
        return hits == 1

    m_candidates = [d for d in row_dims if _appears_in_one_input(d)]
    if len(m_candidates) != 1:
        return None
    m_dim = m_candidates[0]
    batch_dims = [d for d in row_dims if d is not m_dim]

    # K is the lone reduction dim (multi-K matmul not modelled here).
    reduction = [d for d in it_space_adjusted if d not in output_coord_vars]
    if len(reduction) != 1:
        return None
    k_dim = reduction[0]

    # The iteration space measures N and K in sticks; the cost model wants real
    # elements so its byte and MAC counts are physical.
    elems_per_stick = output_td.layout.device_layout.device_dtype.elems_per_stick()
    M_e = concretize_expr(it_space_adjusted[m_dim])
    N_e = concretize_expr(it_space_adjusted[n_dim]) * elems_per_stick
    K_e = concretize_expr(it_space_adjusted[k_dim]) * elems_per_stick
    B_total = math.prod(concretize_expr(it_space_adjusted[bd]) for bd in batch_dims)

    return _MatmulDims(m_dim, n_dim, k_dim, batch_dims, M_e, N_e, K_e, B_total)


@dataclasses.dataclass
class _PointwiseDims:
    dims: list[Symbol]
    input_dim_sets: list[set[Symbol]]
    input_sizes: list[int]
    out_size: int
    stick_idx: int | None
    batch_idx: list[int]


def _pointwise_dims(
    output_td: TensorDep,
    stick_vars: dict[Symbol, int],
    committed_splits: dict[Symbol, int],
    input_tds: list[TensorDep],
    it_space_adjusted: dict[Symbol, Expr],
) -> _PointwiseDims | None:
    """Identify the per-input fanout structure of a pointwise op.

    A pointwise op has no reduction dims, so the iteration space is exactly the
    output coord vars. Returns None when there is nothing to split or a
    span-committed split is already in place.
    """
    if committed_splits:
        return None

    dims = list(it_space_adjusted.keys())
    if not dims:
        return None
    dim_sizes = [concretize_expr(it_space_adjusted[d]) for d in dims]

    # Per input, the iteration dims it carries (from its device coords). A dim
    # the input lacks is broadcast for it; splitting that dim multiplies its
    # bytes by the split (the fanout). The iteration space measures the stick
    # dim in sticks, so byte counts here are in sticks too — consistent across
    # inputs and output, which is all the roofline needs.
    input_dim_sets = [
        {v for e in td.device_coords for v in e.free_symbols} for td in input_tds
    ]
    input_sizes = [
        math.prod(sz for d, sz in zip(dims, dim_sizes) if d in dim_set)
        for dim_set in input_dim_sets
    ]
    out_size = math.prod(dim_sizes)

    # The output's stick dim is the unique iteration dim in stick_vars. Among
    # the rest, the largest is the main work axis (like matmul's M); splitting
    # it parallelises the bulk cleanly. Any other non-stick dim is "batch-like"
    # and pays the b**1.4 split penalty.
    stick_idx = next((i for i, d in enumerate(dims) if d in stick_vars), None)
    non_stick = [i for i in range(len(dims)) if i != stick_idx]
    if non_stick:
        main_idx = max(non_stick, key=lambda i: dim_sizes[i])
        batch_idx = [i for i in non_stick if i != main_idx]
    else:
        batch_idx = []

    return _PointwiseDims(
        dims, input_dim_sets, input_sizes, out_size, stick_idx, batch_idx
    )


@dataclasses.dataclass
class _ReductionDims:
    output_dims: list[Symbol]
    reduction_dims: list[Symbol]
    elems_in: int
    elems_out: int


def _reduction_dims(
    output_td: TensorDep,
    stick_vars: dict[Symbol, int],
    committed_splits: dict[Symbol, int],
    it_space_adjusted: dict[Symbol, Expr],
) -> _ReductionDims | None:
    """Identify the (output, reduction) dim partition of a simple reduction.

    Returns None when the iteration space is empty or a span-committed split is
    already in place.
    """
    if committed_splits:
        return None

    output_dims, reduction_dims = prioritize_dimensions(output_td, it_space_adjusted)
    if not output_dims and not reduction_dims:
        return None

    # The iteration space measures the stick dim in sticks; convert back to
    # elements so the cost model's byte and element counts are physical.
    elems_per_stick = output_td.layout.device_layout.device_dtype.elems_per_stick()

    def _elems(dim: Symbol) -> int:
        s = concretize_expr(it_space_adjusted[dim])
        return s * elems_per_stick if dim in stick_vars else s

    elems_out = math.prod(_elems(d) for d in output_dims) if output_dims else 1
    elems_red = math.prod(_elems(d) for d in reduction_dims) if reduction_dims else 1
    elems_in = elems_out * elems_red

    return _ReductionDims(output_dims, reduction_dims, elems_in, elems_out)


def _enumerate_splits(
    divisor_lists: list[list[int]],
    max_cores: int,
) -> Iterator[tuple[int, ...]]:
    """Yield feasible per-dim core splits: one divisor per dim from the matching
    list, with product <= max_cores.
    """
    for combo in itertools.product(*divisor_lists):
        if math.prod(combo) <= max_cores:
            yield combo


def _cost_model_planner(
    op: ComputedBuffer,
    splits: dict[Symbol, int],
    it_space_adjusted: dict[Symbol, Expr],
    output_td: TensorDep,
    stick_vars: dict[Symbol, int],
    committed_splits: dict[Symbol, int],
    max_cores: int,
    input_tds: list[TensorDep],
) -> dict[Symbol, int]:
    """Pick the lowest-cost feasible core split for ``op`` per its op class.

    Classifies the op, identifies the dims its cost function needs, then runs
    the same enumerate -> score -> argmin pipeline over per-dim divisor lists.
    Returns ``splits`` unchanged for an unclassified op, an op with a
    span-committed split, or a pick that would use fewer cores than the default.
    """
    op_class = _classify_op(op)
    if op_class is None:
        return splits

    # Per class: the ordered dims to split, a parallel list of per-dim divisor
    # lists, and a scorer mapping a candidate split (parallel to dims) to its
    # estimated kernel us. The pipeline below is shared across all three.
    if op_class is OpClass.MATMUL:
        mm = _matmul_dims(
            output_td, stick_vars, committed_splits, input_tds, it_space_adjusted
        )
        if mm is None:
            return splits
        dims = mm.batch_dims + [mm.m_dim, mm.n_dim, mm.k_dim]
        divisor_lists = [
            [int(x) for x in divisors(concretize_expr(it_space_adjusted[dim]))]
            for dim in dims
        ]
        n_batch = len(mm.batch_dims)

        def score(combo: tuple[int, ...]) -> float:
            b = math.prod(combo[:n_batch])
            m, n, k = combo[n_batch:]
            return _matmul_split_cost(
                mm.B_total, mm.M_e, mm.K_e, mm.N_e, b, m, n, k, max_cores
            )

    elif op_class is OpClass.POINTWISE:
        pw = _pointwise_dims(
            output_td, stick_vars, committed_splits, input_tds, it_space_adjusted
        )
        if pw is None:
            return splits
        dims = pw.dims
        divisor_lists = [
            [int(x) for x in divisors(concretize_expr(it_space_adjusted[dim]))]
            for dim in dims
        ]

        def score(combo: tuple[int, ...]) -> float:
            fanouts = [
                math.prod(s for dim, s in zip(dims, combo) if dim not in dim_set)
                for dim_set in pw.input_dim_sets
            ]
            stick_split = combo[pw.stick_idx] if pw.stick_idx is not None else 1
            batch_split = math.prod(combo[i] for i in pw.batch_idx)
            return _pointwise_split_cost(
                pw.input_sizes,
                fanouts,
                pw.out_size,
                math.prod(combo),
                stick_split,
                batch_split,
            )

    else:  # OpClass.REDUCTION
        rd = _reduction_dims(output_td, stick_vars, committed_splits, it_space_adjusted)
        if rd is None:
            return splits
        dims = rd.output_dims + rd.reduction_dims
        divisor_lists = [
            [int(x) for x in divisors(concretize_expr(it_space_adjusted[dim]))]
            for dim in dims
        ]
        n_out = len(rd.output_dims)

        def score(combo: tuple[int, ...]) -> float:
            d_split = math.prod(combo[:n_out])
            r_split = math.prod(combo[n_out:])
            return _reduction_split_cost(
                rd.elems_in, rd.elems_out, d_split, r_split, max_cores
            )

    best = None
    best_cost = float("inf")
    for combo in _enumerate_splits(divisor_lists, max_cores):
        c = score(combo)
        if c < best_cost:
            best_cost = c
            best = combo
    if best is None:
        return splits

    new_splits = dict(splits)
    for dim, s in zip(dims, best):
        new_splits[dim] = int(s)

    # Never trade down to fewer cores than the default distributor found.
    if math.prod(new_splits.values()) < math.prod(splits.values()):
        return splits

    logger.debug(
        f"cost_model work_division {op.get_name()}: {op_class.name} "
        f"split={dict(zip(dims, best))} cost={best_cost:.1f}us"
    )
    return new_splits


def _try_k_fast_split(
    it_space: dict[Symbol, Expr],
    output_td: TensorDep,
    min_splits: dict[Symbol, int] | None,
    max_cores: int,
) -> dict[Symbol, int] | None:
    """Propose (1, n_split, k_split>1) for narrow-N small-M matmul shapes.

    Caller (k_fast_division pass) gates on matmul + the feature flag.
    Range thresholds derived from empirical hardware measurements.
    """
    dims = list(it_space.keys())
    output_coord_vars = {v for e in output_td.device_coords for v in e.free_symbols}
    reduction_dims = [d for d in dims if d not in output_coord_vars]
    # k_fast emits an (m, n, k) split — only matmul's single K dim qualifies.
    if len(reduction_dims) != 1:
        return None
    k_dim = reduction_dims[0]

    output_dims = [d for d in dims if d in output_coord_vars]
    # TODO: 2D matmul only. bmm has a B dim the planner already splits;
    # folding it into m_dims would waste that lever. Needs a bmm-aware policy.
    if len(output_dims) != 2:
        return None
    # Pick the larger of the two output dims to split across cores; "N" is
    # convention (for the target shape M < N, max picks the conventional N).
    n_dim = max(output_dims, key=lambda d: concretize_expr(it_space[d]))
    m_dims = [d for d in output_dims if d != n_dim]

    # k_fast's (1, n, k>1) shape can't sit on top of a split span_reduction
    # already committed on K or an M dim — skip those.
    # TODO: A span_reduction commit on K or an M dim is the minimum split that
    #       gets the span under the limit, not necessarily the final one, so
    #       cores may still be free after it. Returning None here hands the
    #       whole op to the default planner instead of applying k_fast within
    #       the cores span_reduction leaves available, leaving the k_fast
    #       speedup on the table for those shapes.
    if min_splits and (k_dim in min_splits or any(d in min_splits for d in m_dims)):
        return None

    M = math.prod(concretize_expr(it_space[d]) for d in m_dims) if m_dims else 1
    N = concretize_expr(it_space[n_dim])
    K = concretize_expr(it_space[k_dim])

    elems_per_stick = output_td.layout.device_layout.device_dtype.elems_per_stick()
    # iteration_space carries unpadded element counts; skip ragged N/K
    # (e.g. N=99) that the k_fast splits can't divide cleanly.
    if N % elems_per_stick != 0 or K % elems_per_stick != 0:
        return None
    n_sticks = N // elems_per_stick
    k_sticks = K // elems_per_stick

    # The gates below pick shapes where pure-M underfeeds the PT array but
    # a (1, n, k) split keeps it busy.
    rows_per_core = M / max_cores
    # Skip M too small to give one row per core, and M large enough that
    # pure-M already saturates PT — k_fast wins nothing either way.
    if rows_per_core < 1 or rows_per_core > 2 * _PT_ROWS:
        return None
    # Moderate M with wide N already uses cores well — only apply k_fast
    # when N is narrow enough that PT is starved.
    if rows_per_core > _PT_ROWS / 2 and n_sticks >= max_cores:
        return None
    # Need enough K to give every core at least one stick after splitting.
    if k_sticks < max_cores:
        return None

    # n_split must divide max_cores so k_split = max_cores // n_split is an
    # integer. Any divisor works — no power-of-2 restriction.
    candidates = sorted(
        (int(n) for n in divisors(max_cores) if 1 < n < max_cores), reverse=True
    )
    for n_split in candidates:
        if n_sticks % n_split != 0:
            continue
        k_split = max_cores // n_split
        if k_sticks < k_split or k_sticks % k_split != 0:
            continue
        result: dict[Symbol, int] = {k_dim: k_split, n_dim: n_split}
        for d in m_dims:
            result[d] = 1
        return result

    return None


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
        return

    pass_fn(op, args, max_cores)


def _validate_max_cores() -> int:
    max_cores = config.sencores
    if max_cores > 32 or max_cores < 1:
        raise Unsupported(f"invalid SENCORES value {max_cores}")
    return max_cores


def _iter_computed_buffers(operations: list[Operation]):
    """Yield ComputedBuffer ops, handling FallbackKernel/ExternKernel dispatch."""
    it = iter(operations)
    for op in it:
        if op.is_no_op():
            pass
        elif isinstance(op, ComputedBuffer):
            yield op
        elif isinstance(op, FallbackKernel):
            op = next(it, None)
            if not isinstance(op, MultiOutput):
                raise RuntimeError("FallbackKernel must be followed by MultiOutput")
            # Work division not supported on fallback kernels
        elif isinstance(op, ExternKernel):
            if isinstance(op, (SpyreConstantFallback, SpyreEmptyFallback)):
                # Work division not supported on allocation/constant kernels
                pass
            else:
                logger.warning(f"unhandled node type {type(op)}")
        else:
            logger.warning(f"unhandled operation type {type(op)}")


def span_reduction(operations: list[Operation]) -> None:
    """Pass 1: compute minimum per-op splits required by the 256MB span limit."""
    max_cores = _validate_max_cores()
    for op in _iter_computed_buffers(operations):
        rw = op.get_read_writes()
        args = get_mem_deps_from_rw(rw)
        if isinstance(op.data, Pointwise):
            divide_pointwise_op(op, args, max_cores, span_reduction_pass)
        elif isinstance(op.data, Reduction):
            divide_reduction_op(op, args, max_cores, span_reduction_pass)


def work_distribution(
    operations: list[Operation], k_fast_ops: list[Operation] | None = None
) -> None:
    """Pass 3: distribute remaining cores across ops to maximize parallelism.

    Ops in `k_fast_ops` were already divided by k_fast_division; they are
    left untouched so every op is divided by exactly one of the two passes.
    """
    k_fast_ops = k_fast_ops or []
    max_cores = _validate_max_cores()
    for op in _iter_computed_buffers(operations):
        if op in k_fast_ops:
            continue
        rw = op.get_read_writes()
        args = get_mem_deps_from_rw(rw)
        if isinstance(op.data, Pointwise):
            divide_pointwise_op(op, args, max_cores, work_distribution_pass)
        elif isinstance(op.data, Reduction):
            divide_reduction_op(op, args, max_cores, work_distribution_pass)


def _k_fast_divide_op(op: ComputedBuffer, max_cores: int) -> bool:
    """Divide one matmul op with k_fast when the heuristic fires.

    Runs between span_reduction and work_distribution. Reads span_reduction's
    commits straight from op.op_it_space_splits — work_distribution has not run
    yet, so it still holds the span-only subset. Returns True when k_fast
    commits a split, so the caller can exclude the op from work_distribution.
    """
    if not isinstance(op.data, Reduction):
        return False
    if op.data.reduction_type != BATCH_MATMUL_OP:
        return False

    rw = op.get_read_writes()
    args = get_mem_deps_from_rw(rw)
    input_tds, output_td = collect_tensor_deps(op, args)
    all_tds = input_tds + [output_td]

    it_space = iteration_space_from_op(op)
    it_space_adjusted, _ = adjust_it_space_for_sticks(it_space, all_tds)

    # op.op_it_space_splits holds span_reduction's commits here: span_reduction
    # runs before this pass, and work_distribution — which would overwrite it —
    # runs after and skips the ops k_fast claims.
    if hasattr(op, "op_it_space_splits"):
        write_index = next(iter(rw.writes)).index
        read_index = next((d.index for d in rw.reads), write_index)
        span_splits = apply_splits_from_index_coeff(
            op.op_it_space_splits, write_index, read_index, it_space
        )
        span_committed = {s: v for s, v in span_splits.items() if v > 1}
    else:
        span_committed = {}

    forced = _try_k_fast_split(it_space, output_td, span_committed, max_cores)
    if forced is None:
        return False

    apply_splits(op, forced, output_td)
    warn_if_per_core_overflow(all_tds, it_space, forced, op.get_name())

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"k_fast_division work_division {op.get_name()}: "
            f"cores={math.prod(forced.values())}, "
            f"iteration_space={it_space}, it_space_adjusted={it_space_adjusted}, "
            f"priorities=[], min_splits={span_committed}, "
            f"op_it_space_splits={op.op_it_space_splits}"
        )
    return True


def k_fast_division(operations: list[Operation]) -> list[Operation]:
    """Pass 2 (optional): divide narrow-N small-M matmuls with k_fast.

    Runs after span_reduction and before work_distribution. The
    core_id_k_fast_emission feature-flag gate lives in passes.py; this pass
    is only called when it is set. Returns the ops it committed a split for
    so passes.py can exclude them from work_distribution — every op is
    divided by exactly one of the two passes.
    """
    max_cores = _validate_max_cores()
    k_fast_ops: list[Operation] = []
    for op in _iter_computed_buffers(operations):
        if _k_fast_divide_op(op, max_cores):
            k_fast_ops.append(op)
    return k_fast_ops

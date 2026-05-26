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
from typing import Callable

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


# Hardware constants
_PT_ROWS = 8  # PT block rows per corelet

# Constants for the matmul cost-model planner. Each is either a hardware
# constant (PT_ROWS, peak MACs/us, HBM bandwidth) or an empirical coefficient
# fit from device-measured kernel times against split choices (see
# tests/cost_model_offline.py for the offline validator + the fit).
#
# The cost-model planner replaces the prior _maybe_mn_cosplit heuristic for
# matmul/bmm ops: it enumerates feasible (b, m, n, k) splits and picks the
# argmin of _matmul_split_cost. See _cost_model_matmul_planner below.
_MN_SPLIT_M_MIN = _PT_ROWS // 2                 # one half PT pass = smallest useful m-split
_MN_SPLIT_LOWLX_TARGET_N_STICKS = 8             # low-LX:  per-core N-band target (sticks)
_MN_SPLIT_HIGHLX_TARGET_PT_PASSES = 8           # high-LX: per-core M = target * _PT_ROWS rows
_MN_SPLIT_HIGHLX_LX_FRAC_THRESHOLD = 0.5        # LX_FRAC regime gate

# Cost-model coefficients (empirical fit, see cost_model_offline.py)
_COST_PEAK_MACS_US_CORE = (300e12 / 2 / 32) / 1e6   # AIU 1.0: ~4.7M MACs/us/core
_COST_HBM_BW_GBS = 166.0                            # AIU 1.0 aggregate HBM bandwidth
_COST_DTYPE_BYTES = 2                               # fp16
_COST_PSUM_PER_ELEM_US = 4e-4                       # fit: MoE k=2 added ~3.4ms / 8.4M elems / 1 hop
_COST_COHORT_LIMIT = 8                              # cohort sizes > this incur HBM broadcast contention
_COST_BATCH_SPLIT_PENALTY = 0.6                     # mult. penalty per b-step (large-K bmm fit)
_COST_TARGET_M_PENALTY_US = 50.0                    # per log2 step from target_m_split


def _matmul_names_unsafe_for_2d(operations: list[Operation]) -> set[str]:
    """Matmul output buf names that share a fusion group with a non-matmul op.

    A 2D m×n split on a matmul whose output is read by a non-matmul (or whose
    input is written by a non-matmul) forces cross-core data redistribution
    inside the resulting fused bundle (the non-matmul stays on default
    (mb, out) split). Measured cost on mlp [512,4096]: kernel +14% despite
    each standalone matmul winning 1.26-1.49x.
    """
    matmul_names: set[str] = set()
    matmul_input_names: dict[str, set[str]] = {}
    nm_outputs: set[str] = set()
    nm_inputs: set[str] = set()
    for op in _iter_computed_buffers(operations):
        rw = op.get_read_writes()
        is_matmul = (
            isinstance(op.data, Reduction)
            and op.data.reduction_type == BATCH_MATMUL_OP
        )
        if is_matmul:
            matmul_names.add(op.get_name())
            matmul_input_names[op.get_name()] = {d.name for d in rw.reads}
        else:
            nm_outputs.add(op.get_name())
            for d in rw.reads:
                nm_inputs.add(d.name)
    unsafe: set[str] = set()
    for mname in matmul_names:
        if mname in nm_inputs:
            unsafe.add(mname)
            continue
        if any(inp in nm_outputs for inp in matmul_input_names[mname]):
            unsafe.add(mname)
    return unsafe


def _matmul_split_cost(
    B: int, M: int, K: int, N: int,
    b: int, m: int, n: int, k: int,
    max_cores: int,
) -> float:
    """Estimate kernel cost (us) for a matmul ``[B,M,K]@[B,K,N]`` under the
    given (b, m, n, k) split. Lower is better. Fit + validated against
    device-measured kernel times for the 7 perf-suite shapes; see
    tests/cost_model_offline.py.

    Cost terms (additive, multiplied by batch-split penalty):
      compute_us : per-core MACs / peak; derated by PT-pipeline efficiency
                   when per-core M can't sustain TARGET_PT_PASSES passes.
      hbm_us     : total unique LHS+RHS+OUT bytes / HBM_BW; multiplied by a
                   cohort penalty when max(m, n) exceeds the broadcast limit.
      psum_us    : (k-1) ring hops over the output element count.
      target_m_us: log2-distance penalty around the shape-aware sweet spot.
    """
    cores_used = b * m * n * k
    if cores_used == 0 or cores_used > max_cores:
        return float("inf")

    # Compute (PT-pipeline efficiency)
    m_t = M // m if m else 1
    pt_passes = max(1.0, m_t / _PT_ROWS)
    pt_eff = min(1.0, pt_passes / _MN_SPLIT_HIGHLX_TARGET_PT_PASSES)
    effective_peak = _COST_PEAK_MACS_US_CORE * pt_eff
    compute_us = (B * M * N * K / cores_used) / effective_peak

    # HBM with cohort-broadcast penalty
    bytes_total = (B * M * K + B * K * N + B * M * N) * _COST_DTYPE_BYTES
    cohort = max(m, n)
    cohort_penalty = max(1.0, cohort / _COST_COHORT_LIMIT)
    hbm_us = bytes_total / (_COST_HBM_BW_GBS * 1000) * cohort_penalty

    # PSUM (scales with output elements per ring hop)
    psum_us = max(0, k - 1) * (B * M * N) * _COST_PSUM_PER_ELEM_US

    # Target-m_split tie-break (heuristic-aware): empirical fit shows m_split
    # close to ``clamp(M_MIN, cores//2, M / (TARGET_PT_PASSES * PT_ROWS))``
    # is the PT-pipeline sweet spot. Penalize log2-distance from that target.
    target_m = max(
        _MN_SPLIT_M_MIN,
        min(max_cores // 2, max(1, M // (_MN_SPLIT_HIGHLX_TARGET_PT_PASSES * _PT_ROWS))),
    )
    m_dist = abs(math.log2(max(1, m) / target_m))
    target_m_us = m_dist * _COST_TARGET_M_PENALTY_US

    # Batch-split penalty (multiplicative, fits bmm regression data)
    batch_penalty = 1.0 + _COST_BATCH_SPLIT_PENALTY * max(0, b - 1)

    return (compute_us + hbm_us + psum_us + target_m_us) * batch_penalty


def _cost_model_matmul_planner(
    op: ComputedBuffer,
    splits: dict[Symbol, int],
    it_space_adjusted: dict[Symbol, Expr],
    output_td: TensorDep,
    stick_vars: dict[Symbol, int],
    committed_splits: dict[Symbol, int],
    max_cores: int,
    unsafe_for_2d: set[str],
    input_tds: list[TensorDep],
) -> dict[Symbol, int]:
    """Pick the minimum-cost feasible (b, m, n, k) split for a matmul / bmm.

    Identifies M/N/K/batch dims via **input deps** (M appears in one input,
    batch in both, K is the reduction), enumerates feasible splits
    (divisors of each dim, product <= max_cores), scores each by
    _matmul_split_cost, and returns the argmin.

    Validated against device-measured kernel times for QO/KV/MLP/MoE
    gate-up/MoE down/bmm large-K/bmm small-K -- in every case the
    heuristic's empirically-best split is the cost-model argmin.

    Replaces _maybe_mn_cosplit as the matmul work-division policy.
    """
    if op.get_name() in unsafe_for_2d:
        return splits
    if not isinstance(op.data, Reduction):
        return splits
    if op.data.reduction_type != BATCH_MATMUL_OP:
        return splits
    if committed_splits:
        return splits

    output_coord_vars = {
        v for e in output_td.device_coords[:-1] for v in e.free_symbols
    }
    n_dims = [d for d in output_coord_vars if d in stick_vars]
    m_dims = [d for d in output_coord_vars if d not in stick_vars]
    if len(n_dims) != 1 or not m_dims:
        return splits
    n_dim = n_dims[0]

    # M = output dim that appears in exactly one input (the LHS).
    # Batch dims (if any) appear in both inputs.
    def _n_inputs(dim: Symbol) -> int:
        return sum(
            1
            for td in input_tds
            if dim in {v for e in td.device_coords for v in e.free_symbols}
        )

    m_candidates = [d for d in m_dims if _n_inputs(d) == 1]
    if len(m_candidates) != 1:
        return splits  # can't identify M unambiguously
    m_dim = m_candidates[0]
    batch_dims = [d for d in m_dims if d is not m_dim]

    # K: the single reduction dim. (Multi-K matmul not handled here.)
    reduction = [d for d in it_space_adjusted if d not in output_coord_vars]
    if len(reduction) != 1:
        return splits
    k_dim = reduction[0]

    # Sizes. M is already in elements; N and K are in sticks (the planner's
    # iteration_space view), so convert back for the cost calculation.
    M_e = concretize_expr(it_space_adjusted[m_dim])
    n_sticks = concretize_expr(it_space_adjusted[n_dim])
    k_sticks = concretize_expr(it_space_adjusted[k_dim])
    elems_per_stick = output_td.layout.device_layout.device_dtype.elems_per_stick()
    N_e = n_sticks * elems_per_stick
    K_e = k_sticks * elems_per_stick

    # Batch dims: track per-dim size for split enumeration; cost model only
    # sees the total batch count.
    batch_per_dim = [
        (bd, concretize_expr(it_space_adjusted[bd])) for bd in batch_dims
    ]
    B_total = 1
    for _, s in batch_per_dim:
        B_total *= s

    # Enumerate feasible splits.
    if batch_per_dim:
        bd_divs = [[int(d) for d in divisors(s)] for _, s in batch_per_dim]
        b_combos = list(itertools.product(*bd_divs))
    else:
        b_combos = [()]
    m_divs = [int(d) for d in divisors(M_e)]
    n_divs = [int(d) for d in divisors(n_sticks)]
    k_divs = [int(d) for d in divisors(k_sticks)]

    best_cost = float("inf")
    best = None
    for b_combo in b_combos:
        b_prod = 1
        for bs in b_combo:
            b_prod *= bs
        for mm in m_divs:
            for nn in n_divs:
                for kk in k_divs:
                    if b_prod * mm * nn * kk > max_cores:
                        continue
                    c = _matmul_split_cost(
                        B_total, M_e, K_e, N_e, b_prod, mm, nn, kk, max_cores
                    )
                    if c < best_cost:
                        best_cost = c
                        best = (b_combo, mm, nn, kk)

    if best is None:
        return splits

    b_combo, m_s, n_s, k_s = best
    new_splits = dict(splits)
    for (bd, _sz), bs in zip(batch_per_dim, b_combo):
        new_splits[bd] = int(bs)
    new_splits[m_dim] = m_s
    new_splits[n_dim] = n_s
    new_splits[k_dim] = k_s

    # Sanity: never use fewer cores than the default split.
    if math.prod(new_splits.values()) < math.prod(splits.values()):
        return splits

    logger.debug(
        f"cost_model work_division {op.get_name()}: "
        f"b={b_combo} m={m_s} n={n_s} k={k_s} cost={best_cost:.1f}us "
        f"[B={B_total} M={M_e} K={K_e} N={N_e}]"
    )
    return new_splits


def work_distribution_pass(
    op: ComputedBuffer,
    args: list[SchedNodeArg],
    max_cores: int,
    unsafe_for_2d: set[str] | None = None,
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
    if config.two_d_mn_split:
        splits = _cost_model_matmul_planner(
            op,
            splits,
            it_space_adjusted,
            output_td,
            stick_vars,
            committed_splits,
            max_cores,
            unsafe_for_2d if unsafe_for_2d is not None else set(),
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


def work_distribution(operations: list[Operation]) -> None:
    """Pass 3: distribute cores across ops via the cost-model planner (matmul)
    or the default multi-dim distributor (pointwise / non-matmul reductions).

    The cost-model planner (``_cost_model_matmul_planner``) subsumes the
    former k_fast / mn-cosplit / 2D-rule heuristics into a single search.
    """
    max_cores = _validate_max_cores()
    unsafe_for_2d = (
        _matmul_names_unsafe_for_2d(operations) if config.two_d_mn_split else set()
    )

    def pass_fn(op_, args_, max_cores_):
        work_distribution_pass(op_, args_, max_cores_, unsafe_for_2d)

    for op in _iter_computed_buffers(operations):
        rw = op.get_read_writes()
        args = get_mem_deps_from_rw(rw)
        if isinstance(op.data, Pointwise):
            divide_pointwise_op(op, args, max_cores, pass_fn)
        elif isinstance(op.data, Reduction):
            divide_reduction_op(op, args, max_cores, pass_fn)

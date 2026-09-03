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

"""Map a logical work division onto physical cores."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from itertools import permutations, product
from numbers import Integral
from typing import TYPE_CHECKING

from sympy import Expr, Integer, Mod, Symbol, floor, sympify

from .constants import BYTES_PER_STICK
from .op_spec import TensorWorkDivision

if TYPE_CHECKING:
    from .views import AlignmentInputs


_MAX_OWNER_PERMUTATION_DIMS = 5


_MAX_EXACT_DIRECT_AXIS_POINTS = 1 << 16
_DIRECT_AXIS_LOOP = Symbol("direct_axis_loop", integer=True, nonnegative=True)


@lru_cache(maxsize=256)
def _direct_axis_ownership_matches(
    loop_extent: int,
    candidate_split: int,
    candidate_slot_expr: Expr,
    device_extent: int,
    device_coordinate: Expr,
    physical_split: int,
    physical_slot_expr: Expr,
    num_cores: int,
) -> bool:
    """Exactly prove one loop's ownership of one physical device axis.

    Compiling the symbolic coordinate once keeps large canonical axes cheap.
    The cache key contains the complete logical and physical ownership claim,
    so differently ordered owners can never share a result.
    """

    try:
        if (
            loop_extent <= 0
            or loop_extent > _MAX_EXACT_DIRECT_AXIS_POINTS
            or candidate_split <= 0
            or physical_split <= 0
            or loop_extent % candidate_split
            or device_extent <= 0
            or device_extent % physical_split
            or candidate_split != physical_split
            or num_cores <= 0
        ):
            return False

        core_id = Symbol("core_id")
        candidate_slot_expr = sympify(candidate_slot_expr)
        physical_slot_expr = sympify(physical_slot_expr)
        device_coordinate = sympify(device_coordinate)
        if (
            candidate_slot_expr.free_symbols - {core_id}
            or physical_slot_expr.free_symbols - {core_id}
            or device_coordinate.free_symbols != {_DIRECT_AXIS_LOOP}
        ):
            return False

        candidate_slots = []
        physical_slots = []
        for core in range(num_cores):
            candidate_value = sympify(candidate_slot_expr.subs(core_id, core))
            physical_value = sympify(physical_slot_expr.subs(core_id, core))
            if (
                candidate_value.free_symbols
                or candidate_value.is_integer is not True
                or physical_value.free_symbols
                or physical_value.is_integer is not True
            ):
                return False
            candidate_slot = int(candidate_value)
            physical_slot = int(physical_value)
            if not 0 <= candidate_slot < candidate_split or not (
                0 <= physical_slot < physical_split
            ):
                return False
            candidate_slots.append(candidate_slot)
            physical_slots.append(physical_slot)

        coordinate = lambdify(_DIRECT_AXIS_LOOP, device_coordinate, modules="math")
        loop_partition_width = loop_extent // candidate_split
        physical_partition_width = device_extent // physical_split
        physical_slot_by_partition: dict[int, int] = {}
        for point in range(loop_extent):
            coordinate_value = coordinate(point)
            if not isinstance(coordinate_value, Integral):
                return False
            coordinate_value = int(coordinate_value)
            if not 0 <= coordinate_value < device_extent:
                return False
            partition = point // loop_partition_width
            physical_slot = coordinate_value // physical_partition_width
            previous = physical_slot_by_partition.setdefault(partition, physical_slot)
            if previous != physical_slot:
                return False

        if (
            len(physical_slot_by_partition) != candidate_split
            or len(set(physical_slot_by_partition.values())) != candidate_split
        ):
            return False
        return all(
            {
                core
                for core, slot in enumerate(physical_slots)
                if slot == physical_slot_by_partition[partition]
            }
            == {core for core, slot in enumerate(candidate_slots) if slot == partition}
            for partition in range(candidate_split)
        )
    except (
        AttributeError,
        ImportError,
        KeyError,
        NameError,
        NotImplementedError,
        OverflowError,
        SyntaxError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        return False


_MAX_EXACT_OWNERSHIP_POINTS = 1024


def work_division_matches_physical_ownership(
    division: TensorWorkDivision,
    loop_extents: Mapping[Symbol, int],
    device_size: Sequence[int],
    device_coordinates: Sequence[Expr],
    physical_splits: Sequence[tuple[int, int]],
    physical_core_to_slot: Sequence[tuple[int, Expr]],
    num_cores: int,
) -> bool:
    """Prove that loop and physical ownership assign every point identically.

    The proof deliberately treats coordinate expressions as black boxes.  It
    evaluates the concrete domain instead of trying to recognize floor/mod or
    mixed-radix formulas.  This is only for the small fused-axis cases that the
    ordinary one-loop-to-one-device-axis projection cannot represent.
    """

    try:
        if num_cores <= 0 or division.physical_core_count != num_cores:
            return False
        if len(device_size) != len(device_coordinates):
            return False
        if division.work_slices.keys() != division.core_id_to_work_slice.keys():
            return False
        if not division.work_slices.keys() <= loop_extents.keys():
            return False

        splits = dict(physical_splits)
        slots = dict(physical_core_to_slot)
        if (
            len(splits) != len(physical_splits)
            or len(slots) != len(physical_core_to_slot)
            or splits.keys() != slots.keys()
        ):
            return False

        core_id = Symbol("core_id")
        physical_slots_by_core: dict[int, dict[int, int]] = {}
        for device_dim, split in splits.items():
            if not 0 <= device_dim < len(device_size):
                return False
            extent = sympify(device_size[device_dim])
            if (
                extent.free_symbols
                or extent.is_integer is not True
                or int(extent) <= 0
                or split <= 0
                or int(extent) % split
            ):
                return False
            slot_expr = sympify(slots[device_dim])
            if slot_expr.free_symbols - {core_id}:
                return False
            for core in range(num_cores):
                value = sympify(slot_expr.subs(core_id, core))
                if value.free_symbols or value.is_integer is not True:
                    return False
                slot = int(value)
                if not 0 <= slot < split:
                    return False
                physical_slots_by_core.setdefault(core, {})[device_dim] = slot

        owned_coordinates = {
            device_dim: sympify(device_coordinates[device_dim]) for device_dim in splits
        }
        coordinate_symbols = set().union(
            *(coordinate.free_symbols for coordinate in owned_coordinates.values())
        )
        if coordinate_symbols - loop_extents.keys():
            return False
        relevant_dims = tuple(dim for dim in loop_extents if dim in coordinate_symbols)
        physical_dims_by_loop: dict[Symbol, list[int]] = {}
        for device_dim, coordinate in owned_coordinates.items():
            matches = coordinate.free_symbols & loop_extents.keys()
            if len(matches) != 1:
                return False
            physical_dims_by_loop.setdefault(next(iter(matches)), []).append(device_dim)
        concrete_extents = {}
        for dim in relevant_dims:
            value = sympify(loop_extents[dim])
            if value.free_symbols or value.is_integer is not True:
                return False
            concrete_extent = int(value)
            if concrete_extent <= 0:
                return False
            concrete_extents[dim] = concrete_extent
        candidate_splits = {
            dim: int(split)
            for dim, split in division.work_slices.items()
            if dim in coordinate_symbols and int(split) > 1
        }
        candidate_slots_by_core: dict[int, dict[Symbol, int]] = {}
        for dim, split in candidate_splits.items():
            if dim not in concrete_extents or concrete_extents[dim] % split:
                return False
            slot_expr = sympify(division.core_id_to_work_slice[dim])
            if slot_expr.free_symbols - {core_id}:
                return False
            for core in range(num_cores):
                value = sympify(slot_expr.subs(core_id, core))
                if value.free_symbols or value.is_integer is not True:
                    return False
                slot = int(value)
                if not 0 <= slot < split:
                    return False
                candidate_slots_by_core.setdefault(core, {})[dim] = slot

        # Prove each loop independently before combining its partitions.  This
        # is exact because every owned device coordinate depends on exactly one
        # loop symbol: once all values in a candidate partition have the same
        # physical slot tuple, the full Cartesian product is represented by the
        # Cartesian product of those tuples.  No loop value is sampled.
        physical_slots_by_partition: dict[Symbol, dict[int, tuple[int, ...]]] = {}
        exact_states = sum(concrete_extents.values())
        partition_count = math.prod(
            candidate_splits.get(dim, 1) for dim in relevant_dims
        )
        if exact_states + partition_count > _MAX_EXACT_OWNERSHIP_POINTS:
            return False

        for dim in relevant_dims:
            extent = concrete_extents[dim]
            split = candidate_splits.get(dim, 1)
            slots_by_partition: dict[int, tuple[int, ...]] = {}
            physical_dims = physical_dims_by_loop[dim]
            for point in range(extent):
                physical_slots = []
                for device_dim in physical_dims:
                    coordinate = owned_coordinates[device_dim]
                    value = sympify(coordinate.subs(dim, point))
                    if value.free_symbols or value.is_integer is not True:
                        return False
                    coordinate_value = int(value)
                    device_extent = int(device_size[device_dim])
                    if not 0 <= coordinate_value < device_extent:
                        return False
                    physical_slots.append(
                        coordinate_value // (device_extent // splits[device_dim])
                    )
                partition = point // (extent // split)
                signature = tuple(physical_slots)
                previous = slots_by_partition.setdefault(partition, signature)
                if previous != signature:
                    return False
            if len(slots_by_partition) != split:
                return False
            if len(set(slots_by_partition.values())) != split:
                return False
            physical_slots_by_partition[dim] = slots_by_partition

        partition_ranges = [
            range(candidate_splits.get(dim, 1)) for dim in relevant_dims
        ]
        for partitions in product(*partition_ranges):
            candidate_point_slots = {
                dim: partition
                for dim, partition in zip(relevant_dims, partitions)
                if candidate_splits.get(dim, 1) > 1
            }
            physical_point_slots = {}
            for dim, partition in zip(relevant_dims, partitions):
                for device_dim, slot in zip(
                    physical_dims_by_loop[dim],
                    physical_slots_by_partition[dim][partition],
                ):
                    physical_point_slots[device_dim] = slot

            physical_owners = {
                core
                for core in range(num_cores)
                if physical_slots_by_core.get(core, {}) == physical_point_slots
            }
            candidate_owners = {
                core
                for core in range(num_cores)
                if candidate_slots_by_core.get(core, {}) == candidate_point_slots
            }
            if not physical_owners or physical_owners != candidate_owners:
                return False
        return True
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def decompose_fused_split_view(
    fused_symbol: Symbol,
    fused_split: int,
    fused_slot_expr: Expr,
    loop_extents: Mapping[Symbol, int],
    device_size: Sequence[int],
    device_coordinates: Sequence[Expr],
    num_cores: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, Expr], ...]] | None:
    """Express one contiguous fused-loop partition on physical device axes.

    A fused loop may drive several physical axes. The ordinary view builder
    sees one split and therefore cannot place it on one axis. This helper
    enumerates the concrete fused domain, accepts only rectangular per-slot
    regions, constructs candidates with the canonical mapping generator, and
    lets the exact ownership proof decide. It never parses coordinate formulas
    or represents disjoint regions.
    """

    try:
        fused_split = int(fused_split)
        num_cores = int(num_cores)
        if fused_split <= 1 or num_cores <= 0 or num_cores % fused_split:
            return None
        if len(device_size) != len(device_coordinates):
            return None

        extent_expr = sympify(loop_extents[fused_symbol])
        if (
            extent_expr.free_symbols
            or extent_expr.is_integer is not True
            or int(extent_expr) <= 0
        ):
            return None
        extent = int(extent_expr)
        if extent % fused_split:
            return None
        if extent + fused_split > _MAX_EXACT_OWNERSHIP_POINTS:
            return None

        core_id = Symbol("core_id")
        fused_slot_expr = sympify(fused_slot_expr)
        if fused_slot_expr.free_symbols - {core_id}:
            return None

        # Replicated fused partitions must use the same contiguous owner groups
        # as the canonical tensor-division contract. Interleaved groups are a
        # different ownership representation and remain fail-closed.
        canonical_slot = derive_partition_mapping(
            (fused_symbol,), (fused_split,), num_cores
        )[fused_symbol]
        if any(
            sympify(fused_slot_expr.subs(core_id, core))
            != sympify(canonical_slot.subs(core_id, core))
            for core in range(num_cores)
        ):
            return None

        driven_dims = tuple(
            device_dim
            for device_dim, coordinate in enumerate(device_coordinates)
            if fused_symbol in sympify(coordinate).free_symbols
        )
        if not 2 <= len(driven_dims) <= 5:
            return None
        if any(
            sympify(device_coordinates[device_dim]).free_symbols != {fused_symbol}
            for device_dim in driven_dims
        ):
            return None

        concrete_device_size: dict[int, int] = {}
        for device_dim in driven_dims:
            device_extent = sympify(device_size[device_dim])
            if (
                device_extent.free_symbols
                or device_extent.is_integer is not True
                or int(device_extent) <= 0
            ):
                return None
            concrete_device_size[device_dim] = int(device_extent)

        coordinates_by_point: list[tuple[int, ...]] = []
        for point in range(extent):
            point_coordinates = []
            for device_dim in driven_dims:
                value = sympify(
                    sympify(device_coordinates[device_dim]).subs(fused_symbol, point)
                )
                if value.free_symbols or value.is_integer is not True:
                    return None
                coordinate = int(value)
                if not 0 <= coordinate < concrete_device_size[device_dim]:
                    return None
                point_coordinates.append(coordinate)
            coordinates_by_point.append(tuple(point_coordinates))

        slot_width = extent // fused_split
        run_widths: tuple[int, ...] | None = None
        for slot in range(fused_split):
            points = coordinates_by_point[slot * slot_width : (slot + 1) * slot_width]
            if len(set(points)) != slot_width:
                return None
            slot_run_widths = []
            for axis in range(len(driven_dims)):
                values = {point[axis] for point in points}
                if max(values) - min(values) + 1 != len(values):
                    return None
                slot_run_widths.append(len(values))
            widths = tuple(slot_run_widths)
            if math.prod(widths) != slot_width:
                return None
            if run_widths is None:
                run_widths = widths
            elif widths != run_widths:
                return None

        if run_widths is None:
            return None
        factor_by_dim = {
            device_dim: concrete_device_size[device_dim] // width
            for device_dim, width in zip(driven_dims, run_widths)
        }
        if any(
            concrete_device_size[device_dim] % width
            for device_dim, width in zip(driven_dims, run_widths)
        ):
            return None

        division = TensorWorkDivision(
            {fused_symbol: fused_split},
            {fused_symbol: fused_slot_expr},
            num_cores=num_cores,
        )
        split_dims = tuple(
            (device_dim, factor)
            for device_dim, factor in factor_by_dim.items()
            if factor > 1
        )
        if not split_dims:
            return None

        synthetic_dims = {
            device_dim: Symbol(f"physical_dim_{device_dim}")
            for device_dim in driven_dims
        }
        for ordered_device_dims in permutations(driven_dims):
            mapping = core_to_slice_mapping(
                tuple(synthetic_dims[device_dim] for device_dim in ordered_device_dims),
                tuple(factor_by_dim[device_dim] for device_dim in ordered_device_dims),
                fused_split,
            )
            core_slots = tuple(
                sorted(
                    (
                        device_dim,
                        mapping[synthetic_dims[device_dim]].subs(
                            core_id, fused_slot_expr
                        ),
                    )
                    for device_dim, factor in factor_by_dim.items()
                    if factor > 1
                )
            )
            if work_division_matches_physical_ownership(
                division,
                loop_extents,
                device_size,
                device_coordinates,
                split_dims,
                core_slots,
                num_cores,
            ):
                return tuple(sorted(split_dims)), core_slots
        return None
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def core_to_slice_mapping(
    dims: Sequence[Symbol],
    dim_splits: Sequence[int],
    num_cores: int,
    *,
    contiguous_dim: int | None = None,
) -> dict[Symbol, Expr]:
    """Return the logical work slice assigned to each physical core.

    By default dimensions vary in iteration-space order. ``contiguous_dim``
    moves one caller-selected dimension first so its participants are adjacent.
    """

    dims = tuple(dims)
    splits = tuple(dim_splits)
    if len(dims) != len(splits):
        raise ValueError(f"dimension/split count differs: {len(dims)} != {len(splits)}")

    logical_cores = math.prod(splits)
    if num_cores < logical_cores or num_cores % logical_cores != 0:
        raise ValueError(
            "num_cores must be a multiple of the logical work split "
            f"({logical_cores}), got {num_cores}"
        )

    dim_order = list(range(len(dims)))
    if contiguous_dim is not None and splits[contiguous_dim] > 1:
        dim_order.remove(contiguous_dim)
        dim_order.insert(0, contiguous_dim)

    core_id: Expr = Symbol("core_id")
    stride = Integer(1)
    result: dict[Symbol, Expr] = {}
    for dim in dim_order:
        split = Integer(splits[dim])
        if split == 1:
            coordinate = Integer(0)
        elif stride == 1:
            coordinate = Mod(core_id, split)
        else:
            coordinate = Mod(floor(core_id / stride), split)
        result[dims[dim]] = coordinate
        stride *= split
    return result


def derive_core_mapping(
    dims: Sequence[Symbol],
    dim_splits: Sequence[int],
    num_cores: int,
    *,
    contiguous_dim: Symbol | None = None,
    grouped_splits: Mapping[Symbol, int] | None = None,
) -> dict[Symbol, Expr]:
    """Derive one complete mapping from final dimensions and group geometry.

    ``grouped_splits`` describes logical owners that must occupy contiguous,
    equal-size core groups. Its device-dimension order defines the group
    topology; final loop order does not. Dimensions outside that mapping divide
    work within each group. No planning-time core assignment is consumed.
    """

    dims = tuple(dims)
    splits = tuple(int(split) for split in dim_splits)
    if len(dims) != len(splits):
        raise ValueError(f"dimension/split count differs: {len(dims)} != {len(splits)}")
    split_by_dim = dict(zip(dims, splits))
    if math.prod(splits) != num_cores:
        raise ValueError(
            f"operation split product must equal num_cores: {math.prod(splits)} != {num_cores}"
        )

    grouped_splits = dict(grouped_splits or {})
    unknown_dims = grouped_splits.keys() - split_by_dim.keys()
    if unknown_dims:
        raise ValueError(
            "grouped dimensions are not in the operation: "
            f"{sorted(map(str, unknown_dims))}"
        )
    for dim, split in grouped_splits.items():
        if int(split) != split_by_dim[dim]:
            raise ValueError(
                f"grouped split {dim}={split} does not match operation split "
                f"{split_by_dim[dim]}"
            )

    if not grouped_splits:
        contiguous_index = (
            dims.index(contiguous_dim) if contiguous_dim in dims else None
        )
        return core_to_slice_mapping(
            dims,
            splits,
            num_cores,
            contiguous_dim=contiguous_index,
        )

    grouped_dims = tuple(grouped_splits)
    local_dims = tuple(dim for dim in dims if dim not in grouped_splits)
    owner_count = math.prod(grouped_splits[dim] for dim in grouped_dims)
    if owner_count <= 0 or num_cores % owner_count:
        raise ValueError("grouped ownership does not divide the operation")
    group_size = num_cores // owner_count
    if math.prod(split_by_dim[dim] for dim in local_dims) != group_size:
        raise ValueError("operation splits do not fill each owner group")

    core_id = Symbol("core_id")
    group_id = floor(core_id / group_size)
    local_core_id = Mod(core_id, group_size)
    owner_mapping = core_to_slice_mapping(
        grouped_dims,
        tuple(grouped_splits[dim] for dim in grouped_dims),
        owner_count,
    )
    local_contiguous = (
        local_dims.index(contiguous_dim) if contiguous_dim in local_dims else None
    )
    local_mapping = core_to_slice_mapping(
        local_dims,
        tuple(split_by_dim[dim] for dim in local_dims),
        group_size,
        contiguous_dim=local_contiguous,
    )
    return {
        **{
            dim: expression.subs(core_id, group_id)
            for dim, expression in owner_mapping.items()
        },
        **{
            dim: expression.subs(core_id, local_core_id)
            for dim, expression in local_mapping.items()
        },
    }


def derive_partition_mapping(
    dims: Sequence[Symbol],
    dim_splits: Sequence[int],
    num_cores: int,
) -> dict[Symbol, Expr]:
    """Derive tensor owners from final partition geometry.

    A partition may have fewer logical owners than physical cores. In that
    case each owner occupies one contiguous, equal-size core group.
    """

    dims = tuple(dims)
    splits = tuple(int(split) for split in dim_splits)
    owner_count = math.prod(splits)
    if owner_count <= 0 or num_cores <= 0 or num_cores % owner_count:
        raise ValueError(
            f"partition owner count must divide num_cores: {owner_count}, {num_cores}"
        )
    group_size = num_cores // owner_count
    core_id = Symbol("core_id")
    group_id = floor(core_id / group_size)
    mapping = core_to_slice_mapping(dims, splits, owner_count)
    return {
        dim: expression.subs(core_id, group_id) for dim, expression in mapping.items()
    }


def remap_work_division(
    division: TensorWorkDivision,
    dimension_remap: Mapping[Symbol, Sequence[tuple[Symbol, int]]],
) -> TensorWorkDivision:
    """Express tensor ownership in an aligned iteration space.

    ``align_tensors`` may split one loop dimension into several dimensions. The
    physical partition does not change; only the symbols used to describe it do.
    """

    num_cores = division.physical_core_count
    new_splits: dict[Symbol, int] = {}
    new_core_map: dict[Symbol, Expr] = {}
    for old_dim, split in division.work_slices.items():
        new_dims = dimension_remap.get(old_dim)
        if new_dims is None:
            raise ValueError(f"tensor ownership dimension {old_dim} has no alignment")
        remaining_split = int(split)
        split_factors: list[tuple[Symbol, int]] = []
        if len(new_dims) == 1:
            split_factors = [(new_dims[0][0], remaining_split)]
            remaining_split = 1
        else:
            for new_dim, basis in reversed(new_dims):
                factor = math.gcd(remaining_split, int(basis))
                split_factors.append((new_dim, factor))
                remaining_split //= factor
            split_factors.reverse()
        if remaining_split != 1:
            raise ValueError(f"cannot normalize {split}-way split on {old_dim}")

        slot = division.core_id_to_work_slice[old_dim]
        slot_stride = 1
        for new_dim, factor in split_factors:
            if factor == 1:
                continue
            new_slot = Mod(floor(slot / slot_stride), factor)
            previous_split = new_splits.get(new_dim)
            previous_slot = new_core_map.get(new_dim)
            if previous_split is not None and (
                previous_split != factor
                or previous_slot is None
                or not core_mappings_equal(
                    {new_dim: previous_slot},
                    {new_dim: new_slot},
                    num_cores,
                )
            ):
                raise ValueError(f"conflicting normalized ownership on {new_dim}")
            new_splits[new_dim] = factor
            new_core_map[new_dim] = new_slot
            slot_stride *= factor
    return TensorWorkDivision(
        new_splits,
        new_core_map,
        num_cores=num_cores,
    )


def finalize_tensor_work_divisions(
    iteration_space: Mapping[Symbol, tuple[Expr, int]],
    divisions: Sequence[TensorWorkDivision | None],
) -> tuple[TensorWorkDivision | None, ...]:
    """Verify committed tensor owners in the final aligned iteration space."""

    result: list[TensorWorkDivision | None] = []
    for division in divisions:
        if division is None:
            result.append(None)
            continue
        work_slices = {
            dim: int(split)
            for dim, split in division.work_slices.items()
            if int(split) > 1
        }
        unknown_dims = work_slices.keys() - iteration_space.keys()
        if unknown_dims:
            raise ValueError(
                "tensor ownership dimensions are not aligned: "
                f"{sorted(map(str, unknown_dims))}"
            )

        try:
            core_map = {dim: division.core_id_to_work_slice[dim] for dim in work_slices}
        except KeyError as exc:
            raise ValueError(
                f"tensor ownership has no owner for {exc.args[0]}"
            ) from exc
        verified = TensorWorkDivision(
            work_slices,
            core_map,
            num_cores=division.physical_core_count,
        )
        verified.to_core_slices(verified.physical_core_count)
        result.append(verified)
    return tuple(result)


def operation_contiguous_dim(
    iteration_space: Mapping[Symbol, tuple[Expr, int]],
    *,
    is_matmul: bool,
    core_id_k_fast: bool,
) -> Symbol | None:
    """Return the one operation dimension selected to vary fastest by core."""

    return (
        next(reversed(iteration_space))
        if iteration_space and is_matmul and core_id_k_fast
        else None
    )


def _mapping_satisfies_division(
    mapping: Mapping[Symbol, Expr],
    division: TensorWorkDivision,
    num_cores: int,
) -> bool:
    if division.physical_core_count != num_cores:
        return False
    split_dims = {dim for dim, split in division.work_slices.items() if int(split) > 1}
    if not split_dims <= mapping.keys():
        return False
    try:
        return core_mappings_equal(
            {dim: mapping[dim] for dim in split_dims},
            {dim: division.core_id_to_work_slice[dim] for dim in split_dims},
            num_cores,
        )
    except KeyError:
        return False


def finalize_core_mapping_pure(
    alignment_inputs: "AlignmentInputs",
    tensor_divisions: Sequence[TensorWorkDivision | None],
    *,
    is_matmul: bool,
    core_id_k_fast: bool,
    is_relayout: bool,
) -> tuple[
    dict[Symbol, tuple[Expr, int]],
    list[dict[str, list]],
    tuple[TensorWorkDivision | None, ...],
    dict[Symbol, Expr],
    dict[Symbol, tuple[tuple[Symbol, int], ...]],
]:
    """Align tensors and adopt their committed physical ownership once.

    The scheduler preflight and codegen call this same pure sequence. It may
    translate dimension names, but it never chooses new owners for a buffer.
    """

    from .views import align_tensors_pure

    aligned_space, tensors, dimension_remap = align_tensors_pure(alignment_inputs)
    if len(tensor_divisions) != len(tensors):
        raise ValueError(
            "tensor division count does not match aligned tensor count: "
            f"{len(tensor_divisions)} != {len(tensors)}"
        )
    remapped = tuple(
        remap_work_division(division, dimension_remap) if division is not None else None
        for division in tensor_divisions
    )
    divisions = finalize_tensor_work_divisions(aligned_space, remapped)
    logical_cores = math.prod(int(split) for _, split in aligned_space.values())

    if is_relayout:
        if len(divisions) != 2:
            raise ValueError(
                "LX relayout finalization requires source and destination ownership"
            )
        destination = divisions[-1]
        if destination is None:
            raise ValueError("LX relayout destination has no committed ownership")
        # A relayout is the one operation where tensor ownership intentionally
        # differs from the logical copy loop split. The destination physical
        # view defines the copy's core map; SuperDSC carries each tensor's own
        # domain and raises the execution domain to the largest nested domain.
        mapping = dict(destination.core_id_to_work_slice)
        if not _mapping_satisfies_division(
            mapping, destination, destination.physical_core_count
        ):
            raise ValueError(
                "LX relayout destination ownership does not define its operation map"
            )
        source = divisions[0]
        if source is None:
            raise ValueError("LX relayout source has no committed ownership")
        execution_cores = max(
            logical_cores,
            source.physical_core_count,
            destination.physical_core_count,
        )
        domains = {
            "operation": logical_cores,
            "source": source.physical_core_count,
            "destination": destination.physical_core_count,
        }
        owner_counts = {
            "source": math.prod(int(split) for split in source.work_slices.values()),
            "destination": math.prod(
                int(split) for split in destination.work_slices.values()
            ),
        }
        invalid_domain = next(
            (
                name
                for name, domain in domains.items()
                if domain <= 0 or execution_cores % domain
            ),
            None,
        )
        invalid_owners = next(
            (
                name
                for name, owners in owner_counts.items()
                if owners <= 0 or domains[name] % owners
            ),
            None,
        )
        if invalid_domain is not None or invalid_owners is not None:
            raise ValueError(
                "LX relayout operation and tensor core domains must divide the "
                f"execution domain; domains={domains}, owners={owner_counts}"
            )
        oversized = {
            name: {
                str(dim): (int(split), int(aligned_space[dim][0]))
                for dim, split in division.work_slices.items()
                if sympify(aligned_space[dim][0]).is_number
                and int(split) > int(aligned_space[dim][0])
            }
            for name, division in (("source", source), ("destination", destination))
        }
        oversized = {name: dims for name, dims in oversized.items() if dims}
        if oversized:
            raise ValueError(
                f"LX relayout tensor split exceeds its aligned extent: {oversized}"
            )
        if source.same_ownership(destination):
            raise ValueError(
                "LX relayout source and destination ownership collapse after alignment"
            )
    else:
        num_cores = logical_cores
        mapping = derive_operation_mapping(
            aligned_space,
            divisions,
            contiguous_dim=operation_contiguous_dim(
                aligned_space,
                is_matmul=is_matmul,
                core_id_k_fast=core_id_k_fast,
            ),
        )
        for division in divisions:
            if division is not None and not _mapping_satisfies_division(
                mapping, division, num_cores
            ):
                raise ValueError(
                    "final operation map does not reproduce committed LX ownership"
                )

    return aligned_space, tensors, divisions, mapping, dimension_remap


def derive_operation_mapping(
    iteration_space: Mapping[Symbol, tuple[Expr, int]],
    tensor_divisions: Sequence[TensorWorkDivision | None] = (),
    *,
    contiguous_dim: Symbol | None = None,
) -> dict[Symbol, Expr]:
    """Derive one operation mapping that satisfies every LX tensor owner."""

    dims = tuple(iteration_space)
    splits = tuple(int(iteration_space[dim][1]) for dim in dims)
    num_cores = math.prod(splits)
    split_by_dim = dict(zip(dims, splits))
    constrained: dict[Symbol, Expr] = {}
    for division in tensor_divisions:
        if division is None:
            continue
        if division.work_slices and division.physical_core_count != num_cores:
            raise ValueError(
                "LX tensor ownership and operation use different core domains: "
                f"{division.physical_core_count} != {num_cores}"
            )
        for dim, split in division.work_slices.items():
            if int(split) <= 1:
                continue
            if dim not in split_by_dim:
                raise ValueError(f"LX tensor dimension {dim} is not in the operation")
            if split_by_dim[dim] != int(split):
                raise ValueError(
                    f"LX tensor split for {dim} does not match the operation: "
                    f"{split} != {split_by_dim[dim]}"
                )
            expression = division.core_id_to_work_slice[dim]
            previous = constrained.setdefault(dim, expression)
            if not core_mappings_equal({dim: previous}, {dim: expression}, num_cores):
                raise ValueError(f"LX tensors disagree on core ownership for {dim}")

    if not constrained:
        return derive_core_mapping(
            dims,
            splits,
            num_cores,
            contiguous_dim=contiguous_dim,
        )

    # Preserve main's operation map whenever it already satisfies the physical
    # tensor owners. The grouped search below is only needed when it does not.
    default = derive_core_mapping(
        dims,
        splits,
        num_cores,
        contiguous_dim=contiguous_dim,
    )
    if all(
        core_mappings_equal({dim: default[dim]}, {dim: expression}, num_cores)
        for dim, expression in constrained.items()
    ):
        return default

    # Tensor-owned dimensions occupy the outer, contiguous groups. At most five
    # dimensions can be split on 32 cores, so trying their radix orders is small.
    if len(constrained) > _MAX_OWNER_PERMUTATION_DIMS:
        raise ValueError(
            "too many aligned tensor-owned dimensions for bounded core-order "
            f"search: {len(constrained)} > {_MAX_OWNER_PERMUTATION_DIMS}"
        )
    for order in permutations(sorted(constrained, key=str)):
        candidate = derive_core_mapping(
            dims,
            splits,
            num_cores,
            contiguous_dim=contiguous_dim,
            grouped_splits={dim: split_by_dim[dim] for dim in order},
        )
        if all(
            core_mappings_equal({dim: candidate[dim]}, {dim: expression}, num_cores)
            for dim, expression in constrained.items()
        ):
            return candidate

    raise ValueError("no operation core mapping satisfies every LX tensor owner")


def partition_physical_span_bytes(
    device_size: Sequence[int],
    elems_per_stick: int,
    split_by_device_dim: Mapping[int, int],
) -> int:
    """Bound a standard-layout partition, retaining physical gaps between rows.

    Device dimensions are stored in decreasing physical-stride order. The
    layout's ``stride_map`` instead addresses HOST memory and must not size LX.
    A backend may pack a partition more tightly; retaining the original device
    strides is a conservative bound. The final dimension is one complete stick
    and is never split. This measures placement, not the split-cost estimate.
    """

    if not device_size or any(extent <= 0 for extent in device_size):
        raise ValueError("device extents must be positive")
    if elems_per_stick <= 0:
        raise ValueError("elems_per_stick must be positive")
    for dim, split in split_by_device_dim.items():
        if dim < 0 or dim >= len(device_size) or split <= 0:
            raise ValueError(f"invalid split {split} on device dimension {dim}")
    if device_size[-1] != elems_per_stick:
        raise ValueError("physical span requires one complete final stick dimension")
    if split_by_device_dim.get(len(device_size) - 1, 1) != 1:
        raise ValueError("the final stick dimension cannot be split")

    span_sticks = stride_sticks = 1
    for dim in reversed(range(len(device_size) - 1)):
        extent = device_size[dim]
        split = split_by_device_dim.get(dim, 1)
        slice_extent = (extent + split - 1) // split
        span_sticks += (slice_extent - 1) * stride_sticks
        stride_sticks *= extent
    return span_sticks * BYTES_PER_STICK


def core_mappings_equal(
    left: Mapping[Symbol, Expr],
    right: Mapping[Symbol, Expr],
    num_cores: int,
) -> bool:
    """Return whether two symbolic mappings assign every core identically."""

    if left.keys() != right.keys():
        return False
    if num_cores <= 0:
        return False
    core_id = Symbol("core_id")
    try:
        for dim in left:
            for core in range(num_cores):
                values = [
                    sympify(mapping[dim]).subs(core_id, core)
                    for mapping in (left, right)
                ]
                if any(
                    value.free_symbols or value.is_integer is not True
                    for value in values
                ):
                    return False
                if values[0] != values[1]:
                    return False
        return True
    except (TypeError, ValueError):
        return False

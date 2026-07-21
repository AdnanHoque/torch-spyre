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

"""Map a logical work division onto physical core IDs.

The mapping mechanics in this module are deliberately independent of Inductor
IR, compiler configuration, hints, and SDSC codegen.  Callers select a policy;
this module resolves it into one exact mixed-radix linearization and
materializes the corresponding SymPy expressions.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import operator
from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

from sympy import Expr, Integer, Mod, Symbol, floor


class CoreMappingPolicy(str, enum.Enum):
    """The exact logical-axis ordering to use on physical core IDs."""

    NATURAL = "natural"
    REDUCTION_FAST = "reduction_fast"
    ROW_MAJOR = "row_major"


_AXIS: Literal["axis"] = "axis"
_REPLICA: Literal["replica"] = "replica"
CoreMappingLevel: TypeAlias = tuple[Literal["axis", "replica"], int]


@dataclasses.dataclass(frozen=True)
class ResolvedCoreMapping:
    """Exact mixed-radix physical-core linearization.

    ``levels`` are ordered from fastest- to slowest-varying along physical
    ``core_id``.  ``("axis", ordinal)`` identifies an iteration-space axis;
    ``("replica", factor)`` reserves a physical level which does not change a
    logical coordinate.  Ordinals, rather than loop symbols, keep the resolved
    choice stable when later normalization renames iteration variables.
    """

    levels: tuple[CoreMappingLevel, ...]

    def __post_init__(self) -> None:
        for level in self.levels:
            if len(level) != 2:
                raise ValueError(f"invalid core-mapping level {level!r}")
            kind, value = level
            if kind == _AXIS:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"invalid axis level {level!r}")
            elif kind == _REPLICA:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"invalid replica level {level!r}")
            else:
                raise ValueError(f"unknown core-mapping level kind {kind!r}")


def _positive_splits(axis_splits: Sequence[int]) -> tuple[int, ...]:
    splits: list[int] = []
    for split in axis_splits:
        if isinstance(split, bool):
            raise ValueError(f"axis splits must be positive integers, got {split!r}")
        try:
            value = operator.index(split)
        except TypeError as exc:
            raise ValueError(
                f"axis splits must be positive integers, got {split!r}"
            ) from exc
        if value <= 0:
            raise ValueError(f"axis splits must be positive, got {value!r}")
        splits.append(value)
    return tuple(splits)


def _positive_core_count(num_cores: int) -> int:
    if isinstance(num_cores, bool):
        raise ValueError(f"num_cores must be a positive integer, got {num_cores!r}")
    try:
        value = operator.index(num_cores)
    except TypeError as exc:
        raise ValueError(
            f"num_cores must be a positive integer, got {num_cores!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"num_cores must be positive, got {value!r}")
    return value


def _reduction_axis_ordinal(reduction_axis: int | None, num_axes: int) -> int:
    if reduction_axis is None or isinstance(reduction_axis, bool):
        raise ValueError(
            f"reduction_axis must be an integer within {num_axes} axes, "
            f"got {reduction_axis!r}"
        )
    try:
        value = operator.index(reduction_axis)
    except TypeError as exc:
        raise ValueError(
            f"reduction_axis must be an integer within {num_axes} axes, "
            f"got {reduction_axis!r}"
        ) from exc
    if value < 0 or value >= num_axes:
        raise ValueError(f"reduction_axis {value!r} is outside {num_axes} axes")
    return value


def _mapping_core_product(
    mapping: ResolvedCoreMapping, axis_splits: Sequence[int]
) -> int:
    splits = _positive_splits(axis_splits)
    product = 1
    for kind, value in mapping.levels:
        if kind == _AXIS:
            if value >= len(splits):
                raise ValueError(
                    f"core mapping axis {value} is outside {len(splits)} axes"
                )
            product *= splits[value]
        else:
            product *= value
    return product


def validate_core_mapping(
    mapping: ResolvedCoreMapping,
    axis_splits: Sequence[int],
    num_cores: int,
) -> None:
    """Validate a full axis permutation and its physical-core product."""

    splits = _positive_splits(axis_splits)
    axes = [value for kind, value in mapping.levels if kind == _AXIS]
    expected_axes = list(range(len(splits)))
    if sorted(axes) != expected_axes:
        raise ValueError(
            "core mapping must contain every axis ordinal exactly once; "
            f"expected {expected_axes!r}, got {axes!r}"
        )
    num_cores = _positive_core_count(num_cores)
    product = _mapping_core_product(mapping, splits)
    if product != num_cores:
        raise ValueError(
            "core-mapping level product must equal num_cores; "
            f"got {product} and {num_cores}"
        )


def resolve_core_mapping(
    axis_splits: Sequence[int],
    num_cores: int,
    *,
    policy: CoreMappingPolicy = CoreMappingPolicy.NATURAL,
    reduction_axis: int | None = None,
) -> ResolvedCoreMapping:
    """Resolve one explicit mapping policy.

    ``NATURAL`` exactly preserves the legacy first-axis-fastest mapping.
    ``REDUCTION_FAST`` promotes a split reduction axis to the fastest physical
    level, keeping partial-sum participants adjacent.  ``ROW_MAJOR`` makes the
    last logical axis fastest.  Extra physical cores repeat the complete
    logical grid and therefore do not alter logical-axis strides.
    """

    splits = _positive_splits(axis_splits)
    num_cores = _positive_core_count(num_cores)
    logical_cores = math.prod(splits)
    if num_cores < logical_cores or num_cores % logical_cores != 0:
        raise ValueError(
            "num_cores must be a positive multiple of the logical work split "
            f"({logical_cores}), got {num_cores}"
        )
    replicas = num_cores // logical_cores

    if policy == CoreMappingPolicy.NATURAL:
        axis_order = list(range(len(splits)))
    elif policy == CoreMappingPolicy.ROW_MAJOR:
        axis_order = list(reversed(range(len(splits))))
    elif policy == CoreMappingPolicy.REDUCTION_FAST:
        reduction_axis = _reduction_axis_ordinal(reduction_axis, len(splits))
        axis_order = list(range(len(splits)))
        if splits[reduction_axis] > 1:
            axis_order.remove(reduction_axis)
            axis_order.insert(0, reduction_axis)
    else:
        raise ValueError(
            f"unknown core mapping policy {policy!r}; expected a CoreMappingPolicy"
        )

    levels: list[CoreMappingLevel] = [(_AXIS, ordinal) for ordinal in axis_order]
    if replicas > 1:
        levels.append((_REPLICA, replicas))

    mapping = ResolvedCoreMapping(tuple(levels))
    validate_core_mapping(mapping, splits, num_cores)
    return mapping


def select_core_mapping_policy(
    axis_splits: Sequence[int],
    *,
    reduction_axis: int | None,
    enable_reduction_fast: bool,
) -> CoreMappingPolicy:
    """Select the existing automatic policy from explicit compiler inputs."""

    splits = _positive_splits(axis_splits)
    if not enable_reduction_fast or len(splits) < 3:
        return CoreMappingPolicy.NATURAL
    reduction_axis = _reduction_axis_ordinal(reduction_axis, len(splits))
    if splits[reduction_axis] > 1:
        return CoreMappingPolicy.REDUCTION_FAST
    return CoreMappingPolicy.NATURAL


def materialize_core_mapping(
    mapping: ResolvedCoreMapping,
    axis_symbols: Sequence[Symbol],
    axis_splits: Sequence[int] | Mapping[Symbol, int],
    num_cores: int,
    *,
    core_id: Symbol | None = None,
) -> dict[str, Expr]:
    """Build per-axis SymPy coordinates for a resolved linearization."""

    symbols = tuple(axis_symbols)
    if isinstance(axis_splits, Mapping):
        splits = _positive_splits(tuple(axis_splits[symbol] for symbol in symbols))
    else:
        splits = _positive_splits(axis_splits)
    if len(symbols) != len(splits):
        raise ValueError(
            f"axis symbol/split count differs: {len(symbols)} != {len(splits)}"
        )
    validate_core_mapping(mapping, splits, num_cores)

    core_id = core_id if core_id is not None else Symbol("core_id")
    stride = Integer(1)
    result: dict[str, Expr] = {}
    for kind, value in mapping.levels:
        if kind == _REPLICA:
            stride *= Integer(value)
            continue
        split = Integer(splits[value])
        if split == 1:
            expr = Integer(0)
        elif stride == 1:
            expr = Mod(core_id, split)
        else:
            expr = Mod(floor(core_id / stride), split)
        result[str(symbols[value])] = expr
        stride *= split
    return result

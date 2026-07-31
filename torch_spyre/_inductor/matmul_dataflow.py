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

"""Shared selection helpers for opt-in matmul dataflows."""

from collections.abc import Set


ACTIVATION_STATIONARY_SHAPES_ERROR = (
    "SPYRE_ACTIVATION_STATIONARY_SHAPES must be a comma-separated KxN list"
)


def parse_activation_stationary_shapes(spec: str) -> Set[tuple[int, int]] | None:
    """Parse the optional KxN allowlist.

    ``None`` means that no allowlist was supplied, so every otherwise eligible
    shape is selected.
    """
    spec = spec.strip()
    if not spec:
        return None

    shapes: set[tuple[int, int]] = set()
    try:
        for item in spec.split(","):
            values = tuple(int(value) for value in item.strip().lower().split("x"))
            if len(values) != 2 or any(value <= 0 for value in values):
                raise ValueError
            shapes.add(values)
    except ValueError as error:
        raise ValueError(ACTIVATION_STATIONARY_SHAPES_ERROR) from error
    return shapes


def activation_stationary_shape_is_selected(k: int, n: int, spec: str) -> bool:
    shapes = parse_activation_stationary_shapes(spec)
    return shapes is None or (k, n) in shapes


def activation_stationary_work_division(
    logical_m: int,
    k: int,
    n: int,
) -> dict[str, int] | None:
    """Return a device-measured split for the reversed Design A BMM.

    The BMM computes ``W[N, K] @ A[K, physical_M]`` after padding logical M
    to a 64-row boundary. Keep this table deliberately narrow: every entry
    has passed the matched one-BMM/no-conversion device oracle.
    """
    physical_m = ((logical_m + 63) // 64) * 64
    if (k, n) == (4096, 1024):
        if physical_m == 64:
            return {"M": 1, "N": 16, "K": 2}
        if physical_m == 512:
            return {"M": 8, "N": 4, "K": 1}
    if (physical_m, k, n) == (512, 12800, 4096):
        return {"M": 4, "N": 8, "K": 1}
    return None

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

"""Default-off restickify core mapping alignment."""

from __future__ import annotations

from torch._inductor.ir import ComputedBuffer, Operation

from . import config
from .logging_utils import get_inductor_logger
from .restickify_ring import (
    CORE_MAPPING_OVERRIDE_ATTR,
    build_name_to_op_map,
    build_restickify_core_mapping_override,
    is_restickify_op,
)

logger = get_inductor_logger("mapping_alignment")


def align_restickify_core_mappings(operations: list[Operation]) -> None:
    """Attach producer-aligned core mapping overrides to compatible restickifies."""
    if not config.align_restickify_core_mapping:
        return

    name_to_op = build_name_to_op_map(operations)
    for op in operations:
        if not isinstance(op, ComputedBuffer) or not is_restickify_op(op):
            continue
        override, reason = build_restickify_core_mapping_override(op, name_to_op)
        if override is None:
            logger.info(
                "skip restickify core mapping alignment for %s: %s",
                op.get_name(),
                reason,
            )
            continue
        setattr(op, CORE_MAPPING_OVERRIDE_ATTR, override)
        logger.info(
            "attached restickify core mapping override for %s (%d cores)",
            op.get_name(),
            len(override),
        )

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
    LOCALITY_CERTIFICATE_ATTR,
    build_name_to_op_map,
    build_restickify_core_mapping_override,
    is_restickify_op,
    locality_certificate_for_restickify_override,
)

logger = get_inductor_logger("mapping_alignment")


def align_restickify_core_mappings(
    operations: list[Operation],
    k_fast_ops: list[Operation] | None = None,
) -> None:
    """Attach producer-aligned core mapping overrides to compatible restickifies."""
    if not config.align_restickify_core_mapping:
        return

    name_to_op = build_name_to_op_map(operations)
    for op in operations:
        if not isinstance(op, ComputedBuffer) or not is_restickify_op(op):
            continue
        override, reason = build_restickify_core_mapping_override(
            op,
            name_to_op,
            k_fast_ops,
        )
        if override is None:
            if config.restickify_locality_assert:
                certificate = locality_certificate_for_restickify_override(
                    op,
                    name_to_op,
                    override=None,
                    ring_size=config.sencores,
                    k_fast_ops=k_fast_ops,
                )
                setattr(op, LOCALITY_CERTIFICATE_ATTR, certificate)
            logger.info(
                "skip restickify core mapping alignment for %s: %s",
                op.get_name(),
                reason,
            )
            continue
        if config.restickify_locality_assert:
            certificate = locality_certificate_for_restickify_override(
                op,
                name_to_op,
                override,
                ring_size=config.sencores,
                k_fast_ops=k_fast_ops,
            )
            setattr(op, LOCALITY_CERTIFICATE_ATTR, certificate)
            if not certificate.locality_certified:
                raise RuntimeError(
                    "restickify locality assertion failed for "
                    f"{op.get_name()}: assertion={certificate.locality_assertion}, "
                    f"reason={certificate.locality_skip_reason}, "
                    f"certified_byte_hops={certificate.certified_byte_hops}, "
                    f"producer_splits={certificate.producer_splits}, "
                    f"restickify_splits={certificate.restickify_splits}, "
                    f"symbol_map={certificate.symbol_map}"
                )
        setattr(op, CORE_MAPPING_OVERRIDE_ATTR, override)
        logger.info(
            "attached restickify core mapping override for %s (%d cores)",
            op.get_name(),
            len(override),
        )

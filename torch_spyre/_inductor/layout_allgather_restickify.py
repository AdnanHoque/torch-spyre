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

"""Import-light classifier for flash layout all-gather restickify SDSCs."""

from __future__ import annotations

from typing import Any

LAYOUT_ALLGATHER_RESTICKIFY = "layout_allgather_restickify"
COMM_CLASS_ALL_GATHER = "all_gather"
RESTICKIFY_HBM_OP = "ReStickifyOpHBM"
RESTICKIFY_LX_OP = "ReStickifyOpLx"
RESTICKIFY_OPS = {RESTICKIFY_HBM_OP, RESTICKIFY_LX_OP}


def make_layout_allgather_restickify_contract(
    *,
    producer_op: str,
    restickify_op: str,
    consumer_op: str,
    producer_work_slice_dims: dict[str, int],
    restickify_work_slice_dims: dict[str, int],
    consumer_work_slice_dims: dict[str, int],
) -> dict[str, Any]:
    """Return the logical contract needed to lower the flash restickify edge.

    This is intentionally a logical contract, not a physical movement list. The
    backend still owns route selection and transfer synthesis, but it must see
    the layout/stick transform and dimension rename to avoid treating the edge
    as a direct scatter.
    """

    return {
        "kind": LAYOUT_ALLGATHER_RESTICKIFY,
        "classification": LAYOUT_ALLGATHER_RESTICKIFY,
        "producer_op": producer_op,
        "restickify_op": restickify_op,
        "consumer_op": consumer_op,
        "producer_work_slice_dims": dict(producer_work_slice_dims),
        "restickify_work_slice_dims": dict(restickify_work_slice_dims),
        "consumer_work_slice_dims": dict(consumer_work_slice_dims),
        "producer_layout": {
            "layoutDimOrder_": ["out", "x", "mb"],
            "stickDimOrder_": ["out"],
        },
        "restickify_kernel_layout": {
            "layoutDimOrder_": ["x", "out", "mb"],
            "stickDimOrder_": ["x"],
        },
        "consumer_kernel_layout": {
            "layoutDimOrder_": ["out", "in", "x"],
            "stickDimOrder_": ["out"],
        },
        "dimension_rename": {
            "restickify.x": "batchmatmul.out",
            "restickify.out": "batchmatmul.in",
            "restickify.mb": "batchmatmul.x",
        },
        "communication_class": COMM_CLASS_ALL_GATHER,
        "communication_pattern": LAYOUT_ALLGATHER_RESTICKIFY,
        "requires_staged_realization": True,
    }


def _sdsc_op_name(sdsc: dict[str, Any]) -> str:
    return str(sdsc.get("op") or sdsc.get("top_key", "").split("_", 1)[-1])


def _sdsc_work_slices(sdsc: dict[str, Any]) -> dict[str, int]:
    return {
        str(dim): int(split)
        for dim, split in sdsc.get("numWkSlicesPerDim_", {}).items()
    }


def _sdsc_layout(
    sdsc: dict[str, Any], label: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    info = (sdsc.get("primaryDsInfo_") or {}).get(label) or {}
    return (
        tuple(str(dim) for dim in info.get("layoutDimOrder_", ())),
        tuple(str(dim) for dim in info.get("stickDimOrder_", ())),
    )


def _sdsc_has_allocation(
    sdsc: dict[str, Any], *, component: str, layout_dim_order: tuple[str, ...]
) -> bool:
    for allocation in sdsc.get("allocates", ()):
        if allocation.get("component_") != component:
            continue
        if tuple(allocation.get("layoutDimOrder_", ())) == layout_dim_order:
            return True
    return False


def classify_layout_allgather_restickify_sdsc_triplet(
    producer: dict[str, Any],
    restickify: dict[str, Any] | None = None,
    consumer: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Classify the flash pointwise -> HBM restickify -> BMM KERNEL edge.

    This SDSC-snippet classifier is metadata-only. A returned record describes
    the frontend planning gap but intentionally does not claim backend lowering
    support. The accepted shape is narrow and mirrors the representative flash
    edge captured in flash_layout_restickify_gap_20260701.
    """

    if restickify is None and consumer is None:
        triplet = producer
        producer = triplet.get("sdsc_1_mul") or triplet.get("producer") or {}
        restickify = (
            triplet.get("sdsc_2_restickify") or triplet.get("restickify") or {}
        )
        consumer = triplet.get("sdsc_3_batchmatmul") or triplet.get("consumer") or {}
    if restickify is None or consumer is None:
        return None

    producer_op = _sdsc_op_name(producer)
    restickify_op = _sdsc_op_name(restickify)
    consumer_op = _sdsc_op_name(consumer)
    if producer_op != "mul":
        return None
    if restickify_op not in RESTICKIFY_OPS:
        return None
    if consumer_op != "batchmatmul":
        return None

    producer_output_layout = _sdsc_layout(producer, "OUTPUT")
    restickify_input_layout = _sdsc_layout(restickify, "OUTPUT")
    restickify_kernel_layout = _sdsc_layout(restickify, "KERNEL")
    consumer_kernel_layout = _sdsc_layout(consumer, "KERNEL")
    if producer_output_layout != (("out", "x", "mb"), ("out",)):
        return None
    if restickify_input_layout != producer_output_layout:
        return None
    if restickify_kernel_layout != (("x", "out", "mb"), ("x",)):
        return None
    if consumer_kernel_layout != (("out", "in", "x"), ("out",)):
        return None
    if not _sdsc_has_allocation(
        producer, component="lx", layout_dim_order=producer_output_layout[0]
    ):
        return None

    producer_splits = _sdsc_work_slices(producer)
    restickify_splits = _sdsc_work_slices(restickify)
    consumer_splits = _sdsc_work_slices(consumer)
    if producer_splits != restickify_splits:
        return None
    if producer_splits.get("mb") != consumer_splits.get("x"):
        return None
    if producer_splits.get("x") != consumer_splits.get("mb"):
        return None
    if producer_splits.get("out", 1) != 1 or consumer_splits.get("out", 1) != 1:
        return None
    if consumer_splits.get("in", 1) != 1:
        return None
    if int(producer.get("numCoresUsed_", 0)) != int(consumer.get("numCoresUsed_", -1)):
        return None

    return {
        **make_layout_allgather_restickify_contract(
            producer_op=producer_op,
            restickify_op=restickify_op,
            consumer_op=consumer_op,
            producer_work_slice_dims=producer_splits,
            restickify_work_slice_dims=restickify_splits,
            consumer_work_slice_dims=consumer_splits,
        ),
        "realized": False,
        "unsupported_reason": (
            "layout_allgather_restickify is metadata-only; "
            "backend lowering is not implemented"
        ),
    }

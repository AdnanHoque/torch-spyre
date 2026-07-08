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

"""DLDSC relayout contracts emitted by the LX planner."""

from __future__ import annotations

from typing import Any

COMM_CLASS_ALL_GATHER = "all_gather"
MATMUL_OPERAND_BROADCAST = "matmul_operand_broadcast"
MATMUL_OPERAND_ALLGATHER_REPLICATE = "all_gather_replicate"


def make_matmul_operand_allgather_contract(
    *,
    producer_op: str,
    consumer_op: str,
    read_index: int,
    producer_work_slice_dims: dict[str, int],
    consumer_tensor_work_slice_dims: dict[str, int],
    consumer_compute_work_slice_dims: dict[str, int],
) -> dict[str, Any]:
    """Return the logical contract for bounded matmul operand materialization."""

    operand_role = {0: "lhs", 1: "rhs"}.get(int(read_index), f"input_{read_index}")

    return {
        "kind": MATMUL_OPERAND_BROADCAST,
        "classification": MATMUL_OPERAND_BROADCAST,
        "producer_op": producer_op,
        "consumer_op": consumer_op,
        "operand_read_index": int(read_index),
        "operand_role": operand_role,
        "producer_work_slice_dims": dict(producer_work_slice_dims),
        "consumer_tensor_work_slice_dims": dict(consumer_tensor_work_slice_dims),
        "consumer_compute_work_slice_dims": dict(consumer_compute_work_slice_dims),
        "operand_kernel_layout": {
            "layoutDimOrder_": ["out", "in", "x"],
            "stickDimOrder_": ["out"],
        },
        "communication_class": COMM_CLASS_ALL_GATHER,
        "communication_pattern": MATMUL_OPERAND_ALLGATHER_REPLICATE,
        "requires_layout_conversion": True,
        "requires_staged_realization": True,
    }

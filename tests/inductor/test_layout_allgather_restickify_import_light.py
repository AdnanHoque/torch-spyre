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

from torch_spyre._inductor.layout_allgather_restickify import (
    COMM_CLASS_ALL_GATHER,
    LAYOUT_ALLGATHER_RESTICKIFY,
    classify_layout_allgather_restickify_sdsc_triplet,
)


def _flash_triplet():
    return {
        "sdsc_1_mul": {
            "op": "mul",
            "numCoresUsed_": 32,
            "numWkSlicesPerDim_": {"mb": 4, "x": 8, "out": 1},
            "primaryDsInfo_": {
                "OUTPUT": {
                    "layoutDimOrder_": ["out", "x", "mb"],
                    "stickDimOrder_": ["out"],
                }
            },
            "allocates": [
                {"component_": "lx", "layoutDimOrder_": ["out", "x", "mb"]}
            ],
        },
        "sdsc_2_restickify": {
            "op": "ReStickifyOpHBM",
            "numCoresUsed_": 32,
            "numWkSlicesPerDim_": {"mb": 4, "x": 8, "out": 1},
            "primaryDsInfo_": {
                "OUTPUT": {
                    "layoutDimOrder_": ["out", "x", "mb"],
                    "stickDimOrder_": ["out"],
                },
                "KERNEL": {
                    "layoutDimOrder_": ["x", "out", "mb"],
                    "stickDimOrder_": ["x"],
                },
            },
        },
        "sdsc_3_batchmatmul": {
            "op": "batchmatmul",
            "numCoresUsed_": 32,
            "numWkSlicesPerDim_": {"x": 4, "mb": 8, "out": 1, "in": 1},
            "primaryDsInfo_": {
                "KERNEL": {
                    "layoutDimOrder_": ["out", "in", "x"],
                    "stickDimOrder_": ["out"],
                }
            },
        },
    }


def test_import_light_classifier_marks_flash_layout_allgather_restickify():
    classification = classify_layout_allgather_restickify_sdsc_triplet(_flash_triplet())

    assert classification is not None
    assert classification["classification"] == LAYOUT_ALLGATHER_RESTICKIFY
    assert classification["kind"] == LAYOUT_ALLGATHER_RESTICKIFY
    assert classification["communication_pattern"] == LAYOUT_ALLGATHER_RESTICKIFY
    assert classification["communication_class"] == COMM_CLASS_ALL_GATHER
    assert classification["dimension_rename"] == {
        "restickify.x": "batchmatmul.out",
        "restickify.out": "batchmatmul.in",
        "restickify.mb": "batchmatmul.x",
    }
    assert classification["requires_staged_realization"]
    assert not classification["realized"]
    assert "backend lowering is not implemented" in classification["unsupported_reason"]


def test_import_light_classifier_rejects_non_lx_producer():
    triplet = _flash_triplet()
    triplet["sdsc_1_mul"]["allocates"] = [
        {"component_": "hbm", "layoutDimOrder_": ["out", "x", "mb"]}
    ]

    assert classify_layout_allgather_restickify_sdsc_triplet(triplet) is None

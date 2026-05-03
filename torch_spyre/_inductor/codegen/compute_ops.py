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


from torch_spyre._C import encode_constant, DataFormats
from sympy import Symbol

from torch_spyre._inductor import config as _spyre_config


def _get_core_id_permutation(num_cores: int, work_slices=None) -> list[int]:
    """Return a permutation of physical core IDs based on config.core_id_permutation.

    Physical core c executes the slice that the unpermuted emitter would
    have assigned to core perm[c]. Identity preserves current behaviour.
    Other permutations exist to test whether sequential ring placement is
    empirically optimal — see tests/diag_core_permutation_probe.py.

    The optional `work_slices` arg enables the `k_fast` perm: a
    generalized "pack K-collaborators contiguously" permutation that
    adapts to whatever split shape the planner picked.
    """
    name = _spyre_config.core_id_permutation
    if name == "identity":
        return list(range(num_cores))
    if name == "reversed":
        return list(range(num_cores - 1, -1, -1))
    if name == "stride2":
        # [0, 2, 4, ..., 30, 1, 3, ..., 31] — interleaves two half-rings
        return list(range(0, num_cores, 2)) + list(range(1, num_cores, 2))
    if name == "block_cyclic":
        # [0, N/2, 1, N/2+1, ...] — adjacent physical cores hop to opposite halves
        half = num_cores // 2
        out: list[int] = []
        for i in range(half):
            out.append(i)
            out.append(half + i)
        return out
    if name == "antipodal":
        # Swap halves: [N/2, N/2+1, ..., N-1, 0, 1, ..., N/2-1]
        half = num_cores // 2
        return list(range(half, num_cores)) + list(range(half))
    if name == "bit_reverse":
        n_bits = (num_cores - 1).bit_length()
        out = []
        for c in range(num_cores):
            r = 0
            for b in range(n_bits):
                if c & (1 << b):
                    r |= 1 << (n_bits - 1 - b)
            out.append(r)
        return out
    if name.startswith("random_"):
        seed = int(name.split("_", 1)[1])
        import random as _random
        rng = _random.Random(seed)
        out = list(range(num_cores))
        rng.shuffle(out)
        return out
    if name == "k_fast":
        # Generalized "pack K-collaborators contiguously" permutation.
        # For matmul iteration_space [M, N, K], the K dim is iterated
        # last in the unpermuted emitter, so logical core_ids that
        # differ by m*n in the unpermuted ordering are K-collaborators.
        # We map them to consecutive physical positions:
        #   perm[c] = (c mod k) * (m*n) + (c // k)
        # Falls back to identity if work_slices is unavailable or k=1.
        if work_slices is None:
            return list(range(num_cores))
        dim_list = list(work_slices.keys())
        if len(dim_list) < 1:
            return list(range(num_cores))
        k = int(work_slices[dim_list[-1]])
        if k <= 1:
            return list(range(num_cores))
        mn = num_cores // k
        return [(c % k) * mn + (c // k) for c in range(num_cores)]
    if name.startswith("ring_pair_d"):
        # Probe-specific permutations for the (1, 16, 2) shape only.
        # Construct a permutation that puts every K-pair (logical i,
        # logical i+16) at exactly d ring positions apart.
        # Valid d: 1, 2, 4, 8, 16. Only valid for num_cores=32.
        d = int(name[len("ring_pair_d"):])
        assert num_cores == 32, "ring_pair_d perms assume 32 cores"
        assert d in (1, 2, 4, 8, 16), f"d must be in {{1,2,4,8,16}}, got {d}"
        # Each "block" of 2d consecutive physical positions holds d pairs
        # at distance d apart: positions {0..d-1, d..2d-1} hold pairs
        # (0,d), (1,d+1), ..., (d-1,2d-1).
        block = 2 * d
        n_blocks = num_cores // block
        out = [0] * num_cores
        pair_idx = 0
        for b in range(n_blocks):
            for i in range(d):
                out[b * block + i] = pair_idx               # k=0 slot
                out[b * block + i + d] = pair_idx + 16      # k=1 slot
                pair_idx += 1
        return out
    raise ValueError(
        f"unknown CORE_ID_PERMUTATION: {name!r}. "
        "Valid: identity, reversed, stride2, block_cyclic, antipodal, "
        "bit_reverse, k_fast, ring_pair_d<1,2,4,8,16>, random_<seed>."
    )


def core_idx_to_slice_offset(
    arg,
    wk_slice: dict,
    work_slices: dict,
) -> int:
    offset = sum(arg.offsets.values())
    for dim, stride in arg.strides.items():
        if str(dim) in wk_slice and arg.scales[dim] > 0:
            offset += wk_slice[str(dim)] * stride // work_slices[dim]
    return offset


def num_bytes(df: DataFormats) -> int:
    """Try to avoid using this method; it is a bad API due to sub-byte datatypes"""
    num_elems = df.elems_per_stick()
    if num_elems > 128:
        raise RuntimeError(f"sub-byte dataformat {df}")
    return 128 // num_elems


def generate_constant_info(data_format, constants, num_cores):
    if len(constants.keys()) == 0:
        return "{}"
    constant_info = {}
    for name, value in constants.items():
        ci = {
            "dataFormat_": data_format.name,
            "name_": name,
            "data_": {
                "dim_prop_func": [{"Const": {}}, {"Const": {}}, {"Map": {}}],
                "dim_prop_attr": [
                    {"factor_": num_cores, "label_": "core"},
                    {"factor_": 1, "label_": "corelet"},
                    {"factor_": 1, "label_": "time"},
                ],
                "data_": {"[0, 0, 0]": [encode_constant(value, data_format)]},
            },
        }
        constant_info[f"{len(constant_info)}"] = ci
    return constant_info


def add_constant(kwargs, name, value) -> int:
    """
    Add a constant to kwargs['op_info']['constants'] and return its index.
    Returns:
        int: The index of the newly added constant (0-based)
    """
    # Ensure structure exists
    if "op_info" not in kwargs:
        kwargs["op_info"] = {}
    if "constants" not in kwargs["op_info"]:
        kwargs["op_info"]["constants"] = {}

    index = len(kwargs["op_info"]["constants"])
    kwargs["op_info"]["constants"][name] = value

    return index


def gen_coord_info_value(
    size: int,
    nsplits: int,
    elems_per_stick: int,
    is_stick_dim: bool,
    is_stick_reduction: bool = False,
):
    return (
        {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 1,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {
                        "Affine": {
                            "alpha_": size,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 1,
                            "beta_": 0,
                        }
                    },
                ],
                "dim_prop_attr": [
                    {
                        "factor_": nsplits,
                        "label_": "core_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "corelet_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "row_fold",
                    },
                    {
                        "factor_": size,
                        "label_": "elem_arr_0",
                    },
                ],
            },
        }
        if not is_stick_dim
        else {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 2,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {
                        "Affine": {
                            "alpha_": elems_per_stick if is_stick_reduction else size,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": elems_per_stick,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0 if is_stick_reduction else 1,
                            "beta_": 0,
                        }
                    },
                ],
                "dim_prop_attr": [
                    {
                        "factor_": nsplits,
                        "label_": "core_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "corelet_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "row_fold",
                    },
                    {
                        "factor_": 1
                        if is_stick_reduction
                        else (size // elems_per_stick),
                        "label_": "elem_arr_1",
                    },
                    {
                        "factor_": elems_per_stick,
                        "label_": "elem_arr_0",
                    },
                ],
            },
        }
    )


def generate_sdsc(sdsc_spec):
    out_idx = len(sdsc_spec.args) - 1
    perm = _get_core_id_permutation(sdsc_spec.num_cores, sdsc_spec.work_slices)
    core_id_to_wk_slice = {
        str(c): {
            str(dim): int(expr.subs({Symbol("core_id"): perm[c]}))
            for dim, expr in sdsc_spec.core_id_to_work_slice.items()
        }
        for c in range(sdsc_spec.num_cores)
    }
    return {
        sdsc_spec.opfunc: {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": sdsc_spec.num_cores, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": sdsc_spec.num_cores,
            "coreIdToDsc_": {str(c): 0 for c in range(sdsc_spec.num_cores)},
            "numWkSlicesPerDim_": {
                str(dim): num_wk_slices
                for dim, num_wk_slices in sdsc_spec.work_slices.items()
            },
            "coreIdToWkSlice_": core_id_to_wk_slice,
            "coreIdToDscSchedule": {
                str(c): [[-1, 0, 0, 0]] for c in range(sdsc_spec.num_cores)
            },
            "dscs_": [
                {
                    sdsc_spec.opfunc: {
                        "numCoresUsed_": sdsc_spec.num_cores,
                        "numCoreletsUsed_": 1,
                        "coreIdsUsed_": [c for c in range(sdsc_spec.num_cores)],
                        "N_": {
                            "name_": "n",
                            **{
                                str(dim) + "_": size
                                for dim, size in sdsc_spec.iteration_space.items()
                            },
                        },
                        "coordinateMasking_": {
                            str(dim): mask_range
                            for dim, mask_range in sdsc_spec.coordinate_masking.items()
                        },
                        "maskingConstId_": 0 if sdsc_spec.coordinate_masking else -1,
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    **{
                                        str(dim) + "_": size
                                        // sdsc_spec.work_slices[dim]
                                        for dim, size in sdsc_spec.iteration_space.items()
                                    },
                                },
                                "el_": {
                                    "name_": "core",
                                    **{
                                        str(dim) + "_": size
                                        // sdsc_spec.work_slices[dim]
                                        for dim, size in sdsc_spec.iteration_space.items()
                                    },
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            label: {
                                "layoutDimOrder_": [
                                    str(dim) for dim in layout_info["dim_order"]
                                ],
                                "stickDimOrder_": [str(layout_info["stick_dim_order"])],
                                "stickSize_": [layout_info["stick_size"]],
                            }
                            for label, layout_info in sdsc_spec.layouts.items()
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": f"allocate-Tensor{i}_{'hbm' if not tensor.allocation else 'lx'}",
                                "prev_": "",
                                "ldsIdx_": i,
                                "component_": "hbm" if not tensor.allocation else "lx",
                                "layoutDimOrder_": [
                                    str(dim)
                                    for dim in sdsc_spec.layouts[tensor.layout][
                                        "dim_order"
                                    ]
                                ],
                                "maxDimSizes_": [
                                    tensor.max_dim_sizes[dim]
                                    for dim in sdsc_spec.layouts[tensor.layout][
                                        "dim_order"
                                    ]
                                ],
                                "startAddressCoreCorelet_": {
                                    "dim_prop_func": [
                                        {"Map": {}},
                                        {"Const": {}},
                                        {"Const": {}},
                                    ],
                                    "dim_prop_attr": [
                                        {
                                            "factor_": sdsc_spec.num_cores,
                                            "label_": "core",
                                        },
                                        {"factor_": 1, "label_": "corelet"},
                                        {"factor_": 1, "label_": "time"},
                                    ],
                                    "data_": {
                                        f"[{c}, 0, 0]": str(
                                            tensor.start_address
                                            + core_idx_to_slice_offset(
                                                tensor,
                                                core_id_to_wk_slice[str(c)],
                                                sdsc_spec.work_slices,
                                            )
                                            * num_bytes(tensor.data_format)
                                        )
                                        for c in range(sdsc_spec.num_cores)
                                        #  lx addr is baked into tensor.start_addr already
                                    },
                                },
                                **(
                                    {
                                        "backGapCore_": {
                                            str(dim): {
                                                "-1": str(gap)  # HBM is -1
                                            }
                                            for dim, gap in tensor.backGap.items()
                                        }
                                    }
                                    if tensor.backGap
                                    else {}
                                ),
                                "coordinates_": {
                                    "coordInfo": {
                                        str(dim): gen_coord_info_value(
                                            size=sdsc_spec.iteration_space[dim]
                                            // sdsc_spec.work_slices[dim]
                                            if (tensor.scales[dim] == 1)
                                            else 1,
                                            nsplits=sdsc_spec.work_slices[dim]
                                            if (tensor.scales[dim] == 1)
                                            else 1,
                                            elems_per_stick=tensor.data_format.elems_per_stick(),
                                            is_stick_dim=(
                                                sdsc_spec.layouts[tensor.layout][
                                                    "stick_dim_order"
                                                ].has(dim)
                                            ),
                                            is_stick_reduction=(
                                                tensor.scales[dim] == -2
                                            ),
                                        )
                                        for dim in sdsc_spec.layouts[tensor.layout][
                                            "dim_order"
                                        ]
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            }
                            for i, tensor in enumerate(sdsc_spec.args)
                        ],
                        "labeledDs_": [
                            {
                                "ldsIdx_": i,
                                "dsName_": f"Tensor{i}",
                                "dsType_": tensor.layout,
                                "scale_": [
                                    tensor.scales[dim]
                                    for dim in sdsc_spec.layouts[tensor.layout][
                                        "dim_order"
                                    ]
                                ],
                                "wordLength": num_bytes(tensor.data_format),
                                "dataFormat_": tensor.data_format.name,
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                }
                                if not tensor.allocation
                                else {"lx": {"isPresent": 1}},
                            }
                            for i, tensor in enumerate(sdsc_spec.args)
                        ],
                        "constantInfo_": generate_constant_info(
                            sdsc_spec.data_format,
                            sdsc_spec.constants,
                            sdsc_spec.num_cores,
                        ),
                        "computeOp_": [
                            {
                                "exUnit": sdsc_spec.execution_unit,
                                "opFuncName": sdsc_spec.opfunc,
                                "attributes_": {
                                    "dataFormat_": sdsc_spec.data_format.name,
                                    "fidelity_": "regular",
                                },
                                "location": "Inner",
                                "inputLabeledDs": [
                                    f"Tensor{i}-idx{i}"
                                    for i in range(sdsc_spec.num_inputs)
                                ],
                                "outputLabeledDs": [f"Tensor{out_idx}-idx{out_idx}"],
                            }
                        ],
                    }
                }
            ],
        }
    }

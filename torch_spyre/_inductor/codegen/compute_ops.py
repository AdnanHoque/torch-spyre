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


def _k_fast_core_id_permutation(num_cores: int, work_slices) -> list[int]:
    """Permute physical core IDs so K-collaborators sit on adjacent ring cores.

    perm[c] = (c % k) * (m * n) + (c // k), where (m, n, k) is the planner's
    matmul split. Physical core c executes the slice that the unpermuted
    emitter would have given to logical core perm[c]. Degenerates to
    identity when k=1 or when the feature flag is off.
    """
    if not _spyre_config.core_id_k_fast_emission:
        return list(range(num_cores))
    if not work_slices:
        return list(range(num_cores))
    dim_list = list(work_slices.keys())
    # For matmul iteration_space [M, N, K], the last dim in work_slices
    # is the K (reduction) dim by convention. For non-matmul ops with
    # k = 1 the formula collapses to identity, so this is a no-op for
    # those cases.
    k = int(work_slices[dim_list[-1]])
    if k <= 1:
        return list(range(num_cores))
    mn = num_cores // k
    return [(c % k) * mn + (c // k) for c in range(num_cores)]


def _m_fast_core_id_permutation(num_cores: int, work_slices) -> list[int]:
    """Permute physical core IDs so M-collaborators sit on adjacent ring cores.

    perm[c] = (c % m) * (n * k) + (c // m), where (m, n, k) is the matmul
    split. M-collaborators are cores that vary in M but share the same
    (i_n, i_k) — i.e., they share the same B-fragment. Placing them
    adjacent on the RIU ring captures B-multicast.

    Caller is responsible for the gating decision (see
    `_should_use_m_fast`). This function only emits the permutation.
    """
    if not work_slices:
        return list(range(num_cores))
    dim_list = list(work_slices.keys())
    m = int(work_slices[dim_list[0]])
    if m <= 1:
        return list(range(num_cores))
    nk = num_cores // m
    return [(c % m) * nk + (c // m) for c in range(num_cores)]


def _should_use_m_fast(work_slices, iteration_space) -> bool:
    """Decide whether m_fast permutation should fire for this op.

    Empirically validated rule (broad sweep in tests/diag_mfast_sweep_probe.py):
    m_fast wins reliably in a narrow regime. Outside it, identity is
    equal or better. The gate avoids catastrophic regressions on
    (m=2, n=16) splits.

    Fires when ALL hold:
      - m_fast feature flag is on
      - 3-dim iteration space (matmul)
      - k = 1 (k_fast handles k>1 cases)
      - m ≥ 2 AND n ≥ 2 (mixed-MN split)
      - NOT (n ≥ 16 AND m ≤ 2)         — catastrophic regression case
      - M_per ∈ [8, 64]                — PT-saturated sweet spot
      - B-side > 1.5 × A-side          — clear B-multicast win
    """
    if not _spyre_config.core_id_m_fast_emission:
        return False
    if not work_slices or not iteration_space:
        return False
    dim_list = list(work_slices.keys())
    if len(dim_list) != 3:
        return False

    m = int(work_slices[dim_list[0]])
    n = int(work_slices[dim_list[1]])
    k = int(work_slices[dim_list[2]])
    if k > 1 or m < 2 or n < 2:
        return False
    if n >= 16 and m <= 2:
        return False

    M = int(iteration_space[dim_list[0]])
    N = int(iteration_space[dim_list[1]])
    K = int(iteration_space[dim_list[2]])
    M_per = M // m
    if M_per < 8 or M_per > 64:
        return False

    b_side = m * K * N / n
    a_side = n * M * K / m
    if b_side <= 1.5 * a_side:
        return False
    return True


def _select_core_id_permutation(
    num_cores: int, work_slices, iteration_space=None
) -> list[int]:
    """Pick the right core-ID permutation for the given op.

    Priority:
      1. k_fast for k>1 (PSUM ring adjacency) — verified optimal
      2. m_fast for k=1 mixed-MN in the empirical sweet spot — wins
         up to ~45% on M=128 decoder shapes
      3. identity — falls back; also captures A-multicast on N-cohort
         for k=1 splits because logical encoding naturally adjacents N
    """
    if not work_slices:
        return list(range(num_cores))
    dim_list = list(work_slices.keys())
    if len(dim_list) == 3:
        k = int(work_slices[dim_list[-1]])
        if k <= 1 and _should_use_m_fast(work_slices, iteration_space):
            return _m_fast_core_id_permutation(num_cores, work_slices)
    return _k_fast_core_id_permutation(num_cores, work_slices)


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


def generate_sdsc(idx, sdsc_spec):
    out_idx = len(sdsc_spec.args) - 1
    perm = _select_core_id_permutation(
        sdsc_spec.num_cores,
        sdsc_spec.work_slices,
        getattr(sdsc_spec, "iteration_space", None),
    )
    core_id_to_wk_slice = {
        str(c): {
            str(dim): int(expr.subs({Symbol("core_id"): perm[c]}))
            for dim, expr in sdsc_spec.core_id_to_work_slice.items()
        }
        for c in range(sdsc_spec.num_cores)
    }
    return {
        f"{idx}_{sdsc_spec.opfunc}": {
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
                                        if not tensor.allocation
                                        else str(tensor.start_address)
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

import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_HELPER = (
    _ROOT
    / "torch_spyre"
    / "_inductor"
    / "codegen"
    / "causal_mask_dataop.py"
)
_CONSTANTS = _ROOT / "torch_spyre" / "_inductor" / "constants.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "_test_causal_mask_dataop", _HELPER
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_constants():
    spec = importlib.util.spec_from_file_location(
        "_test_spyre_inductor_constants", _CONSTANTS
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _contract(**overrides):
    contract = {
        "opfunc": "causal_score_bias_like",
        "input_count": 1,
        "output_count": 1,
        "constants": ["keyStart"],
        "iteration_sizes": {"name_": "n", "mb_": 2, "x_": 4, "out_": 64},
        "work_slices": {"mb": 2, "x": 4, "out": 1},
        "output_layout": {
            "labeled_ds": "Tensor1",
            "layout_dim_order": ["x", "mb", "out"],
            "stick_dim_order": ["out"],
            "stick_size": [64],
        },
        "inferred_query_dim": "x",
        "inferred_key_dim": "out",
        "supported_score_layout": True,
    }
    contract.update(overrides)
    return contract


def test_causal_idx_to_mask_candidate_records_backend_metadata():
    helper = _load_helper()

    candidate = helper.build_causal_idx_to_mask_candidate(
        _contract(),
        key_start=2,
    )

    assert candidate["strategy"] == "idx_to_mask_plus_where3"
    assert candidate["feasible"] is True
    assert candidate["rejection_reasons"] == []
    assert candidate["runtime_emission"] == {
        "torch_spyre_descriptor_only": True,
        "datadsc_json_accepts_idx_to_mask": False,
        "requires_deeptools_dataop_parser_extension": True,
        "blocking_reason": (
            "DeepTools DataOpDsc does not currently accept "
            "op.name=IdxToMask from imported SuperDSC datadscs_ JSON"
        ),
    }
    assert candidate["layout"]["query_length"] == 4
    assert candidate["layout"]["key_length"] == 64
    assert candidate["layout"]["score_layout_sizes"] == {
        "x": 4,
        "mb": 2,
        "out": 64,
    }
    assert candidate["layout"]["mask_layout_sizes"] == {
        "x": 4,
        "mb": 1,
        "out": 64,
    }
    assert candidate["layout"]["broadcast_dims"] == ["mb"]

    idx_to_mask = candidate["idx_to_mask"]
    assert idx_to_mask["isIdxToMaskSdc"] is True
    assert idx_to_mask["idxToMaskDim"] == "out"
    assert idx_to_mask["idxToMaskDimIdx"] == 2
    assert idx_to_mask["idxToMaskValidElementOffset"] == -2
    assert idx_to_mask["causalMask"] is True
    assert idx_to_mask["invertedMask"] is False
    assert idx_to_mask["reversedMask"] is False
    assert idx_to_mask["input"] == {
        "kind": "length_one_query_length_vector",
        "shape": [1],
        "value": 4,
        "dtype": "IEEE_INT64",
    }
    assert idx_to_mask["output"]["dtype"] == "SEN169_FP16"
    assert idx_to_mask["output"]["layout_sizes"] == {
        "x": 4,
        "mb": 1,
        "out": 64,
    }

    assert candidate["dci"]["dcOpName_"] == "IDX_TO_MASK"
    assert candidate["dci"]["input_shape_"] == [1]
    assert candidate["dci"]["output_shape_"] == [64, 4, 1, 1]
    assert candidate["dci"]["imi_"] == {
        "idxToMaskValidElementOffset_": -2,
        "maskInnerRepeat_": 1,
        "invertMask_": False,
        "reverseMask_": False,
        "isCausalMask_": True,
        "causalDimLength_": 4,
        "continuousMaskElems_": 64,
        "strideAfterContinuous_": 256,
    }
    assert candidate["dataop_json_extension"]["op"] == {
        "name": "IdxToMask",
        "idxToMaskDimIdx": 2,
        "idxToMaskValidElementOffset": -2,
        "invertedMask": 0,
        "reversedMask": 0,
        "causalMask": 1,
    }

    assert candidate["where3"] == {
        "opFuncName": "where3",
        "predicate": "idx_to_mask.output",
        "true_value": 0.0,
        "false_value": "-inf",
        "output": "causal_score_bias_like.output",
        "broadcast_predicate_dims": ["mb"],
    }


def test_causal_idx_to_mask_emission_plan_materializes_dataop_shape():
    helper = _load_helper()

    plan = helper.build_causal_idx_to_mask_emission_plan(
        _contract(
            core_id_to_work_slice={
                "0": {"mb": 0, "x": 0, "out": 0},
                "1": {"mb": 1, "x": 0, "out": 0},
                "2": {"mb": 0, "x": 1, "out": 0},
                "3": {"mb": 1, "x": 1, "out": 0},
                "4": {"mb": 0, "x": 2, "out": 0},
                "5": {"mb": 1, "x": 2, "out": 0},
                "6": {"mb": 0, "x": 3, "out": 0},
                "7": {"mb": 1, "x": 3, "out": 0},
            },
            num_cores=8,
        ),
        key_start=2,
    )

    body = plan["causal_idx_to_mask_where3_candidate"]
    assert body["causalIdxToMaskPlan_"]["runtime_status"] == "not_emitted"
    assert body["coreIdToDscSchedule"]["0"] == [[0, -1, 0, 1], [-1, 0, 1, 0]]
    assert body["coreIdToDscSchedule"]["7"] == [[0, -1, 0, 1], [-1, 0, 1, 0]]

    dataop = body["datadscs_"][0]["0_IdxToMask_dataop"]
    assert dataop["coreIdsUsed_"] == list(range(8))
    assert dataop["dimPool_"] == ["x", "mb", "out"]
    assert dataop["primaryDs_"] == [
        {"name_": "maskIndex", "dimNames": []},
        {"name_": "maskOut", "dimNames": ["x", "mb", "out"]},
    ]
    assert dataop["op"] == {
        "name": "IdxToMask",
        "idxToMaskDimIdx": 2,
        "idxToMaskValidElementOffset": -2,
        "invertedMask": 0,
        "reversedMask": 0,
        "causalMask": 1,
    }

    mask_input, mask_output = dataop["labeledDs_"]
    assert mask_input["dataformat"] == "IEEE_INT64"
    assert mask_input["wordLength"] == 8
    assert mask_input["layoutDimOrder_"] == []
    assert len(mask_input["PieceInfo"]) == 8
    assert mask_output["dataformat"] == "SEN169_FP16"
    assert mask_output["wordLength"] == 2
    assert mask_output["layoutDimOrder_"] == ["x", "mb", "out"]
    assert mask_output["stickDimOrder_"] == ["out"]
    assert mask_output["dimToLayoutSize_"] == {"x": 4, "mb": 1, "out": 64}
    assert mask_output["dimToStickSize_"] == {"out": 64}

    first_piece = mask_output["PieceInfo"][0]
    duplicate_mb_piece = mask_output["PieceInfo"][1]
    last_piece = mask_output["PieceInfo"][-1]
    assert first_piece["dimToStartCordinate"] == {"x": 0, "mb": 0, "out": 0}
    assert first_piece["dimToSize_"] == {"x": 1, "mb": 1, "out": 64}
    assert duplicate_mb_piece["dimToStartCordinate"] == {
        "x": 0,
        "mb": 0,
        "out": 0,
    }
    assert duplicate_mb_piece["PlacementInfo"][0]["memId"] == [1]
    assert last_piece["dimToStartCordinate"] == {"x": 3, "mb": 0, "out": 0}

    where3 = body["where3_compute_fragment"]
    assert where3["computeOp_"][0]["opFuncName"] == "where3"
    assert where3["computeOp_"][0]["inputLabeledDs"] == [
        "maskOut-idx0",
        "zeroBias-idx1",
        "negInfBias-idx2",
    ]
    assert where3["predicate_broadcast_dims"] == ["mb"]
    assert where3["required_tensor_inputs"] == {
        "zeroBias": 0.0,
        "negInfBias": "-inf",
    }


def test_causal_idx_to_mask_candidate_rejects_split_key_stick_dim():
    helper = _load_helper()

    candidate = helper.build_causal_idx_to_mask_candidate(
        _contract(work_slices={"mb": 1, "x": 2, "out": 2}),
        key_start=0,
    )

    assert candidate["feasible"] is False
    assert "key stick dimension must remain unsplit" in candidate["rejection_reasons"]


def test_causal_idx_to_mask_candidate_rejects_missing_key_start_constant():
    helper = _load_helper()

    candidate = helper.build_causal_idx_to_mask_candidate(
        _contract(constants=[]),
        key_start=1,
    )

    assert candidate["feasible"] is False
    assert "missing keyStart constant" in candidate["rejection_reasons"]


def test_fp32_allowlist_includes_emitted_where3_opfunc():
    constants = _load_constants()

    assert "where3" in constants.SPYRE_FP32_OPS


def _run_all():
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    fails = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            fails.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - len(fails)}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())

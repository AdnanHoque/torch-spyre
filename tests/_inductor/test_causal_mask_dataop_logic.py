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

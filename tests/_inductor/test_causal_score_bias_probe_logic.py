import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "tools" / "causal_score_bias_backend_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "_test_causal_score_bias_backend_probe", _PROBE
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metadata_records_causal_score_bias_layout_contract():
    probe = _load_probe()
    payload = {
        "0_causal_score_bias_like": {
            "numWkSlicesPerDim_": {"mb": 2, "x": 4, "out": 1},
            "coreIdToWkSlice_": {
                "0": {"mb": 0, "x": 0, "out": 0},
                "7": {"mb": 1, "x": 3, "out": 0},
            },
            "dscs_": [
                {
                    "causal_score_bias_like": {
                        "numCoresUsed_": 8,
                        "N_": {"name_": "n", "mb_": 2, "x_": 4, "out_": 64},
                        "primaryDsInfo_": {
                            "OUTPUT": {
                                "layoutDimOrder_": ["x", "mb", "out"],
                                "stickDimOrder_": ["out"],
                                "stickSize_": [64],
                            }
                        },
                        "labeledDs_": [
                            {
                                "ldsIdx_": 0,
                                "dsName_": "Tensor0",
                                "dsType_": "OUTPUT",
                            },
                            {
                                "ldsIdx_": 1,
                                "dsName_": "Tensor1",
                                "dsType_": "OUTPUT",
                            },
                        ],
                        "constantInfo_": {
                            "0": {
                                "dataFormat_": "SEN169_FP16",
                                "name_": "keyStart",
                            }
                        },
                        "computeOp_": [
                            {
                                "opFuncName": "causal_score_bias_like",
                                "inputLabeledDs": ["Tensor0-idx0"],
                                "outputLabeledDs": ["Tensor1-idx1"],
                            }
                        ],
                    }
                }
            ]
        }
    }

    metadata = probe._metadata_from_sdsc_payload(
        Path("sdsc.json"),
        payload,
        key_start=2,
    )

    assert metadata["opfuncs"] == ["causal_score_bias_like"]
    assert metadata["constants"] == ["keyStart"]
    contract = metadata["causal_score_bias_contract"]
    assert contract["input_count"] == 1
    assert contract["output_count"] == 1
    assert contract["num_cores"] == 8
    assert contract["split_dims"] == {"mb": 2, "x": 4}
    assert contract["inferred_query_dim"] == "x"
    assert contract["inferred_key_dim"] == "out"
    assert contract["supported_score_layout"] is True
    candidate = metadata["causal_idx_to_mask_candidate"]
    assert candidate["feasible"] is True
    assert candidate["runtime_emission"]["datadsc_json_accepts_idx_to_mask"] is False
    assert candidate["layout"]["mask_layout_sizes"] == {"x": 4, "mb": 1, "out": 64}
    assert candidate["layout"]["broadcast_dims"] == ["mb"]
    assert candidate["idx_to_mask"]["idxToMaskDimIdx"] == 2
    assert candidate["idx_to_mask"]["idxToMaskValidElementOffset"] == -2
    assert candidate["dci"]["output_shape_"] == [64, 4, 1, 1]
    assert candidate["where3"]["opFuncName"] == "where3"

    plans = probe._candidate_emission_plans([metadata], key_start=2)
    body = plans[0]["causal_idx_to_mask_where3_candidate"]
    dataop = body["datadscs_"][0]["0_IdxToMask_dataop"]
    assert dataop["op"]["name"] == "IdxToMask"
    assert dataop["op"]["idxToMaskValidElementOffset"] == -2
    assert body["where3_compute_fragment"]["computeOp_"][0]["opFuncName"] == "where3"


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

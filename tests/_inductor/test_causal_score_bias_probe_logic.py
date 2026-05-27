import inspect
import importlib.util
import json
import tempfile
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


def _payload():
    return {
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


def _parser_probe_bundle_writer(probe):
    for name in (
        "_write_causal_idx_to_mask_parser_probe_bundle",
        "write_causal_idx_to_mask_parser_probe_bundle",
        "_emit_candidate_parser_probe_bundle",
        "_write_parser_probe_bundle",
        "write_parser_probe_bundle",
    ):
        writer = getattr(probe, name, None)
        if callable(writer):
            return writer
    return None


def _invoke_parser_probe_bundle_writer(writer, tmp_path, payload):
    signature = inspect.signature(writer)
    cache_dir = tmp_path / "cache"
    bundle_dir = tmp_path / "bundle"
    if "cache_dir" in signature.parameters:
        cache_dir.mkdir()
        (cache_dir / "sdsc_0_causal_score_bias_like.json").write_text(
            json.dumps(payload)
        )
    args = []
    kwargs = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name == "cache_dir":
            value = cache_dir
        elif name in ("output_dir", "out_dir", "bundle_dir", "directory"):
            value = bundle_dir
        elif name in ("payload", "sdsc_payload", "sdsc_json"):
            value = payload
        elif name == "key_start":
            value = 2
        elif name in ("name", "probe_name"):
            value = "causal_idx_to_mask_parser_probe"
        elif name == "run":
            value = False
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            raise AssertionError(f"unsupported writer parameter: {name}")

        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = value
        else:
            args.append(value)
    return writer(*args, **kwargs), bundle_dir


def _assert_parser_probe_body(body):
    dataop = body["datadscs_"][0]["0_IdxToMask_dataop"]
    assert dataop["op"]["name"] == "IdxToMask"
    assert body["coreIdToDscSchedule"]["0"] == [[0, -1, 0, 1], [-1, 0, 1, 0]]
    assert body["coreIdToDscSchedule"]["7"] == [[0, -1, 0, 1], [-1, 0, 1, 0]]
    assert body["opFuncsUsed_"] == []

    dsc_name, dsc = next(iter(body["dscs_"][0].items()))
    assert dsc_name == "identity"
    compute_op = dsc["computeOp_"][0]
    assert compute_op["opFuncName"] == "identity"
    assert compute_op["opFuncName"] != "causal_score_bias_like"


def test_metadata_records_causal_score_bias_layout_contract():
    probe = _load_probe()
    payload = _payload()

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


def test_parser_probe_bundle_writer_writes_bundle_without_running_dxp(tmp_path):
    probe = _load_probe()
    writer = _parser_probe_bundle_writer(probe)
    assert writer is not None

    result, bundle_dir = _invoke_parser_probe_bundle_writer(
        writer,
        tmp_path,
        _payload(),
    )
    if isinstance(result, dict) and "status" in result:
        assert result["status"] == "emitted"

    sdsc_paths = sorted(bundle_dir.glob("sdsc_*.json"))
    assert len(sdsc_paths) == 1
    payload = json.loads(sdsc_paths[0].read_text())
    body = payload["causal_idx_to_mask_parser_probe"]
    _assert_parser_probe_body(body)

    bundle_mlir = (bundle_dir / "bundle.mlir").read_text()
    assert f"sdsc_filename=\"{sdsc_paths[0].name}\"" in bundle_mlir


def _run_all():
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    fails = []
    for name, fn in tests:
        try:
            parameters = inspect.signature(fn).parameters
            if "tmp_path" in parameters:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    fn(Path(tmp_dir))
            else:
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

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

from __future__ import annotations

import argparse
import io
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock


_HERE = Path(__file__).resolve().parent
_SWEEP = _HERE.parents[1] / "tools" / "onchip_sdpa_sweep.py"
_LAYOUT_PAIR_ENV = "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE"
_IFN_PREFIX_FORCE_ENV = "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PREFIX_FORCE"
_LAYOUT_PAIR_OVERLAP_ENV = (
    "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_OVERLAP"
)
_LAYOUT_LOOKAHEAD_ENV = (
    "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_LOOKAHEAD_TILE"
)
_LAYOUT_HOIST_ENV = (
    "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_HOIST_TILE"
)
_KV_REPACK_PLAN_ENV = "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PLAN_ARTIFACT"
_KV_REPACK_PAIR_ENV = "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_TILE"
_KV_REPACK_PAIR_IFN_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_IFN_TRANSFER"
)
_KV_REPACK_PAIR_REUSE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_SUBPIECE_REUSE"
)
_KV_REPACK_PAIR_GROUP_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_GROUP_SIZE"
)
_KV_REPACK_PAIR_SELF_RESIDENT_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_SELF_RESIDENT_SOURCE"
)
_KV_REPACK_PAIR_HBM_SOURCE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_HBM_SOURCE"
)
_KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_HBM_DIRECT_LOAD"
)
_KV_REPACK_PAIR_HBM_STAGED_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_HBM_STAGED"
)
_KV_REPACK_PAIR_CONSUMER_CSI_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_CONSUMER_CORE_STATE_INIT"
)
_KV_REPACK_PAIR_CONSUMER_DS_TYPE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_CONSUMER_DS_TYPE"
)
_KV_REPACK_PAIR_CONSUMER_LX_ALLOC_STYLE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_CONSUMER_LX_ALLOC_STYLE"
)
_KV_REPACK_PAIR_USE_UNICAST_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_USE_UNICAST"
)
_KV_REPACK_PAIR_FORCE_MC_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_FORCE_MC_MODE"
)
_KV_REPACK_HBM_STAGED_HOIST_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_STAGED_HOIST_TILE"
)
_KV_REPACK_HBM_PREFETCH_HOIST_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_HOIST_TILE"
)
_KV_REPACK_HBM_PREFETCH_LX_BASE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LX_BASE"
)
_KV_REPACK_HBM_PREFETCH_SERIAL_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SERIAL"
)
_KV_REPACK_HBM_PREFETCH_REDUNDANT_FUTURE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_REDUNDANT_FUTURE"
)
_KV_REPACK_HBM_PREFETCH_SERIALIZE_CURRENT_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SERIALIZE_CURRENT"
)
_KV_REPACK_HBM_PREFETCH_EXTERNAL_FUTURE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_EXTERNAL_FUTURE"
)
_KV_REPACK_HBM_PREFETCH_OVERLAP_AFTER_SYNC_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_OVERLAP_AFTER_SYNC"
)
_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT"
)
_KV_REPACK_HBM_PREFETCH_SOURCE_FANOUT_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SOURCE_FANOUT"
)
_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT"
)
_KV_REPACK_HBM_PREFETCH_LOADER_CORE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_CORE"
)
_KV_REPACK_HBM_PREFETCH_LOADER_LX_BASE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_LX_BASE"
)
_KV_REPACK_HBM_PREFETCH_FANOUT_USE_UNICAST_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_USE_UNICAST"
)
_KV_REPACK_HBM_PREFETCH_FANOUT_USE_LXSFP_LX_TRANSFERS_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_USE_LXSFP_LX_TRANSFERS"
)
_KV_REPACK_HBM_PREFETCH_FANOUT_COPYBACK_CORE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_COPYBACK_CORE"
)
_KV_REPACK_HBM_PREFETCH_FANOUT_RESTRICT_TO_COPYBACK_CORE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_FANOUT_RESTRICT_TO_COPYBACK_CORE"
)
_KV_REPACK_HBM_PREFETCH_LOADER_COPYBACK_WITHOUT_FANOUT_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_COPYBACK_WITHOUT_FANOUT"
)
_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES"
)
_KV_REPACK_HBM_PREFETCH_SERIALIZE_LOADER_CORE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SERIALIZE_LOADER_CORE"
)
_KV_REPACK_HBM_PREFETCH_LX_ROUNDTRIP_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LX_ROUNDTRIP"
)
_KV_REPACK_HBM_PREFETCH_CORELET1_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_CORELET1"
)
_KV_REPACK_COPYBACK_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_TILE"
)
_KV_REPACK_COPYBACK_CORE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_CORE"
)
_KV_REPACK_COPYBACK_DIRECT_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_DIRECT_SOURCE"
)
_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_HBM_ROUNDTRIP"
)
_KV_REPACK_COPYBACK_HBM_SOURCE_FANOUT_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_HBM_SOURCE_FANOUT"
)
_KV_REPACK_COPYBACK_HBM_DIRECT_LOAD_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_HBM_DIRECT_LOAD"
)
_KV_REPACK_COPYBACK_HBM_LOAD_ONLY_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_HBM_ROUNDTRIP_LOAD_ONLY"
)
_KV_REPACK_COPYBACK_HBM_BARRIER_ONLY_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_HBM_ROUNDTRIP_BARRIER_ONLY"
)
_KV_REPACK_COPYBACK_DATA_ONLY_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_DATA_ONLY"
)
_KV_REPACK_COPYBACK_REPLACE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_REPLACE_CONSUMER"
)
_KV_REPACK_COPYBACK_COMPUTE_ONLY_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_COMPUTE_ONLY"
)
_KV_REPACK_COPYBACK_EXACT_CLONE_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_EXACT_CLONE"
)
_KV_REPACK_COPYBACK_PRESERVE_NAME_ENV = (
    "SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_PRESERVE_CONSUMER_NAME"
)


def _load_sweep():
    spec = importlib.util.spec_from_file_location("_test_onchip_sdpa_sweep", _SWEEP)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sweep = _load_sweep()


def test_sweep_script_bootstraps_repo_root_on_import_path():
    repo_root = str(_SWEEP.parents[1])

    assert repo_root in sys.path


def test_layout_xform_candidate_summary_reports_bounded_rejections():
    fake_spyre = types.ModuleType("torch_spyre")
    fake_inductor = types.ModuleType("torch_spyre._inductor")
    fake_rz = types.ModuleType("torch_spyre._inductor.onchip_realize")
    fake_rz.LAYOUT_XFORM_PAIR_AUTO_TILE = -2
    calls = []

    def fake_lookahead(sdscs, tile):
        calls.append(("lookahead", tile, [next(iter(item)) for item in sdscs]))
        return [f"lookahead_reason_{idx}" for idx in range(15)]

    def fake_hoist(sdscs, tile):
        calls.append(("hoist", tile, [next(iter(item)) for item in sdscs]))
        return []

    def fake_pair(sdscs, tile):
        calls.append(("pair", tile, [next(iter(item)) for item in sdscs]))
        return ["pair_reason"]

    fake_rz.flash_attention_layout_xform_lookahead_rejection_reasons = (
        fake_lookahead
    )
    fake_rz.flash_attention_layout_xform_hoist_rejection_reasons = fake_hoist
    fake_rz.flash_attention_layout_xform_pair_rejection_reasons = fake_pair
    fake_inductor.onchip_realize = fake_rz

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        bundle_dir = cache_dir / "inductor-spyre" / "bundle_a"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "sdsc_10_second.json").write_text(
            json.dumps({"10_second": {}})
        )
        (bundle_dir / "sdsc_2_first.json").write_text(
            json.dumps({"2_first": {}})
        )
        (bundle_dir / "sdsc_mixed_flash_ignore.json").write_text(
            json.dumps({"mixed_flash_ignore": {"opFuncsUsed_": ["STCDPOpLx"]}})
        )

        with mock.patch.dict(
            sys.modules,
            {
                "torch_spyre": fake_spyre,
                "torch_spyre._inductor": fake_inductor,
                "torch_spyre._inductor.onchip_realize": fake_rz,
            },
        ):
            summary = sweep._layout_xform_candidate_summary(cache_dir)

    assert summary == [
        {
            "dir": "inductor-spyre/bundle_a",
            "sdscs": 2,
            "lookahead_selectable": False,
            "lookahead_rejections": {
                "count": 15,
                "first": [f"lookahead_reason_{idx}" for idx in range(12)],
                "truncated": True,
            },
            "hoist_selectable": True,
            "hoist_rejections": {"count": 0, "first": [], "truncated": False},
            "pair_selectable": False,
            "pair_rejections": {
                "count": 1,
                "first": ["pair_reason"],
                "truncated": False,
            },
        }
    ]
    assert calls == [
        ("lookahead", -2, ["2_first", "10_second"]),
        ("hoist", -2, ["2_first", "10_second"]),
        ("pair", -2, ["2_first", "10_second"]),
    ]


def _args():
    return argparse.Namespace(
        batch=1,
        heads=2,
        dim=64,
        block_size=64,
        lengths="128",
        variants="onchip_master_layout_xform",
        warmup=1,
        iters=2,
        seed=0xA771,
        atol=0.1,
        rtol=0.1,
        timeout_s=480.0,
        cache_prefix="/tmp/sdpa-sweep-test",
        output_json="",
        dxp_debug=True,
        is_causal=False,
        forbid_fallbacks=False,
        env=[],
    )


def test_child_explicitly_autoloads_spyre_before_fallbacks_or_devices():
    args = _args()
    args.variant = "onchip_master_layout_xform"
    args.length = 128
    args.warmup = 0
    args.iters = 1
    events = []
    state = {"autoloaded": False}

    class FakeTensor:
        def __init__(self, device):
            self.device = types.SimpleNamespace(type=device)

        def to(self, device=None, dtype=None):
            if device == "spyre":
                events.append(("to_spyre", state["autoloaded"]))
                assert state["autoloaded"]
            if device is None:
                device = self.device.type
            return FakeTensor(str(device).split(":")[0])

        def cpu(self):
            return FakeTensor("cpu")

        def __sub__(self, other):
            return self

        def abs(self):
            return self

        def max(self):
            return self

        def item(self):
            return 0.0

    fake_torch = types.ModuleType("torch")
    fake_torch.__path__ = []
    fake_torch.float16 = object()
    fake_torch.manual_seed = lambda seed: None
    fake_torch.randn = lambda shape, dtype: FakeTensor("cpu")
    fake_torch.compile = lambda fn, backend: fn
    fake_torch.testing = types.SimpleNamespace(assert_close=lambda *args, **kwargs: None)
    fake_torch._dynamo = types.SimpleNamespace(reset_code_caches=lambda: None)
    fake_torch._inductor = types.SimpleNamespace(
        codecache=types.SimpleNamespace(
            FxGraphCache=types.SimpleNamespace(clear=lambda: None)
        )
    )

    fake_nn = types.ModuleType("torch.nn")
    fake_nn.__path__ = []
    fake_functional = types.ModuleType("torch.nn.functional")

    def fake_sdpa(q, k, v, *, is_causal):
        return FakeTensor(q.device.type)

    fake_functional.scaled_dot_product_attention = fake_sdpa
    fake_nn.functional = fake_functional
    fake_torch.nn = fake_nn

    fake_spyre = types.ModuleType("torch_spyre")
    fake_spyre.__path__ = []

    def fake_autoload():
        events.append("autoload")
        state["autoloaded"] = True
        fake_torch.spyre = types.SimpleNamespace(synchronize=lambda: None)

    fake_spyre._autoload = fake_autoload

    fake_inductor = types.ModuleType("torch_spyre._inductor")
    fake_inductor.__path__ = []
    fake_config = types.ModuleType("torch_spyre._inductor.config")
    fake_config.flash_attention_prefill_block_size = 64
    fake_inductor.config = fake_config

    fake_ops = types.ModuleType("torch_spyre.ops")
    fake_ops.__path__ = []

    class FallbackLoader:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch_spyre.ops.fallbacks":
                return importlib.util.spec_from_loader(fullname, self)
            return None

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            events.append(("fallbacks", state["autoloaded"]))
            assert state["autoloaded"]

            class FallbackWarning(UserWarning):
                pass

            module.FallbackWarning = FallbackWarning

    fake_modules = {
        "torch": fake_torch,
        "torch.nn": fake_nn,
        "torch.nn.functional": fake_functional,
        "torch_spyre": fake_spyre,
        "torch_spyre._inductor": fake_inductor,
        "torch_spyre._inductor.config": fake_config,
        "torch_spyre.ops": fake_ops,
    }
    fallback_loader = FallbackLoader()

    with tempfile.TemporaryDirectory() as cache_dir:
        with (
            mock.patch.dict(sys.modules, fake_modules),
            mock.patch.dict(
                os.environ,
                {
                    "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                    "TORCHINDUCTOR_CACHE_DIR": cache_dir,
                },
            ),
            mock.patch("sys.stdout", new=io.StringIO()),
        ):
            sys.modules.pop("torch_spyre.ops.fallbacks", None)
            sys.meta_path.insert(0, fallback_loader)
            try:
                assert sweep._run_child(args) == 0
            finally:
                sys.meta_path.remove(fallback_loader)

    assert events.index("autoload") < events.index(("fallbacks", True))
    assert events.index("autoload") < events.index(("to_spyre", True))


def test_master_layout_xform_variant_uses_config_adjunct_default():
    old = os.environ.get(_LAYOUT_PAIR_ENV)
    os.environ[_LAYOUT_PAIR_ENV] = "7"
    try:
        env = sweep._child_env(_args(), "onchip_master_layout_xform", 128)
    finally:
        if old is None:
            os.environ.pop(_LAYOUT_PAIR_ENV, None)
        else:
            os.environ[_LAYOUT_PAIR_ENV] = old

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert "onchip_master_layout_xform" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_hbm_kv_layout_xform_variant_keeps_kv_repack_disabled():
    parent_overrides = {
        _LAYOUT_PAIR_ENV: "7",
        _KV_REPACK_PLAN_ENV: "1",
        _KV_REPACK_PAIR_ENV: "-2",
        _KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV: "1",
        _KV_REPACK_PAIR_HBM_STAGED_ENV: "1",
        _KV_REPACK_PAIR_CONSUMER_DS_TYPE_ENV: "INPUT",
        _KV_REPACK_COPYBACK_ENV: "-2",
        _KV_REPACK_COPYBACK_HBM_DIRECT_LOAD_ENV: "1",
    }
    old = {key: os.environ.get(key) for key in parent_overrides}
    os.environ.update(parent_overrides)
    try:
        env = sweep._child_env(_args(), "onchip_hbm_kv_layout_xform", 128)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PLAN_ENV] == "0"
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_STAGED_ENV] == "0"
    assert env[_KV_REPACK_PAIR_CONSUMER_DS_TYPE_ENV] == ""
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert env[_KV_REPACK_COPYBACK_HBM_DIRECT_LOAD_ENV] == "0"
    assert "onchip_hbm_kv_layout_xform" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_hbm_kv_layout_xform_kv_hbm_staged_probe_variant_combines_sidecars():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_staged_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_STAGED_ENV] == "1"
    assert env[_KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert "onchip_hbm_kv_layout_xform_kv_hbm_staged_probe" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_hbm_kv_layout_xform_kv_hbm_staged_hoist_probe_variant_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_staged_hoist_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_STAGED_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert "onchip_hbm_kv_layout_xform_kv_hbm_staged_hoist_probe" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_hbm_kv_layout_xform_kv_hbm_prefetch_hoist_probe_variant_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_hoist_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIAL_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_REDUNDANT_FUTURE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIALIZE_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_EXTERNAL_FUTURE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_OVERLAP_AFTER_SYNC_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_hoist_probe" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_child_env_overrides_variant_defaults_after_probe_gate():
    args = _args()
    args.env = [
        f"{_KV_REPACK_HBM_PREFETCH_LX_BASE_ENV}=1625344",
        f"{_KV_REPACK_HBM_PREFETCH_SERIAL_ENV}=1",
    ]

    env = sweep._child_env(
        args,
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_hoist_probe",
        128,
    )

    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LX_BASE_ENV] == "1625344"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIAL_ENV] == "1"


def test_child_env_rejects_malformed_env_override():
    args = _args()
    args.env = ["MISSING_EQUALS"]

    try:
        sweep._child_env(args, "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_hoist_probe", 128)
    except ValueError as exc:
        assert "KEY=VALUE" in str(exc)
    else:
        raise AssertionError("expected malformed --env override to fail")


def test_hbm_kv_layout_xform_kv_hbm_prefetch_serial_probe_variant_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_serial_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIAL_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_REDUNDANT_FUTURE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIALIZE_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_EXTERNAL_FUTURE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_OVERLAP_AFTER_SYNC_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_serial_probe" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_hbm_kv_layout_xform_kv_hbm_prefetch_serialize_current_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_serialize_current_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIAL_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_REDUNDANT_FUTURE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIALIZE_CURRENT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_EXTERNAL_FUTURE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_OVERLAP_AFTER_SYNC_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_serialize_current_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_redundant_future_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_redundant_future_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIAL_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_REDUNDANT_FUTURE_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIALIZE_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_EXTERNAL_FUTURE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_OVERLAP_AFTER_SYNC_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_redundant_future_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_no_after_sync_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_no_after_sync_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIAL_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_REDUNDANT_FUTURE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIALIZE_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_EXTERNAL_FUTURE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_OVERLAP_AFTER_SYNC_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_no_after_sync_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_corelet1_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_corelet1_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIAL_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_CORELET1_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_corelet1_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_roundtrip_corelet1_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_roundtrip_corelet1_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LX_ROUNDTRIP_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_CORELET1_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_roundtrip_corelet1_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_tail_current_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_tail_current_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_tail_current_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_source_fanout_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_source_fanout_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_SOURCE_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_source_fanout_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_source_fanout_tail_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_source_fanout_tail_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_SOURCE_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_source_fanout_tail_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_tail_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_tail_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_USE_UNICAST_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_USE_LXSFP_LX_TRANSFERS_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_COPYBACK_CORE_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_tail_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_tail_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_tail_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_tail_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_core31_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_core31_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_CORE_ENV] == "31"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_core31_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_core31_serialize_loader_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_core31_serialize_loader_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_CORE_ENV] == "31"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIALIZE_LOADER_CORE_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_core31_serialize_loader_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_warpspec_kv_hbm_prefetch_loader_core31_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_warpspec_kv_hbm_prefetch_loader_core31",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_CORE_ENV] == "31"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIALIZE_LOADER_CORE_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_warpspec_kv_hbm_prefetch_loader_core31"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_warpspec_kv_hbm_prefetch_loader_core31_decoupled_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "0"
    assert env[_LAYOUT_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_CORE_ENV] == "31"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_SERIALIZE_LOADER_CORE_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_safesrc_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_safesrc_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_LX_BASE_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_safesrc_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_no_after_sync_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_no_after_sync_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_OVERLAP_AFTER_SYNC_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_no_after_sync_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_unicast_tail_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_unicast_tail_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_USE_UNICAST_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_unicast_tail_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_copyback_tail_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_copyback_tail_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_COPYBACK_CORE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_USE_LXSFP_LX_TRANSFERS_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_copyback_tail_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_lxfifo_copyback_tail_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_lxfifo_copyback_tail_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_USE_LXSFP_LX_TRANSFERS_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_COPYBACK_CORE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_lxfifo_copyback_tail_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_local_copyback_tail_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_local_copyback_tail_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_COPYBACK_CORE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_RESTRICT_TO_COPYBACK_CORE_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_local_copyback_tail_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_tail_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_tail_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_COPYBACK_CORE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_COPYBACK_WITHOUT_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_tail_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_core31_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_core31_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_CORE_ENV] == "31"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_COPYBACK_CORE_ENV] == "31"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_COPYBACK_WITHOUT_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_core31_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_safesrc_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_safesrc_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_LX_BASE_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_COPYBACK_CORE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_COPYBACK_WITHOUT_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_safesrc_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_no_after_sync_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_no_after_sync_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_COPYBACK_CORE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_COPYBACK_WITHOUT_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_OVERLAP_AFTER_SYNC_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_overlap_no_after_sync_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_local_copyback_tail_probe_sets_gate():
    env = sweep._child_env(
        _args(),
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_local_copyback_tail_probe",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_HBM_PREFETCH_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_COPYBACK_CORE_ENV] == "0"
    assert env[_KV_REPACK_HBM_PREFETCH_FANOUT_RESTRICT_TO_COPYBACK_CORE_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_LOADER_FANOUT_FULL_TILE_PIECES_ENV] == "1"
    assert env[_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert (
        "onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_local_copyback_tail_probe"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_hbm_kv_layout_xform_lookahead_variant_keeps_kv_repack_disabled():
    old = {
        _LAYOUT_PAIR_ENV: os.environ.get(_LAYOUT_PAIR_ENV),
        _LAYOUT_LOOKAHEAD_ENV: os.environ.get(_LAYOUT_LOOKAHEAD_ENV),
        _KV_REPACK_PAIR_ENV: os.environ.get(_KV_REPACK_PAIR_ENV),
    }
    os.environ[_LAYOUT_PAIR_ENV] = "7"
    os.environ[_LAYOUT_LOOKAHEAD_ENV] = "4"
    os.environ[_KV_REPACK_PAIR_ENV] = "-2"
    try:
        env = sweep._child_env(_args(), "onchip_hbm_kv_layout_xform_lookahead", 128)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_LAYOUT_LOOKAHEAD_ENV] == "-2"
    assert env[_LAYOUT_HOIST_ENV] == "-1"
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert "onchip_hbm_kv_layout_xform_lookahead" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_hbm_kv_layout_xform_hoist_variant_keeps_kv_repack_disabled():
    old = {
        _LAYOUT_PAIR_ENV: os.environ.get(_LAYOUT_PAIR_ENV),
        _LAYOUT_HOIST_ENV: os.environ.get(_LAYOUT_HOIST_ENV),
        _KV_REPACK_PAIR_ENV: os.environ.get(_KV_REPACK_PAIR_ENV),
    }
    os.environ[_LAYOUT_PAIR_ENV] = "7"
    os.environ[_LAYOUT_HOIST_ENV] = "4"
    os.environ[_KV_REPACK_PAIR_ENV] = "-2"
    try:
        env = sweep._child_env(_args(), "onchip_hbm_kv_layout_xform_hoist", 128)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert env[_LAYOUT_LOOKAHEAD_ENV] == "-1"
    assert env[_LAYOUT_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-1"
    assert "onchip_hbm_kv_layout_xform_hoist" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_master_variant_keeps_low_level_layout_xform_disabled():
    env = sweep._child_env(_args(), "onchip_master", 128)

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "0"
    assert env[_LAYOUT_PAIR_ENV] == "-1"


def test_warp_ifn_prefix_probe_forces_ifn_overlap_tile():
    env = sweep._child_env(_args(), "warp_ifn_prefix_probe", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE"] == "0"
    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP"] == "1"
    assert env[_IFN_PREFIX_FORCE_ENV] == "1"
    assert "warp_ifn_prefix_probe" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_warp_overlap_probe_clears_parent_ifn_prefix_force():
    old = os.environ.get(_IFN_PREFIX_FORCE_ENV)
    os.environ[_IFN_PREFIX_FORCE_ENV] = "1"
    try:
        env = sweep._child_env(_args(), "warp_overlap_probe", 128)
    finally:
        if old is None:
            os.environ.pop(_IFN_PREFIX_FORCE_ENV, None)
        else:
            os.environ[_IFN_PREFIX_FORCE_ENV] = old

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP"] == "1"
    assert env[_IFN_PREFIX_FORCE_ENV] == "0"


def test_layout_xform_pair_overlap_auto_enables_overlap_probe():
    env = sweep._child_env(_args(), "layout_xform_pair_overlap_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_LAYOUT_PAIR_ENV] == "-2"
    assert env[_LAYOUT_PAIR_OVERLAP_ENV] == "1"
    assert "layout_xform_pair_overlap_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_layout_xform_pair_auto_clears_parent_overlap_probe():
    old = os.environ.get(_LAYOUT_PAIR_OVERLAP_ENV)
    os.environ[_LAYOUT_PAIR_OVERLAP_ENV] = "1"
    try:
        env = sweep._child_env(_args(), "layout_xform_pair_auto", 128)
    finally:
        if old is None:
            os.environ.pop(_LAYOUT_PAIR_OVERLAP_ENV, None)
        else:
            os.environ[_LAYOUT_PAIR_OVERLAP_ENV] = old

    assert env[_LAYOUT_PAIR_ENV] == "-2"
    assert env[_LAYOUT_PAIR_OVERLAP_ENV] == "0"


def test_layout_xform_lookahead_auto_enables_lookahead_probe():
    env = sweep._child_env(_args(), "layout_xform_lookahead_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_LAYOUT_PAIR_ENV] == "-2"
    assert env[_LAYOUT_PAIR_OVERLAP_ENV] == "0"
    assert env[_LAYOUT_LOOKAHEAD_ENV] == "-2"
    assert "layout_xform_lookahead_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_layout_xform_hoist_auto_enables_hoist_probe():
    env = sweep._child_env(_args(), "layout_xform_hoist_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_LAYOUT_PAIR_ENV] == "-2"
    assert env[_LAYOUT_LOOKAHEAD_ENV] == "-1"
    assert env[_LAYOUT_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_PLAN_ENV] == "0"
    assert "layout_xform_hoist_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_layout_xform_hoist_kv_repack_plan_variant_emits_descriptor_only():
    env = sweep._child_env(
        _args(),
        "layout_xform_hoist_kv_repack_plan_auto",
        128,
    )

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_LAYOUT_PAIR_ENV] == "-2"
    assert env[_LAYOUT_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_PLAN_ENV] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "1"
    assert env[_KV_REPACK_PAIR_REUSE_ENV] == "1"
    assert env[_KV_REPACK_PAIR_GROUP_ENV] == "0"
    assert "layout_xform_hoist_kv_repack_plan_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_pair_auto_enables_executable_probe():
    env = sweep._child_env(_args(), "kv_repack_pair_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_LAYOUT_PAIR_ENV] == "-1"
    assert env[_LAYOUT_HOIST_ENV] == "-1"
    assert env[_KV_REPACK_PLAN_ENV] == "0"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "1"
    assert env[_KV_REPACK_PAIR_REUSE_ENV] == "1"
    assert env[_KV_REPACK_PAIR_GROUP_ENV] == "0"
    assert env[_KV_REPACK_PAIR_SELF_RESIDENT_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_SOURCE_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV] == "0"
    assert env[_KV_REPACK_PAIR_CONSUMER_CSI_ENV] == "1"
    assert env[_KV_REPACK_PAIR_USE_UNICAST_ENV] == "-1"
    assert env[_KV_REPACK_PAIR_FORCE_MC_ENV] == "-1"
    assert "kv_repack_pair_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_pair_no_ifn_auto_disables_transfer_marker():
    env = sweep._child_env(_args(), "kv_repack_pair_no_ifn_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "0"
    assert env[_KV_REPACK_PAIR_REUSE_ENV] == "1"
    assert env[_KV_REPACK_PAIR_GROUP_ENV] == "0"
    assert "kv_repack_pair_no_ifn_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_pair_no_reuse_auto_disables_subpiece_reuse():
    env = sweep._child_env(_args(), "kv_repack_pair_no_reuse_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "1"
    assert env[_KV_REPACK_PAIR_REUSE_ENV] == "0"
    assert env[_KV_REPACK_PAIR_GROUP_ENV] == "0"
    assert "kv_repack_pair_no_reuse_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_pair_group16_auto_splits_broadcast_groups():
    env = sweep._child_env(_args(), "kv_repack_pair_group16_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "1"
    assert env[_KV_REPACK_PAIR_REUSE_ENV] == "1"
    assert env[_KV_REPACK_PAIR_GROUP_ENV] == "16"
    assert env[_KV_REPACK_PAIR_SELF_RESIDENT_ENV] == "0"
    assert env[_KV_REPACK_PAIR_FORCE_MC_ENV] == "-1"
    assert "kv_repack_pair_group16_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_pair_group8_auto_splits_broadcast_groups_further():
    env = sweep._child_env(_args(), "kv_repack_pair_group8_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_GROUP_ENV] == "8"
    assert env[_KV_REPACK_PAIR_SELF_RESIDENT_ENV] == "0"
    assert env[_KV_REPACK_PAIR_FORCE_MC_ENV] == "-1"
    assert "kv_repack_pair_group8_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_pair_group4_auto_splits_broadcast_groups_further():
    env = sweep._child_env(_args(), "kv_repack_pair_group4_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_GROUP_ENV] == "4"
    assert env[_KV_REPACK_PAIR_SELF_RESIDENT_ENV] == "0"
    assert env[_KV_REPACK_PAIR_FORCE_MC_ENV] == "-1"
    assert "kv_repack_pair_group4_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_pair_self_resident_auto_skips_producer_self_copy():
    env = sweep._child_env(_args(), "kv_repack_pair_self_resident_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_SELF_RESIDENT_ENV] == "1"
    assert env[_KV_REPACK_PAIR_FORCE_MC_ENV] == "-1"
    assert "kv_repack_pair_self_resident_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_pair_hbm_source_auto_loads_source_from_hbm():
    env = sweep._child_env(_args(), "kv_repack_pair_hbm_source_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_SELF_RESIDENT_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_SOURCE_ENV] == "1"
    assert env[_KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV] == "0"
    assert env[_KV_REPACK_PAIR_FORCE_MC_ENV] == "-1"
    assert "kv_repack_pair_hbm_source_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_pair_hbm_direct_load_auto_skips_fanout():
    env = sweep._child_env(_args(), "kv_repack_pair_hbm_direct_load_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "1"
    assert env[_KV_REPACK_PAIR_SELF_RESIDENT_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_SOURCE_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV] == "1"
    assert env[_KV_REPACK_PAIR_CONSUMER_CSI_ENV] == "1"
    assert env[_KV_REPACK_PAIR_FORCE_MC_ENV] == "-1"
    assert "kv_repack_pair_hbm_direct_load_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_pair_hbm_direct_load_no_ifn_auto_skips_transfer_marker():
    env = sweep._child_env(
        _args(), "kv_repack_pair_hbm_direct_load_no_ifn_auto", 128
    )

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "0"
    assert env[_KV_REPACK_PAIR_SELF_RESIDENT_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_SOURCE_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV] == "1"
    assert env[_KV_REPACK_PAIR_CONSUMER_CSI_ENV] == "1"
    assert env[_KV_REPACK_PAIR_FORCE_MC_ENV] == "-1"
    assert "kv_repack_pair_hbm_direct_load_no_ifn_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_pair_hbm_direct_load_no_csi_auto_omits_core_state_init():
    env = sweep._child_env(_args(), "kv_repack_pair_hbm_direct_load_no_csi_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "1"
    assert env[_KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV] == "1"
    assert env[_KV_REPACK_PAIR_CONSUMER_CSI_ENV] == "0"
    assert "kv_repack_pair_hbm_direct_load_no_csi_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_pair_hbm_direct_load_no_ifn_no_csi_auto_combines_controls():
    env = sweep._child_env(
        _args(), "kv_repack_pair_hbm_direct_load_no_ifn_no_csi_auto", 128
    )

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV] == "1"
    assert env[_KV_REPACK_PAIR_CONSUMER_CSI_ENV] == "0"
    assert "kv_repack_pair_hbm_direct_load_no_ifn_no_csi_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_pair_hbm_direct_load_dsinput_auto_overrides_consumer_role():
    env = sweep._child_env(
        _args(), "kv_repack_pair_hbm_direct_load_dsinput_auto", 128
    )

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV] == "1"
    assert env[_KV_REPACK_PAIR_CONSUMER_CSI_ENV] == "0"
    assert env[_KV_REPACK_PAIR_CONSUMER_DS_TYPE_ENV] == "INPUT"
    assert "kv_repack_pair_hbm_direct_load_dsinput_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_pair_hbm_staged_auto_preserves_hbm_consumer_contract():
    env = sweep._child_env(_args(), "kv_repack_pair_hbm_staged_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_STAGED_ENV] == "1"
    assert env[_KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV] == "0"
    assert env[_KV_REPACK_PAIR_CONSUMER_DS_TYPE_ENV] == ""
    assert "kv_repack_pair_hbm_staged_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_pair_hbm_direct_load_canonical_name_auto_retargets_alloc():
    env = sweep._child_env(
        _args(), "kv_repack_pair_hbm_direct_load_canonical_name_auto", 128
    )

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV] == "1"
    assert env[_KV_REPACK_PAIR_CONSUMER_LX_ALLOC_STYLE_ENV] == "canonical_name"
    assert "kv_repack_pair_hbm_direct_load_canonical_name_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_pair_hbm_direct_load_canonical_loop_auto_retargets_alloc():
    env = sweep._child_env(
        _args(), "kv_repack_pair_hbm_direct_load_canonical_loop_auto", 128
    )

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_IFN_ENV] == "0"
    assert env[_KV_REPACK_PAIR_HBM_DIRECT_LOAD_ENV] == "1"
    assert env[_KV_REPACK_PAIR_CONSUMER_LX_ALLOC_STYLE_ENV] == "canonical_loop"
    assert "kv_repack_pair_hbm_direct_load_canonical_loop_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_pair_force_mc3_auto_forces_replication_mode():
    env = sweep._child_env(_args(), "kv_repack_pair_force_mc3_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_SELF_RESIDENT_ENV] == "0"
    assert env[_KV_REPACK_PAIR_FORCE_MC_ENV] == "3"
    assert "kv_repack_pair_force_mc3_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_copyback_auto_enables_copyback_probe():
    env = sweep._child_env(_args(), "kv_repack_copyback_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_PAIR_ENV] == "-1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_CORE_ENV] == "-1"
    assert env[_KV_REPACK_COPYBACK_DIRECT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_SOURCE_FANOUT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_LOAD_ONLY_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_BARRIER_ONLY_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_DATA_ONLY_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_REPLACE_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_COMPUTE_ONLY_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_EXACT_CLONE_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_PRESERVE_NAME_ENV] == "0"
    assert env[_KV_REPACK_PAIR_GROUP_ENV] == "0"
    assert "kv_repack_copyback_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_copyback_group4_auto_splits_copyback_fanout():
    env = sweep._child_env(_args(), "kv_repack_copyback_group4_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_GROUP_ENV] == "4"
    assert "kv_repack_copyback_group4_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_copyback_direct_auto_bypasses_consumer_fanout():
    env = sweep._child_env(_args(), "kv_repack_copyback_direct_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_DIRECT_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "0"
    assert "kv_repack_copyback_direct_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_copyback_hbm_roundtrip_auto_preserves_original_producer():
    env = sweep._child_env(_args(), "kv_repack_copyback_hbm_roundtrip_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_DIRECT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_HBM_SOURCE_FANOUT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_DIRECT_LOAD_ENV] == "0"
    assert "kv_repack_copyback_hbm_roundtrip_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_copyback_hbm_source_fanout_auto_loads_hbm_then_fans_out():
    env = sweep._child_env(
        _args(), "kv_repack_copyback_hbm_source_fanout_auto", 128
    )

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_DIRECT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_SOURCE_FANOUT_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_HBM_DIRECT_LOAD_ENV] == "0"
    assert (
        "kv_repack_copyback_hbm_source_fanout_auto"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_kv_repack_copyback_hbm_direct_load_auto_loads_consumer_lx():
    env = sweep._child_env(_args(), "kv_repack_copyback_hbm_direct_load_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_CORE_ENV] == "-1"
    assert env[_KV_REPACK_COPYBACK_DIRECT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_SOURCE_FANOUT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_DIRECT_LOAD_ENV] == "1"
    assert "kv_repack_copyback_hbm_direct_load_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_copyback_hbm_direct_load_core_variants_select_readback_core():
    for core_id in (0, 1, 8, 16, 31):
        variant = f"kv_repack_copyback_hbm_direct_load_core{core_id}_auto"
        env = sweep._child_env(_args(), variant, 128)

        assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
        assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
        assert env[_KV_REPACK_COPYBACK_CORE_ENV] == str(core_id)
        assert env[_KV_REPACK_COPYBACK_HBM_DIRECT_LOAD_ENV] == "1"
        assert variant in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_copyback_hbm_load_only_auto_skips_hbm_store():
    env = sweep._child_env(_args(), "kv_repack_copyback_hbm_load_only_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_DIRECT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_HBM_LOAD_ONLY_ENV] == "1"
    assert "kv_repack_copyback_hbm_load_only_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_copyback_hbm_barrier_only_auto_skips_hbm_dataops():
    env = sweep._child_env(_args(), "kv_repack_copyback_hbm_barrier_only_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_DIRECT_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_HBM_LOAD_ONLY_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_BARRIER_ONLY_ENV] == "1"
    assert "kv_repack_copyback_hbm_barrier_only_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_copyback_hbm_load_data_only_auto_omits_compute():
    env = sweep._child_env(_args(), "kv_repack_copyback_hbm_load_data_only_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_HBM_LOAD_ONLY_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_HBM_BARRIER_ONLY_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_DATA_ONLY_ENV] == "1"
    assert "kv_repack_copyback_hbm_load_data_only_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_kv_repack_copyback_hbm_barrier_data_only_auto_omits_compute():
    env = sweep._child_env(
        _args(), "kv_repack_copyback_hbm_barrier_data_only_auto", 128
    )

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_HBM_LOAD_ONLY_ENV] == "0"
    assert env[_KV_REPACK_COPYBACK_HBM_BARRIER_ONLY_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_DATA_ONLY_ENV] == "1"
    assert (
        "kv_repack_copyback_hbm_barrier_data_only_auto"
        in env["TORCHINDUCTOR_CACHE_DIR"]
    )


def test_kv_repack_copyback_hbm_barrier_replace_auto_replaces_consumer():
    env = sweep._child_env(_args(), "kv_repack_copyback_hbm_barrier_replace_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_HBM_BARRIER_ONLY_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_REPLACE_ENV] == "1"
    assert "kv_repack_copyback_hbm_barrier_replace_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_copyback_hbm_load_replace_auto_replaces_consumer():
    env = sweep._child_env(_args(), "kv_repack_copyback_hbm_load_replace_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_HBM_LOAD_ONLY_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_REPLACE_ENV] == "1"
    assert "kv_repack_copyback_hbm_load_replace_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_copyback_hbm_compute_replace_auto_wraps_compute_only():
    env = sweep._child_env(_args(), "kv_repack_copyback_hbm_compute_replace_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_REPLACE_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_COMPUTE_ONLY_ENV] == "1"
    assert "kv_repack_copyback_hbm_compute_replace_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_copyback_hbm_exact_clone_auto_replaces_with_clone():
    env = sweep._child_env(_args(), "kv_repack_copyback_hbm_exact_clone_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_EXACT_CLONE_ENV] == "1"
    assert "kv_repack_copyback_hbm_exact_clone_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_copyback_hbm_exact_clone_inplace_auto_preserves_name():
    env = sweep._child_env(
        _args(), "kv_repack_copyback_hbm_exact_clone_inplace_auto", 128
    )

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_EXACT_CLONE_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_PRESERVE_NAME_ENV] == "1"
    assert "kv_repack_copyback_hbm_exact_clone_inplace_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_copyback_hbm_compute_inplace_auto_preserves_name():
    env = sweep._child_env(_args(), "kv_repack_copyback_hbm_compute_inplace_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_COMPUTE_ONLY_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_PRESERVE_NAME_ENV] == "1"
    assert "kv_repack_copyback_hbm_compute_inplace_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_kv_repack_copyback_hbm_barrier_inplace_auto_preserves_name():
    env = sweep._child_env(_args(), "kv_repack_copyback_hbm_barrier_inplace_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_KV_REPACK_COPYBACK_ENV] == "-2"
    assert env[_KV_REPACK_COPYBACK_HBM_ROUNDTRIP_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_HBM_BARRIER_ONLY_ENV] == "1"
    assert env[_KV_REPACK_COPYBACK_PRESERVE_NAME_ENV] == "1"
    assert "kv_repack_copyback_hbm_barrier_inplace_auto" in env[
        "TORCHINDUCTOR_CACHE_DIR"
    ]


def test_layout_xform_hoist_auto_clears_parent_kv_repack_plan_probe():
    old = os.environ.get(_KV_REPACK_PLAN_ENV)
    os.environ[_KV_REPACK_PLAN_ENV] = "1"
    try:
        env = sweep._child_env(_args(), "layout_xform_hoist_auto", 128)
    finally:
        if old is None:
            os.environ.pop(_KV_REPACK_PLAN_ENV, None)
        else:
            os.environ[_KV_REPACK_PLAN_ENV] = old

    assert env[_LAYOUT_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_PLAN_ENV] == "0"


def test_layout_xform_hoist_auto_clears_parent_kv_repack_pair_probe():
    old = os.environ.get(_KV_REPACK_PAIR_ENV)
    os.environ[_KV_REPACK_PAIR_ENV] = "-2"
    try:
        env = sweep._child_env(_args(), "layout_xform_hoist_auto", 128)
    finally:
        if old is None:
            os.environ.pop(_KV_REPACK_PAIR_ENV, None)
        else:
            os.environ[_KV_REPACK_PAIR_ENV] = old

    assert env[_LAYOUT_HOIST_ENV] == "-2"
    assert env[_KV_REPACK_PAIR_ENV] == "-1"


def test_layout_xform_pair_auto_clears_parent_lookahead_probe():
    old = os.environ.get(_LAYOUT_LOOKAHEAD_ENV)
    os.environ[_LAYOUT_LOOKAHEAD_ENV] = "-2"
    try:
        env = sweep._child_env(_args(), "layout_xform_pair_auto", 128)
    finally:
        if old is None:
            os.environ.pop(_LAYOUT_LOOKAHEAD_ENV, None)
        else:
            os.environ[_LAYOUT_LOOKAHEAD_ENV] = old

    assert env[_LAYOUT_PAIR_ENV] == "-2"
    assert env[_LAYOUT_LOOKAHEAD_ENV] == "-1"


def test_layout_xform_pair_auto_clears_parent_hoist_probe():
    old = os.environ.get(_LAYOUT_HOIST_ENV)
    os.environ[_LAYOUT_HOIST_ENV] = "-2"
    try:
        env = sweep._child_env(_args(), "layout_xform_pair_auto", 128)
    finally:
        if old is None:
            os.environ.pop(_LAYOUT_HOIST_ENV, None)
        else:
            os.environ[_LAYOUT_HOIST_ENV] = old

    assert env[_LAYOUT_PAIR_ENV] == "-2"
    assert env[_LAYOUT_HOIST_ENV] == "-1"


def test_causal_flag_is_reflected_in_cache_key():
    args = _args()
    args.is_causal = True

    env = sweep._child_env(args, "onchip_master_layout_xform", 128)

    assert "-C1-" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_parent_forwards_causal_flag_to_child_command():
    args = _args()
    args.is_causal = True
    calls = []

    def fake_run(cmd, *, env, text, capture_output, timeout):
        calls.append(
            {
                "cmd": cmd,
                "env": env,
                "text": text,
                "capture_output": capture_output,
                "timeout": timeout,
            }
        )
        payload = {
            "status": "ok",
            "variant": "onchip_master_layout_xform",
            "shape": {
                "batch": args.batch,
                "heads": args.heads,
                "length": 128,
                "dim": args.dim,
            },
            "cache_dir": env["TORCHINDUCTOR_CACHE_DIR"],
            "median_ms": 0.1,
            "mean_ms": 0.1,
            "max_abs_error": 0.0,
            "mixed_sdscs": [],
        }
        return argparse.Namespace(
            returncode=0,
            stdout="RESULT_JSON:" + json.dumps(payload, sort_keys=True),
            stderr="",
        )

    with mock.patch.object(sweep.subprocess, "run", side_effect=fake_run):
        assert sweep._run_parent(args) == 0

    assert len(calls) == 1
    assert "--is-causal" in calls[0]["cmd"]
    assert "-C1-" in calls[0]["env"]["TORCHINDUCTOR_CACHE_DIR"]


def test_parent_forwards_forbid_fallbacks_to_child_command():
    args = _args()
    args.forbid_fallbacks = True
    calls = []

    def fake_run(cmd, *, env, text, capture_output, timeout):
        calls.append(cmd)
        payload = {
            "status": "ok",
            "variant": "onchip_master_layout_xform",
            "shape": {
                "batch": args.batch,
                "heads": args.heads,
                "length": 128,
                "dim": args.dim,
            },
            "cache_dir": env["TORCHINDUCTOR_CACHE_DIR"],
            "median_ms": 0.1,
            "mean_ms": 0.1,
            "max_abs_error": 0.0,
            "mixed_sdscs": [],
        }
        return argparse.Namespace(
            returncode=0,
            stdout="RESULT_JSON:" + json.dumps(payload, sort_keys=True),
            stderr="",
        )

    with mock.patch.object(sweep.subprocess, "run", side_effect=fake_run):
        assert sweep._run_parent(args) == 0

    assert len(calls) == 1
    assert "--forbid-fallbacks" in calls[0]


def test_parent_records_fallback_readiness_failure_metadata():
    args = _args()
    args.is_causal = True
    args.forbid_fallbacks = True
    args.output_json = ""

    def fake_run(cmd, *, env, text, capture_output, timeout):
        return argparse.Namespace(
            returncode=1,
            stdout="",
            stderr="FallbackWarning: aten.triu.default is falling back to cpu",
        )

    rows = []

    def fake_print_last_result(row):
        rows.append(row)

    with (
        mock.patch.object(sweep.subprocess, "run", side_effect=fake_run),
        mock.patch.object(sweep, "_print_last_result", side_effect=fake_print_last_result),
    ):
        assert sweep._run_parent(args) == 1

    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["is_causal"] is True
    assert rows[0]["fallbacks_forbidden"] is True
    assert "aten.triu.default" in rows[0]["stderr_tail"]


def test_parent_failed_row_summarizes_generated_cache():
    args = _args()
    args.output_json = ""

    def fake_run(cmd, *, env, text, capture_output, timeout):
        cache_dir = Path(env["TORCHINDUCTOR_CACHE_DIR"])
        bundle_dir = cache_dir / "inductor-spyre" / "sdsc_failed_probe"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "sdsc_mixed_flash_kv_repack_probe.json").write_text(
            json.dumps(
                {
                    "mixed_flash_kv_repack_probe": {
                        "opFuncsUsed_": ["STCDPOpHBM"],
                        "datadscs_": [],
                        "flashAttentionPipeline_": {
                            "kv_repack_hbm_prefetch_hoist_role": "current_prefetch",
                        },
                    }
                }
            )
        )
        return argparse.Namespace(
            returncode=1,
            stdout="",
            stderr="assert_close failed",
        )

    rows = []

    def fake_print_last_result(row):
        rows.append(row)

    with (
        mock.patch.object(sweep.subprocess, "run", side_effect=fake_run),
        mock.patch.object(sweep, "_print_last_result", side_effect=fake_print_last_result),
    ):
        assert sweep._run_parent(args) == 1

    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["mixed_sdscs"] == [
        {
            "file": (
                "inductor-spyre/sdsc_failed_probe/"
                "sdsc_mixed_flash_kv_repack_probe.json"
            ),
            "name": "mixed_flash_kv_repack_probe",
            "opFuncsUsed": ["STCDPOpHBM"],
            "datadscs": 0,
            "first_dataop": None,
            "flash_pipeline": {
                "kv_repack_hbm_prefetch_hoist_role": "current_prefetch",
            },
        }
    ]


def test_parent_timeout_metadata_accepts_bytes_stdout_stderr():
    args = _args()
    args.output_json = ""

    def fake_run(cmd, *, env, text, capture_output, timeout):
        raise sweep.subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=timeout,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    rows = []

    def fake_print_last_result(row):
        rows.append(row)

    with (
        mock.patch.object(sweep.subprocess, "run", side_effect=fake_run),
        mock.patch.object(sweep, "_print_last_result", side_effect=fake_print_last_result),
    ):
        assert sweep._run_parent(args) == 1

    assert len(rows) == 1
    assert rows[0]["status"] == "timeout"
    assert rows[0]["stdout_tail"] == "partial stdout"
    assert rows[0]["stderr_tail"] == "partial stderr"


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

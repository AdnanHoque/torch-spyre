# Copyright 2026 IBM Corporation
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

"""Private DD2 bridge from FMS-MO ``FP8Linear`` to Torch-Spyre FP8 ops.

This is deliberately an opt-in integration experiment. It replaces only the
``forward`` method on FMS-MO's concrete ``FP8Linear`` class, and the replacement
delegates back to FMS-MO for every non-Spyre input.
"""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Callable
from typing import Any

import torch


_ENABLE_ENV = "TORCH_SPYRE_ENABLE_FMS_MO_FP8_BRIDGE"
_TARGET_ENV = "TORCH_SPYRE_FP8_TARGET"
_EXPECTED_FMS_MO_VERSION = "0.8.5"
_EXPECTED_TORCHAO_VERSION = "0.11.0"
_EXPECTED_TORCH_PREFIX = "2.11.0+aiu.kineto.1.1.2"
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_SCALE_EPS = torch.finfo(torch.float32).eps

_original_forward: Callable[..., torch.Tensor] | None = None
_patched_class: type[torch.nn.Module] | None = None


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"required package is not installed: {name}") from error


def _reject_non_dd2_environment() -> None:
    if os.environ.get(_TARGET_ENV) != "dd2":
        raise RuntimeError(
            f"{_TARGET_ENV}=dd2 is required; this bridge is a DD2-only experiment"
        )

    target_variables = (
        "AIU_ARCH",
        "DATA_PREC_CONFIG",
        "PREC_CONFIG",
        "SENARCH",
        "SENTARGET",
        "SPYRE_ARCH",
    )
    for name in target_variables:
        value = os.environ.get(name, "").lower()
        if "1p5" in value or "1.5" in value:
            raise RuntimeError(
                f"refusing non-DD2 target from {name}={os.environ[name]!r}"
            )


def _validate_dependency_versions() -> dict[str, str]:
    versions = {
        "torch": str(torch.__version__),
        "fms-model-optimizer": _package_version("fms-model-optimizer"),
        "torchao": _package_version("torchao"),
    }
    if not versions["torch"].startswith(_EXPECTED_TORCH_PREFIX):
        raise RuntimeError(
            "bridge was validated only with Torch "
            f"{_EXPECTED_TORCH_PREFIX}; found {versions['torch']}"
        )
    if versions["fms-model-optimizer"] != _EXPECTED_FMS_MO_VERSION:
        raise RuntimeError(
            "bridge requires fms-model-optimizer "
            f"{_EXPECTED_FMS_MO_VERSION}; found "
            f"{versions['fms-model-optimizer']}"
        )
    if versions["torchao"] != _EXPECTED_TORCHAO_VERSION:
        raise RuntimeError(
            f"bridge requires torchao {_EXPECTED_TORCHAO_VERSION}; "
            f"found {versions['torchao']}"
        )
    return versions


def _validate_spyre_ops() -> None:
    required_ops = (
        "apply_fp8_scale",
        "fp8_matmul_raw",
        "quantize_fp8_with_scale",
    )
    missing = [name for name in required_ops if not hasattr(torch.ops.spyre, name)]
    if missing:
        raise RuntimeError(
            "Torch-Spyre does not contain the complete scaled-matmul candidate; "
            f"missing spyre ops: {', '.join(missing)}"
        )


def dynamic_fp8_row_scale(input_2d: torch.Tensor) -> torch.Tensor:
    """Return TorchAO-compatible symmetric E4M3 scale for every input row."""

    if input_2d.dim() != 2:
        raise ValueError(f"row-scale input must be 2D, got {input_2d.dim()}D")
    max_abs = torch.amax(torch.abs(input_2d.to(torch.float32)), dim=1, keepdim=True)
    return torch.clamp(max_abs / _FP8_MAX, min=_SCALE_EPS)


def _validate_fp8_linear(module: Any, input_2d: torch.Tensor) -> None:
    config = module.linear_config
    activation_config = config.get("input_activations")
    weight_config = config.get("weights")

    if activation_config is None or not activation_config.get("dynamic", False):
        raise RuntimeError("bridge requires dynamic FP8 input activation scaling")
    if activation_config.get("strategy") == "tensor":
        raise RuntimeError("bridge requires per-row input activation scaling")
    if weight_config is None or weight_config.get("dynamic", True):
        raise RuntimeError("bridge requires static checkpoint FP8 weights")
    if weight_config.get("strategy") == "tensor":
        raise RuntimeError("bridge requires per-output-channel weight scaling")

    if input_2d.dtype != torch.float16:
        raise RuntimeError(
            f"DD2 bridge requires FP16 activation input, got {input_2d.dtype}"
        )
    if module.weight.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            "checkpoint weight must already be E4M3 FP8, got "
            f"{module.weight.dtype}"
        )
    if module.weight.dim() != 2 or tuple(module.weight.shape) != (
        module.out_features,
        module.in_features,
    ):
        raise RuntimeError(
            "unexpected FP8 checkpoint weight shape: "
            f"{tuple(module.weight.shape)}"
        )
    if input_2d.shape[1] != module.in_features:
        raise RuntimeError(
            f"input K={input_2d.shape[1]} does not match "
            f"weight K={module.in_features}"
        )
    if module.weight_scale.numel() != module.out_features:
        raise RuntimeError(
            "checkpoint weight_scale must contain one value per output channel; "
            f"got {module.weight_scale.numel()} for N={module.out_features}"
        )


def _spyre_fp8_forward(module: Any, input: torch.Tensor) -> torch.Tensor:
    if _original_forward is None:
        raise RuntimeError("FMS-MO FP8 bridge was not installed")
    if input.device.type != "spyre":
        return _original_forward(module, input)

    if input.dim() < 2:
        raise RuntimeError(f"FP8Linear input must be at least 2D, got {input.dim()}D")

    original_shape = tuple(input.shape)
    input_2d = input.reshape(-1, original_shape[-1])
    _validate_fp8_linear(module, input_2d)

    # Match TorchAO 0.11's dynamic PerRow E4M3 qparams exactly: one FP32
    # symmetric scale per flattened activation row, clamped to FP32 epsilon.
    scale_a = dynamic_fp8_row_scale(input_2d)
    quantized_input = torch.ops.spyre.quantize_fp8_with_scale(input_2d, scale_a)

    # The model checkpoint already owns the E4M3 weight and its static scale.
    # Keep that weight static and present its [K,N] view plus [1,N] FP32 scale
    # directly to the public scaled-matmul contract.
    weight_kn = module.weight.transpose(0, 1)
    scale_b = module.weight_scale.reshape(-1).to(torch.float32).unsqueeze(0)
    bias = module.bias if module.has_bias else None

    output_2d = torch.ops.aten._scaled_mm.default(
        quantized_input,
        weight_kn,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        scale_result=None,
        out_dtype=input.dtype,
        use_fast_accum=True,
    )
    return output_2d.reshape(*original_shape[:-1], module.out_features)


def install_fms_mo_spyre_fp8_bridge() -> dict[str, str]:
    """Install the guarded FMS-MO ``FP8Linear`` forward override."""

    global _original_forward, _patched_class

    if os.environ.get(_ENABLE_ENV) != "1":
        raise RuntimeError(f"explicit opt-in {_ENABLE_ENV}=1 is required")
    _reject_non_dd2_environment()
    versions = _validate_dependency_versions()

    # Importing torch_spyre registers the Spyre custom operators used below.
    import torch_spyre  # noqa: F401
    from fms_mo.aiu_addons.fp8.fp8_linear import FP8Linear

    _validate_spyre_ops()
    if FP8Linear.__module__ != "fms_mo.aiu_addons.fp8.fp8_linear":
        raise RuntimeError(f"unexpected FP8Linear class: {FP8Linear!r}")
    if _patched_class is not None:
        if _patched_class is not FP8Linear:
            raise RuntimeError("a different FP8Linear class was already patched")
        return versions

    _original_forward = FP8Linear.forward
    FP8Linear.forward = _spyre_fp8_forward
    _patched_class = FP8Linear
    return versions

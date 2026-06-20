"""Perf-suite custom op for FMS SwiGLU with device-resident empty weights.

This is a benchmark-only wrapper around the FMS ``GatedLinearUnit`` module.  The
standard ``fms_granite_micro.swiglu`` perf-suite path constructs random CPU
parameters and then copies the whole module to ``spyre``.  For Granite 3 8B
SwiGLU that is about 629 MB of host-to-device parameter traffic before the first
compiled run.

For kernel/SDSC benchmarking we only need tensors with the right shapes and
placements, not checkpoint values.  This wrapper keeps the FMS module structure
but materializes parameters directly on AIU with ``Module.to_empty`` so the
benchmark measures the compute graph instead of a large parameter copy.

Usage:

    python benchmark.py --stack torch-spyre --op fms_swiglu_empty \
        --op-file /tmp/torch-spyre-co-remap-native/tools/perf_suite_fms_swiglu_empty_params_op.py \
        --shape 1 512 4096

    python benchmark.py --stack torch-spyre --op fms_swiglu_unfused_empty \
        --op-file /tmp/torch-spyre-co-remap-native/tools/perf_suite_fms_swiglu_empty_params_op.py \
        --shape 1 512 4096
"""

from __future__ import annotations

import os


_FUSED_OPS = {
    "fms_swiglu_empty",
    "fms_swiglu_fused_empty",
    "swiglu",
    "fms_granite_micro.swiglu",
}
_UNFUSED_OPS = {
    "fms_swiglu_unfused_empty",
    "swiglu_unfused",
    "fms_granite_micro.swiglu_unfused",
}


def _validate_input_shapes(input_shapes) -> None:
    if len(input_shapes) != 1 or len(input_shapes[0]) != 3:
        raise ValueError("FMS SwiGLU empty-params op expects one --shape: B S E")
    if int(input_shapes[0][-1]) != 4096:
        raise ValueError("FMS Granite 3 8B SwiGLU benchmark expects E=4096")


def _fused_weights(op: str) -> bool:
    if op in _FUSED_OPS:
        return True
    if op in _UNFUSED_OPS:
        return False
    raise ValueError(
        "unsupported op for FMS empty-params SwiGLU benchmark: "
        f"{op!r}; expected one of {sorted(_FUSED_OPS | _UNFUSED_OPS)}"
    )


def create_tensors(torch, input_shapes, op, stack):
    _validate_input_shapes(input_shapes)
    return (torch.rand(tuple(input_shapes[0]), dtype=torch.float16),)


def get_module(op, torch, stack, input_shapes):
    _validate_input_shapes(input_shapes)
    from ops.fms_granite_micro import _config_for_shape, _make_swiglu_module

    config = _config_for_shape(input_shapes)
    module = _make_swiglu_module(config, torch, fused_weights=_fused_weights(op))
    module = module.to(dtype=torch.float16)

    if stack == "tsp" and os.environ.get("SPYRE_FMS_SWIGLU_TO_EMPTY", "1") != "0":
        module = module.to_empty(device=torch.device("spyre"))

    module.requires_grad_(False)
    module.eval()
    return module

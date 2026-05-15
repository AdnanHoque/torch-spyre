#!/usr/bin/env python3
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

"""Synthetic restickify scenario survey.

This is a Stage 1/3A probe tool for the Restickify Locality RFC. It runs a taxonomy
of small Torch programs under torch.compile on Spyre, captures the existing
compiler restickify plan, and writes one JSONL row plus one CSV summary row per
case and size.

With --ring-telemetry, it also captures the compiler's restickify byte-hop JSONL
for each case and summarizes total byte-hops, average hops, max hops, and skip
reasons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


torch = None


@dataclass(frozen=True)
class ProbeCase:
    name: str
    scenario: str
    source_hint: str
    description: str
    input_builder: Callable[[int, Any], tuple[tuple[Any, ...], str]]
    fn: Callable[..., Any]
    forward_looking: bool = False


def _pointwise_transpose_add(a, b):
    return a.t() + b


def _pointwise_three_mixed(a, b, c):
    return a.t() + b.t() + c


def _matmul_lhs_wrong_stick(a, b):
    return a.t() @ b


def _matmul_rhs_wrong_stick(a, b):
    return a @ b.t()


def _adds_then_matmul(a, b, c, d):
    return (a + b.t() + c.t()) @ d


def _matmul_then_add(a, b, c):
    return (a @ b) + c.t()


def _transpose_chain(a, b, c):
    return (a.t() + b).t() + c


def _fanout_diamond(a, b, c, d):
    u = a + b.t()
    v = u + c
    w = u.t() + d
    return v + w.t()


def _transpose_4d_chain(x, b, c):
    return (x.transpose(2, 3) + b).transpose(2, 3) + c


def _attention_scores(q, k):
    scale = q.shape[-1] ** -0.5
    return (q @ k.transpose(-2, -1)) * scale


def _attention_value(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


def _linear_weight_transposed(x, w):
    return x @ w.t()


def _mamba_projection_gate(x, w_in, w_out):
    projected = x @ w_in
    x_part, gate = projected.chunk(2, dim=-1)
    gated = x_part * torch.nn.functional.silu(gate)
    return gated @ w_out


def _mamba_chunk_state_join(x, state, w):
    batch, seq, hidden = x.shape
    chunk = 64
    chunks = seq // chunk
    x_chunks = x.view(batch, chunks, chunk, hidden)
    chunk_summary = x_chunks.mean(dim=2)
    state_view = state.unsqueeze(1).expand(batch, chunks, hidden)
    return (chunk_summary + state_view).reshape(batch * chunks, hidden) @ w


def _moe_shared_expert_join(x, shared, expert0, expert1, gate0, gate1):
    shared_out = x @ shared
    routed0 = x @ expert0
    routed1 = x @ expert1
    return shared_out + routed0 * gate0 + routed1 * gate1


def _decode_state_update(x, state, w):
    projected = x @ w
    return state + projected.view_as(state)


def _prefill_projection_join(x, y, z, w):
    return (x + y.t() + z.t()) @ w


def _decode_projection_join(x, y, z, w):
    return (x + y.t() + z.t()) @ w


def _attention_prefill_no_softmax(q, k, v, bias):
    scores = q @ k.transpose(-2, -1)
    mixed = scores + bias.transpose(-2, -1)
    return mixed @ v


def _attention_decode_no_softmax(q, k, v, bias):
    scores = q @ k.transpose(-2, -1)
    mixed = scores + bias.transpose(-2, -1)
    return mixed @ v


def _mamba_chunk_projection_join(x, y, state, w):
    batch, seq, hidden = x.shape
    tokens = batch * seq
    token_view = x.reshape(tokens, hidden)
    state_view = state.unsqueeze(1).expand(batch, seq, hidden).reshape(tokens, hidden)
    return (token_view + y.t() + state_view) @ w


def _moe_two_expert_join(x, dispatch0, dispatch1, shared, expert0, expert1, gate):
    shared_out = x @ shared
    routed0 = (x + dispatch0.t()) @ expert0
    routed1 = (x + dispatch1.t()) @ expert1
    return shared_out + routed0 * gate + routed1 * (1.0 - gate)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _rand(shape: tuple[int, ...], dtype: Any, scale: float = 0.1):
    return torch.randn(shape, dtype=dtype) * scale


def _square_inputs(n: int, dtype: Any, count: int) -> tuple[tuple[Any, ...], str]:
    return tuple(_rand((n, n), dtype) for _ in range(count)), f"{n}x{n}"


def _builder_pointwise2(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    return _square_inputs(n, dtype, 2)


def _builder_pointwise3(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    return _square_inputs(n, dtype, 3)


def _builder_pointwise4(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    return _square_inputs(n, dtype, 4)


def _builder_4d(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    heads = 4
    head_dim = 64
    x = _rand((1, heads, n, head_dim), dtype)
    b = _rand((1, heads, head_dim, n), dtype)
    c = _rand((1, heads, n, head_dim), dtype)
    return (x, b, c), f"1x{heads}x{n}x{head_dim}"


def _builder_attention(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    batch = 1
    heads = 4
    head_dim = 64
    q = _rand((batch, heads, n, head_dim), dtype)
    k = _rand((batch, heads, n, head_dim), dtype)
    v = _rand((batch, heads, n, head_dim), dtype)
    return (q, k, v), f"{batch}x{heads}x{n}x{head_dim}"


def _builder_attention_scores(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    args, shape = _builder_attention(n, dtype)
    q, k, _ = args
    return (q, k), shape


def _builder_linear_weight(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    hidden = n
    out = n
    tokens = max(64, n)
    x = _rand((tokens, hidden), dtype)
    w = _rand((out, hidden), dtype)
    return (x, w), f"tokens={tokens},hidden={hidden},out={out}"


def _builder_mamba_gate(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    tokens = max(128, n)
    hidden = n
    inner = n
    x = _rand((tokens, hidden), dtype)
    w_in = _rand((hidden, 2 * inner), dtype)
    w_out = _rand((inner, hidden), dtype)
    return (x, w_in, w_out), f"tokens={tokens},hidden={hidden},inner={inner}"


def _builder_mamba_state(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    batch = 1
    seq = max(128, n)
    if seq % 64:
        seq += 64 - (seq % 64)
    hidden = n
    x = _rand((batch, seq, hidden), dtype)
    state = _rand((batch, hidden), dtype)
    w = _rand((hidden, hidden), dtype)
    return (x, state, w), f"batch={batch},seq={seq},hidden={hidden}"


def _builder_moe(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    tokens = max(128, n)
    hidden = n
    x = _rand((tokens, hidden), dtype)
    shared = _rand((hidden, hidden), dtype)
    expert0 = _rand((hidden, hidden), dtype)
    expert1 = _rand((hidden, hidden), dtype)
    gate0 = _rand((tokens, 1), dtype)
    gate1 = _rand((tokens, 1), dtype)
    return (
        x,
        shared,
        expert0,
        expert1,
        gate0,
        gate1,
    ), f"tokens={tokens},hidden={hidden},experts=2"


def _builder_decode(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    batch = 1
    hidden = n
    x = _rand((batch, hidden), dtype)
    state = _rand((batch, hidden), dtype)
    w = _rand((hidden, hidden), dtype)
    return (x, state, w), f"batch={batch},hidden={hidden}"


def _builder_prefill_projection(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    tokens = n
    hidden = _env_int("SPYRE_PROBE_HIDDEN", 512)
    x = _rand((tokens, hidden), dtype)
    y = _rand((hidden, tokens), dtype)
    z = _rand((hidden, tokens), dtype)
    w = _rand((hidden, hidden), dtype)
    return (x, y, z, w), f"tokens={tokens},hidden={hidden}"


def _builder_decode_projection(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    tokens = n
    hidden = _env_int("SPYRE_PROBE_HIDDEN", 512)
    x = _rand((tokens, hidden), dtype)
    y = _rand((hidden, tokens), dtype)
    z = _rand((hidden, tokens), dtype)
    w = _rand((hidden, hidden), dtype)
    return (x, y, z, w), f"active_tokens={tokens},hidden={hidden}"


def _builder_attention_prefill_no_softmax(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    batch = _env_int("SPYRE_PROBE_BATCH", 1)
    heads = _env_int("SPYRE_PROBE_HEADS", 4)
    head_dim = _env_int("SPYRE_PROBE_HEAD_DIM", 64)
    seq = n
    q = _rand((batch, heads, seq, head_dim), dtype)
    k = _rand((batch, heads, seq, head_dim), dtype)
    v = _rand((batch, heads, seq, head_dim), dtype)
    bias = _rand((batch, heads, seq, seq), dtype)
    return (q, k, v, bias), f"batch={batch},heads={heads},seq={seq},head_dim={head_dim}"


def _builder_attention_decode_no_softmax(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    batch = _env_int("SPYRE_PROBE_BATCH", 1)
    heads = _env_int("SPYRE_PROBE_HEADS", 4)
    head_dim = _env_int("SPYRE_PROBE_HEAD_DIM", 64)
    kv_seq = n
    q_seq = _env_int("SPYRE_PROBE_Q_SEQ", 1)
    q = _rand((batch, heads, q_seq, head_dim), dtype)
    k = _rand((batch, heads, kv_seq, head_dim), dtype)
    v = _rand((batch, heads, kv_seq, head_dim), dtype)
    bias = _rand((batch, heads, kv_seq, q_seq), dtype)
    return (q, k, v, bias), f"batch={batch},heads={heads},q_seq={q_seq},kv_seq={kv_seq},head_dim={head_dim}"


def _builder_mamba_chunk_projection(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    batch = _env_int("SPYRE_PROBE_BATCH", 1)
    hidden = _env_int("SPYRE_PROBE_HIDDEN", 512)
    seq = n
    tokens = batch * seq
    x = _rand((batch, seq, hidden), dtype)
    y = _rand((hidden, tokens), dtype)
    state = _rand((batch, hidden), dtype)
    w = _rand((hidden, hidden), dtype)
    return (x, y, state, w), f"batch={batch},seq={seq},hidden={hidden}"


def _builder_moe_two_expert(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    tokens = n
    hidden = _env_int("SPYRE_PROBE_HIDDEN", 512)
    x = _rand((tokens, hidden), dtype)
    dispatch0 = _rand((hidden, tokens), dtype)
    dispatch1 = _rand((hidden, tokens), dtype)
    shared = _rand((hidden, hidden), dtype)
    expert0 = _rand((hidden, hidden), dtype)
    expert1 = _rand((hidden, hidden), dtype)
    gate = torch.sigmoid(_rand((tokens, 1), dtype))
    return (
        x,
        dispatch0,
        dispatch1,
        shared,
        expert0,
        expert1,
        gate,
    ), f"tokens={tokens},hidden={hidden},experts=2"


CASES: tuple[ProbeCase, ...] = (
    ProbeCase(
        "pointwise_transpose_add",
        "pointwise_mixed_layout",
        "graph_input",
        "Two-input pointwise join across one transposed input.",
        _builder_pointwise2,
        _pointwise_transpose_add,
    ),
    ProbeCase(
        "pointwise_three_mixed",
        "pointwise_mixed_layout",
        "graph_input",
        "Three-input pointwise join with two transposed inputs.",
        _builder_pointwise3,
        _pointwise_three_mixed,
    ),
    ProbeCase(
        "matmul_lhs_wrong_stick",
        "matmul_fixed_stick",
        "graph_input",
        "Matmul with lhs presented in the wrong stick orientation.",
        _builder_pointwise2,
        _matmul_lhs_wrong_stick,
    ),
    ProbeCase(
        "matmul_rhs_wrong_stick",
        "matmul_fixed_stick",
        "graph_input",
        "Matmul with rhs presented in the wrong stick orientation.",
        _builder_pointwise2,
        _matmul_rhs_wrong_stick,
    ),
    ProbeCase(
        "adds_then_matmul",
        "producer_to_matmul",
        "in_graph_producer",
        "Pointwise producer feeds a matmul that may force restickification.",
        _builder_pointwise4,
        _adds_then_matmul,
    ),
    ProbeCase(
        "matmul_then_add",
        "matmul_to_pointwise",
        "mixed",
        "Matmul producer joins with a transposed pointwise input.",
        _builder_pointwise3,
        _matmul_then_add,
    ),
    ProbeCase(
        "transpose_chain",
        "view_chain",
        "in_graph_producer",
        "Intermediate is consumed through a transposed view.",
        _builder_pointwise3,
        _transpose_chain,
    ),
    ProbeCase(
        "fanout_diamond",
        "fanout_diamond",
        "in_graph_producer",
        "One producer feeds consumers with different view/layout needs.",
        _builder_pointwise4,
        _fanout_diamond,
    ),
    ProbeCase(
        "transpose_4d_chain",
        "view_chain",
        "in_graph_producer",
        "4D transpose chain resembling head/sequence layout pressure.",
        _builder_4d,
        _transpose_4d_chain,
    ),
    ProbeCase(
        "attention_scores",
        "attention",
        "graph_input",
        "QK^T score projection.",
        _builder_attention_scores,
        _attention_scores,
    ),
    ProbeCase(
        "attention_value",
        "attention",
        "mixed",
        "QK^T, softmax, then attention-value matmul.",
        _builder_attention,
        _attention_value,
    ),
    ProbeCase(
        "linear_weight_transposed",
        "graph_input_or_weight",
        "graph_input_or_weight",
        "Linear-style graph-input/weight layout pressure.",
        _builder_linear_weight,
        _linear_weight_transposed,
    ),
    ProbeCase(
        "mamba_projection_gate",
        "mamba_style",
        "mixed",
        "Mamba-style input projection, gate activation, and output projection.",
        _builder_mamba_gate,
        _mamba_projection_gate,
        forward_looking=True,
    ),
    ProbeCase(
        "mamba_chunk_state_join",
        "mamba_style",
        "mixed",
        "Chunked sequence summary joined with persistent state.",
        _builder_mamba_state,
        _mamba_chunk_state_join,
        forward_looking=True,
    ),
    ProbeCase(
        "moe_shared_expert_join",
        "moe_style",
        "mixed",
        "Shared expert plus two routed experts and combine weights.",
        _builder_moe,
        _moe_shared_expert_join,
        forward_looking=True,
    ),
    ProbeCase(
        "decode_state_update",
        "long_context_decode",
        "persistent_state",
        "Decode-like projection merged into persistent state.",
        _builder_decode,
        _decode_state_update,
        forward_looking=True,
    ),
    ProbeCase(
        "prefill_projection_join",
        "prefill_model_slice",
        "in_graph_producer",
        "Prefill-like token projection with transposed joins before matmul.",
        _builder_prefill_projection,
        _prefill_projection_join,
        forward_looking=True,
    ),
    ProbeCase(
        "decode_projection_join",
        "decode_model_slice",
        "in_graph_producer",
        "Decode/batched-decode token projection with transposed joins before matmul.",
        _builder_decode_projection,
        _decode_projection_join,
        forward_looking=True,
    ),
    ProbeCase(
        "attention_prefill_no_softmax",
        "attention_model_slice",
        "in_graph_producer",
        "Prefill attention score/value slice without softmax.",
        _builder_attention_prefill_no_softmax,
        _attention_prefill_no_softmax,
        forward_looking=True,
    ),
    ProbeCase(
        "attention_decode_no_softmax",
        "attention_model_slice",
        "in_graph_producer",
        "Decode attention score/value slice without softmax.",
        _builder_attention_decode_no_softmax,
        _attention_decode_no_softmax,
        forward_looking=True,
    ),
    ProbeCase(
        "mamba_chunk_projection_join",
        "mamba_model_slice",
        "in_graph_producer",
        "Mamba-style chunk/state join before projection.",
        _builder_mamba_chunk_projection,
        _mamba_chunk_projection_join,
        forward_looking=True,
    ),
    ProbeCase(
        "moe_two_expert_join",
        "moe_model_slice",
        "in_graph_producer",
        "MoE-style shared/expert projection and combine.",
        _builder_moe_two_expert,
        _moe_two_expert_join,
        forward_looking=True,
    ),
)


def _dtype_from_name(name: str) -> Any:
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}; choose one of {sorted(table)}") from exc


def _tensor_to_cpu(value: Any) -> Any:
    if hasattr(value, "cpu"):
        return value.cpu()
    if isinstance(value, tuple):
        return tuple(_tensor_to_cpu(v) for v in value)
    if isinstance(value, list):
        return [_tensor_to_cpu(v) for v in value]
    return value


def _sync() -> None:
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "synchronize"):
        try:
            accelerator.synchronize()
            return
        except Exception:
            pass
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and hasattr(cuda, "is_available") and cuda.is_available():
        cuda.synchronize()


def _reset_compile_caches() -> None:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "reset"):
        dynamo.reset()
    try:
        torch._inductor.codecache.FxGraphCache.clear()
    except Exception:
        pass


def _product(values: Any) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _layout_entry_to_dict(consumer: str, entry: dict[str, Any]) -> dict[str, Any]:
    target_layout = entry["target_layout"]
    device_layout = target_layout.device_layout
    elements = _product(target_layout.size)
    dtype = target_layout.dtype
    try:
        element_size = torch.tensor([], dtype=dtype).element_size()
    except Exception:
        element_size = 0
    return {
        "consumer": consumer,
        "arg_name": entry["arg_name"],
        "target_size": [int(s) for s in target_layout.size],
        "target_stride": [int(s) for s in target_layout.stride],
        "target_dtype": str(dtype).replace("torch.", ""),
        "target_device_size": [int(s) for s in device_layout.device_size],
        "target_stride_map": [int(s) for s in device_layout.stride_map],
        "elements": elements,
        "bytes": elements * element_size,
    }


def _summarize_plan(plan: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], int, int]:
    entries: list[dict[str, Any]] = []
    for consumer, resticks in sorted(plan.items()):
        for entry in resticks:
            entries.append(_layout_entry_to_dict(consumer, entry))
    total_elements = sum(entry["elements"] for entry in entries)
    total_bytes = sum(entry["bytes"] for entry in entries)
    return entries, total_elements, total_bytes


def _read_ring_telemetry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "ring_rows": 0,
            "ring_total_bytes": 0,
            "ring_total_byte_hops": 0,
            "ring_avg_hops": 0.0,
            "ring_max_hops": 0,
            "ring_skip_reasons": {},
            "ring_entries": [],
        }

    entries = [json.loads(line) for line in path.read_text().splitlines() if line]
    total_bytes = sum(int(entry.get("bytes_moved") or 0) for entry in entries)
    total_byte_hops = sum(int(entry.get("byte_hops") or 0) for entry in entries)
    max_hops = max((int(entry.get("max_hops") or 0) for entry in entries), default=0)
    skip_reasons: dict[str, int] = {}
    for entry in entries:
        reason = entry.get("skip_reason")
        if reason:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    return {
        "ring_rows": len(entries),
        "ring_total_bytes": total_bytes,
        "ring_total_byte_hops": total_byte_hops,
        "ring_avg_hops": total_byte_hops / total_bytes if total_bytes else 0.0,
        "ring_max_hops": max_hops,
        "ring_skip_reasons": skip_reasons,
        "ring_entries": entries,
    }


def _assert_close(actual: Any, expected: Any, atol: float, rtol: float) -> None:
    if isinstance(actual, tuple):
        assert isinstance(expected, tuple) and len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_close(actual_item, expected_item, atol, rtol)
        return
    if isinstance(actual, list):
        assert isinstance(expected, list) and len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_close(actual_item, expected_item, atol, rtol)
        return
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=True)


def _time_compiled(compiled_fn: Callable[..., Any], dev_args: tuple[Any, ...], warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        compiled_fn(*dev_args)
    _sync()

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        compiled_fn(*dev_args)
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p10_ms": samples[max(0, int(0.10 * (len(samples) - 1)))],
        "p90_ms": samples[min(len(samples) - 1, int(0.90 * (len(samples) - 1)))],
    }


def _run_case(
    case: ProbeCase,
    size: int,
    dtype: Any,
    device: str,
    backend: str,
    skip_correctness: bool,
    do_timing: bool,
    warmup: int,
    iters: int,
    atol: float,
    rtol: float,
    ring_telemetry_path: Path | None,
) -> dict[str, Any]:
    args, shape_label = case.input_builder(size, dtype)
    dev_args = tuple(arg.to(device) if hasattr(arg, "to") else arg for arg in args)

    import torch_spyre._inductor.insert_restickify as insert_restickify

    insert_restickify.restickify_plan = {}
    previous_capture = os.environ.get("SPYRE_CAPTURE_RESTICKIFY_PLAN")
    previous_ring = os.environ.get("SPYRE_RESTICKIFY_RING_TELEMETRY")
    previous_ring_jsonl = os.environ.get("SPYRE_RESTICKIFY_RING_TELEMETRY_JSONL")
    spyre_config = None
    previous_config_ring = None
    previous_config_ring_jsonl = None
    os.environ["SPYRE_CAPTURE_RESTICKIFY_PLAN"] = "1"
    if ring_telemetry_path is not None:
        ring_telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        ring_telemetry_path.unlink(missing_ok=True)
        os.environ["SPYRE_RESTICKIFY_RING_TELEMETRY"] = "1"
        os.environ["SPYRE_RESTICKIFY_RING_TELEMETRY_JSONL"] = str(ring_telemetry_path)
        try:
            import torch_spyre._inductor.config as spyre_config_module

            spyre_config = spyre_config_module
            previous_config_ring = spyre_config.restickify_ring_telemetry
            previous_config_ring_jsonl = spyre_config.restickify_ring_telemetry_jsonl
            spyre_config.restickify_ring_telemetry = True
            spyre_config.restickify_ring_telemetry_jsonl = str(ring_telemetry_path)
        except Exception:
            spyre_config = None

    try:
        _reset_compile_caches()
        compiled = torch.compile(case.fn, backend=backend, dynamic=False)
        start = time.perf_counter()
        result = compiled(*dev_args)
        _sync()
        compile_run_ms = (time.perf_counter() - start) * 1000.0

        plan = dict(insert_restickify.restickify_plan)
        entries, total_elements, total_bytes = _summarize_plan(plan)
        ring_summary = (
            _read_ring_telemetry(ring_telemetry_path)
            if ring_telemetry_path is not None
            else {}
        )

        timing = {}
        if do_timing:
            timing = _time_compiled(compiled, dev_args, warmup, iters)

        if not skip_correctness:
            expected = case.fn(*args)
            actual = _tensor_to_cpu(result)
            _assert_close(actual, expected, atol=atol, rtol=rtol)

        return {
            "status": "ok",
            "case": case.name,
            "scenario": case.scenario,
            "source_hint": case.source_hint,
            "description": case.description,
            "shape": shape_label,
            "size": size,
            "dtype": str(dtype).replace("torch.", ""),
            "forward_looking": case.forward_looking,
            "compile_run_ms": compile_run_ms,
            "restickify_count": len(entries),
            "total_elements": total_elements,
            "total_bytes": total_bytes,
            "entries": entries,
            **ring_summary,
            **timing,
        }
    finally:
        if previous_capture is None:
            os.environ.pop("SPYRE_CAPTURE_RESTICKIFY_PLAN", None)
        else:
            os.environ["SPYRE_CAPTURE_RESTICKIFY_PLAN"] = previous_capture
        if previous_ring is None:
            os.environ.pop("SPYRE_RESTICKIFY_RING_TELEMETRY", None)
        else:
            os.environ["SPYRE_RESTICKIFY_RING_TELEMETRY"] = previous_ring
        if previous_ring_jsonl is None:
            os.environ.pop("SPYRE_RESTICKIFY_RING_TELEMETRY_JSONL", None)
        else:
            os.environ["SPYRE_RESTICKIFY_RING_TELEMETRY_JSONL"] = previous_ring_jsonl
        if spyre_config is not None:
            spyre_config.restickify_ring_telemetry = previous_config_ring
            spyre_config.restickify_ring_telemetry_jsonl = previous_config_ring_jsonl


def _error_row(case: ProbeCase, size: int, dtype: Any, exc: BaseException) -> dict[str, Any]:
    return {
        "status": "error",
        "case": case.name,
        "scenario": case.scenario,
        "source_hint": case.source_hint,
        "description": case.description,
        "size": size,
        "dtype": str(dtype).replace("torch.", ""),
        "forward_looking": case.forward_looking,
        "restickify_count": 0,
        "total_elements": 0,
        "total_bytes": 0,
        "ring_rows": 0,
        "ring_total_bytes": 0,
        "ring_total_byte_hops": 0,
        "ring_avg_hops": 0.0,
        "ring_max_hops": 0,
        "ring_skip_reasons": {},
        "ring_entries": [],
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": row.get("status", ""),
        "scenario": row.get("scenario", ""),
        "case": row.get("case", ""),
        "source_hint": row.get("source_hint", ""),
        "shape": row.get("shape", ""),
        "size": row.get("size", ""),
        "dtype": row.get("dtype", ""),
        "forward_looking": row.get("forward_looking", ""),
        "restickify_count": row.get("restickify_count", 0),
        "total_elements": row.get("total_elements", 0),
        "total_bytes": row.get("total_bytes", 0),
        "ring_rows": row.get("ring_rows", 0),
        "ring_total_bytes": row.get("ring_total_bytes", 0),
        "ring_total_byte_hops": row.get("ring_total_byte_hops", 0),
        "ring_avg_hops": f"{row.get('ring_avg_hops', 0.0):.3f}",
        "ring_max_hops": row.get("ring_max_hops", 0),
        "ring_skip_reasons": json.dumps(row.get("ring_skip_reasons", {}), sort_keys=True),
        "compile_run_ms": f"{row.get('compile_run_ms', 0.0):.3f}" if row.get("compile_run_ms") is not None else "",
        "median_ms": f"{row.get('median_ms', 0.0):.3f}" if row.get("median_ms") is not None else "",
        "p10_ms": f"{row.get('p10_ms', 0.0):.3f}" if row.get("p10_ms") is not None else "",
        "p90_ms": f"{row.get('p90_ms', 0.0):.3f}" if row.get("p90_ms") is not None else "",
        "error_type": row.get("error_type", ""),
        "error": row.get("error", ""),
    }


def _selected_cases(case_names: list[str], scenarios: list[str], include_forward: bool) -> list[ProbeCase]:
    selected = list(CASES)
    if case_names:
        wanted = set(case_names)
        selected = [case for case in selected if case.name in wanted]
    if scenarios:
        wanted_scenarios = set(scenarios)
        selected = [case for case in selected if case.scenario in wanted_scenarios]
    if not include_forward:
        selected = [case for case in selected if not case.forward_looking]
    return selected


def _print_case_list() -> None:
    for case in CASES:
        marker = "forward-looking" if case.forward_looking else "core"
        print(f"{case.name:28} {case.scenario:24} {marker:16} {case.description}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List probe cases and exit.")
    parser.add_argument("--case", action="append", default=[], help="Run one case by name. May be repeated.")
    parser.add_argument("--scenario", action="append", default=[], help="Run one scenario by name. May be repeated.")
    parser.add_argument("--include-forward-looking", action="store_true", help="Include experimental MoE, Mamba, and decode-state probes.")
    parser.add_argument("--size", type=int, action="append", default=[], help="Base square/sequence size. May be repeated.")
    parser.add_argument("--dtype", default="float16", help="Input dtype: float16, bfloat16, or float32.")
    parser.add_argument("--device", default="spyre", help="Execution device, normally spyre.")
    parser.add_argument("--backend", default="inductor", help="torch.compile backend.")
    parser.add_argument("--output-dir", default="/tmp/restickify-scenario-probe", help="Directory for JSONL and CSV output.")
    parser.add_argument("--jsonl-name", default="restickify_scenarios.jsonl", help="JSONL output file name.")
    parser.add_argument("--csv-name", default="restickify_scenarios.csv", help="CSV summary file name.")
    parser.add_argument("--skip-correctness", action="store_true", help="Skip CPU correctness comparison.")
    parser.add_argument("--ring-telemetry", action="store_true", help="Capture restickify byte-hop telemetry per case.")
    parser.add_argument("--time", action="store_true", help="Run warmup/timed iterations after compile.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations for --time.")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations for --time.")
    parser.add_argument("--atol", type=float, default=0.1, help="Correctness absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=0.1, help="Correctness relative tolerance.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--fail-on-error", action="store_true", help="Return nonzero if any selected case fails.")
    return parser.parse_args()


def main() -> int:
    global torch

    args = parse_args()
    if args.list:
        _print_case_list()
        return 0

    import torch as torch_module

    torch = torch_module
    torch.manual_seed(args.seed)

    dtype = _dtype_from_name(args.dtype)
    sizes = args.size or [128]
    selected = _selected_cases(args.case, args.scenario, args.include_forward_looking)
    if not selected:
        raise SystemExit("no probe cases selected")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / args.jsonl_name
    csv_path = output_dir / args.csv_name

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for size in sizes:
            for case in selected:
                telemetry_path = (
                    output_dir / "ring_telemetry" / f"{case.name}_{size}.jsonl"
                    if args.ring_telemetry
                    else None
                )
                try:
                    row = _run_case(
                        case=case,
                        size=size,
                        dtype=dtype,
                        device=args.device,
                        backend=args.backend,
                        skip_correctness=args.skip_correctness,
                        do_timing=args.time,
                        warmup=args.warmup,
                        iters=args.iters,
                        atol=args.atol,
                        rtol=args.rtol,
                        ring_telemetry_path=telemetry_path,
                    )
                except Exception as exc:
                    row = _error_row(case, size, dtype, exc)
                rows.append(row)
                jsonl.write(json.dumps(row, sort_keys=True) + "\n")
                jsonl.flush()
                status = row["status"]
                count = row.get("restickify_count", 0)
                bytes_moved = row.get("total_bytes", 0)
                byte_hops = row.get("ring_total_byte_hops", 0)
                print(
                    f"{status:5} size={size:<5} case={case.name:<28} "
                    f"restickifies={count:<3} bytes={bytes_moved} "
                    f"byte_hops={byte_hops}"
                )

    fieldnames = list(_csv_row(rows[0]).keys())
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))

    errors = [row for row in rows if row["status"] != "ok"]
    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Completed {len(rows)} rows with {len(errors)} errors")
    return 1 if errors and args.fail_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())

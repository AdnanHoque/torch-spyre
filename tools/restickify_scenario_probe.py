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
import copy
import csv
import json
import math
import os
import re
import subprocess
import shutil
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
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


def _transpose_contiguous(a):
    return a.t().contiguous()


def _transpose_clone(a):
    return a.t().clone()


def _adds_then_matmul(a, b, c, d):
    return (a + b.t() + c.t()) @ d


def _plain_adds_then_matmul(a, b, c, d):
    return (a + b + c) @ d


def _computed_transpose_adds_then_matmul(a, b, c, d):
    return (a + (b + c).t()) @ d


def _computed_transpose_adds_then_matmul_tuple(a, b, c, d):
    u = a + (b + c).t()
    return u, u @ d


def _computed_transpose_join(a, b, c):
    return a + (b + c).t()


def _computed_contiguous_then_add(a, b, c):
    return (a + b).t().contiguous() + c


def _computed_contiguous_add_reversed(a, b, c):
    return c + (a + b).t().contiguous()


def _computed_clone_then_add(a, b, c):
    return (a + b).t().clone() + c


def _computed_clone_add_reversed(a, b, c):
    return c + (a + b).t().clone()


def _computed_self_transpose_join(a, b):
    u = a + b
    return u + u.t()


def _computed_self_transpose_join3(a, b, c):
    u = a + b
    return u + u.t() + c


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


def _mlp_gated_projection(x, w_up, w_gate, w_down, residual):
    up = x @ w_up
    gate = x @ w_gate
    activated = up * torch.nn.functional.silu(gate)
    return activated @ w_down + residual


def _mlp_gated_projection_join(x, y, z, w_up, w_gate, w_down, residual):
    joined = x + y.t() + z.t()
    up = joined @ w_up
    gate = joined @ w_gate
    activated = up * torch.nn.functional.silu(gate)
    return activated @ w_down + residual


def _mlp_post_activation_join(x, w_up, post, w_down, residual):
    activated = torch.nn.functional.gelu(x @ w_up)
    joined = activated + post.t()
    return joined @ w_down + residual


def _gated_mlp_post_activation_join(x, w_up, w_gate, post, w_down, residual):
    up = x @ w_up
    gate = x @ w_gate
    activated = up * torch.nn.functional.silu(gate)
    joined = activated + post.t()
    return joined @ w_down + residual


def _attention_prefill_no_softmax(q, k, v, bias):
    scores = q @ k.transpose(-2, -1)
    mixed = scores + bias.transpose(-2, -1)
    return mixed @ v


def _attention_decode_no_softmax(q, k, v, bias):
    scores = q @ k.transpose(-2, -1)
    mixed = scores + bias.transpose(-2, -1)
    return mixed @ v


def _attention_score_join_value_projection(q, k, v, score_bias, score_residual):
    scale = q.shape[-1] ** -0.5
    scores = (q @ k.t()) * scale
    mixed_scores = scores + score_bias.t() + score_residual
    return mixed_scores @ v


def _mamba_chunk_projection_join(x, y, state, w):
    batch, seq, hidden = x.shape
    tokens = batch * seq
    token_view = x.reshape(tokens, hidden)
    state_view = state.unsqueeze(1).expand(batch, seq, hidden).reshape(tokens, hidden)
    return (token_view + y.t() + state_view) @ w


def _mamba_projection_state_gate_join(x, w_in, state, state_mix, gate_bias, w_out):
    projected = x @ w_in
    value, gate = projected.chunk(2, dim=-1)
    state_join = value + state_mix.t() + state
    gated = state_join * torch.nn.functional.silu(gate + gate_bias)
    return gated @ w_out


def _moe_two_expert_join(x, dispatch0, dispatch1, shared, expert0, expert1, gate):
    shared_out = x @ shared
    routed0 = (x + dispatch0.t()) @ expert0
    routed1 = (x + dispatch1.t()) @ expert1
    return shared_out + routed0 * gate + routed1 * (1.0 - gate)


def _moe_combine_join_projection(x, shared, expert0, expert1, gate0, gate1, combine, w_out):
    shared_out = x @ shared
    routed0 = x @ expert0
    routed1 = x @ expert1
    combined = shared_out + routed0 * gate0 + routed1 * gate1
    joined = combined + combine.t()
    return joined @ w_out


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


def _builder_pointwise1(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    return _square_inputs(n, dtype, 1)


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


def _builder_mlp_gated(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    tokens = n
    hidden = _env_int("SPYRE_PROBE_HIDDEN", 512)
    intermediate = _env_int("SPYRE_PROBE_INTERMEDIATE", hidden)
    x = _rand((tokens, hidden), dtype)
    w_up = _rand((hidden, intermediate), dtype)
    w_gate = _rand((hidden, intermediate), dtype)
    w_down = _rand((intermediate, hidden), dtype)
    residual = _rand((tokens, hidden), dtype)
    return (
        x,
        w_up,
        w_gate,
        w_down,
        residual,
    ), f"tokens={tokens},hidden={hidden},intermediate={intermediate}"


def _builder_mlp_gated_join(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    tokens = n
    hidden = _env_int("SPYRE_PROBE_HIDDEN", 512)
    intermediate = _env_int("SPYRE_PROBE_INTERMEDIATE", hidden)
    x = _rand((tokens, hidden), dtype)
    y = _rand((hidden, tokens), dtype)
    z = _rand((hidden, tokens), dtype)
    w_up = _rand((hidden, intermediate), dtype)
    w_gate = _rand((hidden, intermediate), dtype)
    w_down = _rand((intermediate, hidden), dtype)
    residual = _rand((tokens, hidden), dtype)
    return (
        x,
        y,
        z,
        w_up,
        w_gate,
        w_down,
        residual,
    ), f"tokens={tokens},hidden={hidden},intermediate={intermediate}"


def _builder_mlp_post_activation_join(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    tokens = n
    hidden = _env_int("SPYRE_PROBE_HIDDEN", 2048)
    intermediate = _env_int("SPYRE_PROBE_INTERMEDIATE", hidden)
    x = _rand((tokens, hidden), dtype)
    w_up = _rand((hidden, intermediate), dtype)
    post = _rand((intermediate, tokens), dtype)
    w_down = _rand((intermediate, hidden), dtype)
    residual = _rand((tokens, hidden), dtype)
    return (
        x,
        w_up,
        post,
        w_down,
        residual,
    ), f"tokens={tokens},hidden={hidden},intermediate={intermediate}"


def _builder_gated_mlp_post_activation_join(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    tokens = n
    hidden = _env_int("SPYRE_PROBE_HIDDEN", 2048)
    intermediate = _env_int("SPYRE_PROBE_INTERMEDIATE", hidden)
    x = _rand((tokens, hidden), dtype)
    w_up = _rand((hidden, intermediate), dtype)
    w_gate = _rand((hidden, intermediate), dtype)
    post = _rand((intermediate, tokens), dtype)
    w_down = _rand((intermediate, hidden), dtype)
    residual = _rand((tokens, hidden), dtype)
    return (
        x,
        w_up,
        w_gate,
        post,
        w_down,
        residual,
    ), f"tokens={tokens},hidden={hidden},intermediate={intermediate}"


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


def _builder_attention_score_join_value(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    tokens = n
    hidden = _env_int("SPYRE_PROBE_HIDDEN", 2048)
    q = _rand((tokens, hidden), dtype)
    k = _rand((tokens, hidden), dtype)
    v = _rand((tokens, hidden), dtype)
    score_bias = _rand((tokens, tokens), dtype)
    score_residual = _rand((tokens, tokens), dtype)
    return (
        q,
        k,
        v,
        score_bias,
        score_residual,
    ), f"tokens={tokens},hidden={hidden}"


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


def _builder_mamba_projection_state_gate(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    tokens = n
    hidden = _env_int("SPYRE_PROBE_HIDDEN", 2048)
    intermediate = _env_int("SPYRE_PROBE_INTERMEDIATE", hidden)
    x = _rand((tokens, hidden), dtype)
    w_in = _rand((hidden, 2 * intermediate), dtype)
    state = _rand((tokens, intermediate), dtype)
    state_mix = _rand((intermediate, tokens), dtype)
    gate_bias = _rand((tokens, intermediate), dtype)
    w_out = _rand((intermediate, hidden), dtype)
    return (
        x,
        w_in,
        state,
        state_mix,
        gate_bias,
        w_out,
    ), f"tokens={tokens},hidden={hidden},intermediate={intermediate}"


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


def _builder_moe_combine_join_projection(n: int, dtype: Any) -> tuple[tuple[Any, ...], str]:
    tokens = n
    hidden = _env_int("SPYRE_PROBE_HIDDEN", 2048)
    intermediate = _env_int("SPYRE_PROBE_INTERMEDIATE", hidden)
    x = _rand((tokens, hidden), dtype)
    shared = _rand((hidden, intermediate), dtype)
    expert0 = _rand((hidden, intermediate), dtype)
    expert1 = _rand((hidden, intermediate), dtype)
    gate0 = torch.sigmoid(_rand((tokens, 1), dtype))
    gate1 = torch.sigmoid(_rand((tokens, 1), dtype))
    combine = _rand((intermediate, tokens), dtype)
    w_out = _rand((intermediate, hidden), dtype)
    return (
        x,
        shared,
        expert0,
        expert1,
        gate0,
        gate1,
        combine,
        w_out,
    ), f"tokens={tokens},hidden={hidden},intermediate={intermediate},experts=2"


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
        "isolated_transpose_contiguous",
        "isolated_restickify",
        "graph_input",
        "a.t().contiguous(), minimal standalone restickify materialization.",
        _builder_pointwise1,
        _transpose_contiguous,
    ),
    ProbeCase(
        "isolated_transpose_clone",
        "isolated_restickify",
        "graph_input",
        "a.t().clone(), clone-form standalone restickify materialization.",
        _builder_pointwise1,
        _transpose_clone,
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
        "plain_adds_then_matmul",
        "producer_to_matmul",
        "in_graph_producer",
        "Pointwise producer without graph-input transpose feeds a matmul.",
        _builder_pointwise4,
        _plain_adds_then_matmul,
    ),
    ProbeCase(
        "computed_transpose_adds_then_matmul",
        "producer_to_matmul",
        "in_graph_producer",
        "Computed transposed producer feeds a pointwise join before matmul.",
        _builder_pointwise4,
        _computed_transpose_adds_then_matmul,
    ),
    ProbeCase(
        "computed_transpose_adds_then_matmul_tuple",
        "producer_to_matmul",
        "in_graph_producer",
        "Computed transposed producer feeds a pointwise join before matmul and returns the join for bridge validation.",
        _builder_pointwise4,
        _computed_transpose_adds_then_matmul_tuple,
    ),
    ProbeCase(
        "computed_transpose_join",
        "computed_view_join",
        "in_graph_producer",
        "Computed producer is consumed through a transposed pointwise join.",
        _builder_pointwise3,
        _computed_transpose_join,
    ),
    ProbeCase(
        "computed_contiguous_then_add",
        "computed_view_join",
        "in_graph_producer",
        "Computed producer is restickified by contiguous() before a pointwise consumer.",
        _builder_pointwise3,
        _computed_contiguous_then_add,
    ),
    ProbeCase(
        "computed_contiguous_add_reversed",
        "computed_view_join",
        "in_graph_producer",
        "Computed producer is restickified by contiguous() before a reversed pointwise consumer.",
        _builder_pointwise3,
        _computed_contiguous_add_reversed,
    ),
    ProbeCase(
        "computed_clone_then_add",
        "computed_view_join",
        "in_graph_producer",
        "Computed producer is restickified by clone() before a pointwise consumer.",
        _builder_pointwise3,
        _computed_clone_then_add,
    ),
    ProbeCase(
        "computed_clone_add_reversed",
        "computed_view_join",
        "in_graph_producer",
        "Computed producer is restickified by clone() before a reversed pointwise consumer.",
        _builder_pointwise3,
        _computed_clone_add_reversed,
    ),
    ProbeCase(
        "computed_self_transpose_join",
        "computed_view_join",
        "in_graph_producer",
        "Computed producer is consumed in both original and transposed pointwise layouts.",
        _builder_pointwise2,
        _computed_self_transpose_join,
    ),
    ProbeCase(
        "computed_self_transpose_join3",
        "computed_view_join",
        "in_graph_producer",
        "Computed producer is consumed in original and transposed pointwise layouts with an extra add.",
        _builder_pointwise3,
        _computed_self_transpose_join3,
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
        "mlp_gated_projection",
        "mlp_model_slice",
        "in_graph_producer",
        "SwiGLU-style MLP projection, activation, down projection, and residual.",
        _builder_mlp_gated,
        _mlp_gated_projection,
        forward_looking=True,
    ),
    ProbeCase(
        "mlp_gated_projection_join",
        "mlp_model_slice",
        "in_graph_producer",
        "SwiGLU-style MLP with a transposed pointwise join before projection.",
        _builder_mlp_gated_join,
        _mlp_gated_projection_join,
        forward_looking=True,
    ),
    ProbeCase(
        "mlp_post_activation_join",
        "mlp_forward_block_stress",
        "in_graph_producer",
        "MLP slice with a layout-changing post-activation join before down projection.",
        _builder_mlp_post_activation_join,
        _mlp_post_activation_join,
        forward_looking=True,
    ),
    ProbeCase(
        "gated_mlp_post_activation_join",
        "mlp_forward_block_stress",
        "in_graph_producer",
        "SwiGLU-style MLP slice with a post-activation layout join before down projection.",
        _builder_gated_mlp_post_activation_join,
        _gated_mlp_post_activation_join,
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
        "attention_score_join_value_projection",
        "attention_forward_block_stress",
        "in_graph_producer",
        "Attention score/value slice with a layout-changing score join before value projection.",
        _builder_attention_score_join_value,
        _attention_score_join_value_projection,
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
        "mamba_projection_state_gate_join",
        "mamba_forward_block_stress",
        "in_graph_producer",
        "Mamba-style projection/state/gate slice with an in-graph producer join before output projection.",
        _builder_mamba_projection_state_gate,
        _mamba_projection_state_gate_join,
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
    ProbeCase(
        "moe_combine_join_projection",
        "moe_forward_block_stress",
        "in_graph_producer",
        "MoE-style shared/expert/combine slice with an in-graph combine before projection.",
        _builder_moe_combine_join_projection,
        _moe_combine_join_projection,
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
    strict = os.environ.get("SPYRE_PROBE_STRICT_SYNC", "0") == "1"
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "synchronize"):
        try:
            accelerator.synchronize()
            return
        except Exception:
            if strict:
                raise
            pass
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and hasattr(cuda, "is_available") and cuda.is_available():
        cuda.synchronize()


_LX_SPLIT_DATAOP_HANDLED = object()


@contextmanager
def _kernel_launch_debug(
    *,
    sync_after_kernel: bool,
    log_path: Path | None,
    copy_code_dir_root: Path | None = None,
    lx_boundary_stitch_prototype: bool = False,
    lx_split_dataop_prototype: bool = False,
    lx_bridge_same_artifact_splice: bool = False,
    skip_kernel_launch: bool = False,
):
    if (
        not sync_after_kernel
        and log_path is None
        and copy_code_dir_root is None
        and not lx_boundary_stitch_prototype
        and not lx_split_dataop_prototype
        and not lx_bridge_same_artifact_splice
        and not skip_kernel_launch
    ):
        yield
        return

    from torch_spyre.execution import kernel_runner

    original_run = kernel_runner.SpyreSDSCKernelRunner.run
    event_index = 0
    copy_index = 0
    lx_split_bridge_completed = False
    handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
    if copy_code_dir_root is not None:
        copy_code_dir_root.mkdir(parents=True, exist_ok=True)

    def emit(event: dict[str, Any]) -> None:
        nonlocal event_index
        event_index += 1
        event = {
            "event_index": event_index,
            "timestamp_s": time.time(),
            **event,
        }
        text = json.dumps(event, sort_keys=True)
        if handle is not None:
            handle.write(text + "\n")
            handle.flush()
        else:
            print(text, flush=True)

    def patched_run(self, *args, **kw_args):  # noqa: ANN001
        nonlocal copy_index, lx_split_bridge_completed
        files: list[str] = []
        try:
            files = sorted(
                name
                for name in os.listdir(self.code_dir)
                if name.startswith("sdsc_") and name.endswith(".json")
            )
        except Exception:
            files = []
        copied_code_dir = ""
        base = {
            "kernel_name": self.kernel_name,
            "code_dir": self.code_dir,
            "copied_code_dir": copied_code_dir,
            "sdsc_files": files,
            "arg_count": len(args),
        }
        if lx_boundary_stitch_prototype:
            try:
                stitch_info = _apply_lx_boundary_stitch_prototype(Path(self.code_dir))
            except BaseException as exc:
                emit(
                    {
                        "phase": "lx_boundary_stitch_exception",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        **base,
                    }
                )
                raise
            emit({"phase": "lx_boundary_stitch", **stitch_info, **base})
        if lx_bridge_same_artifact_splice:
            try:
                splice_info = _run_lx_bridge_same_artifact_splice_prototype(
                    Path(self.code_dir),
                )
            except BaseException as exc:
                emit(
                    {
                        "phase": "lx_bridge_same_artifact_splice_exception",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        **base,
                    }
                )
                raise
            if splice_info is not None:
                emit({"phase": "lx_bridge_same_artifact_splice", **splice_info, **base})
        if copy_code_dir_root is not None:
            copy_index += 1
            safe_kernel = "".join(
                char if char.isalnum() or char in "._-" else "_"
                for char in str(self.kernel_name)
            )
            copied = copy_code_dir_root / f"{copy_index:04d}_{safe_kernel}"
            shutil.copytree(self.code_dir, copied, dirs_exist_ok=True)
            copied_code_dir = str(copied)
            base["copied_code_dir"] = copied_code_dir
        emit({"phase": "before_launch", **base})
        try:
            if skip_kernel_launch:
                emit({"phase": "skip_launch", **base})
                split_result = _LX_SPLIT_DATAOP_HANDLED
            elif lx_split_dataop_prototype:
                split_result = _run_lx_split_dataop_prototype(
                    Path(self.code_dir),
                    args,
                    emit=emit,
                    base=base,
                )
                if split_result is _LX_SPLIT_DATAOP_HANDLED:
                    lx_split_bridge_completed = True
                if (
                    split_result is None
                    and os.environ.get(
                        "SPYRE_RESTICKIFY_LX_SPLIT_PREPARE_ONLY",
                        "0",
                    )
                    == "1"
                ):
                    emit({"phase": "lx_split_dataop_prepare_only_skip", **base})
                    split_result = _LX_SPLIT_DATAOP_HANDLED
                elif (
                    split_result is None
                    and lx_split_bridge_completed
                    and os.environ.get(
                        "SPYRE_RESTICKIFY_LX_SPLIT_STOP_AFTER_BRIDGE",
                        "0",
                    )
                    == "1"
                ):
                    emit({"phase": "lx_split_dataop_stop_after_bridge_skip", **base})
                    split_result = _LX_SPLIT_DATAOP_HANDLED
            else:
                split_result = None
            if split_result is _LX_SPLIT_DATAOP_HANDLED:
                result = None
            elif split_result is not None:
                result = split_result
            else:
                result = original_run(self, *args, **kw_args)
        except BaseException as exc:
            emit(
                {
                    "phase": "launch_exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **base,
                }
            )
            raise
        emit({"phase": "after_launch", **base})
        if sync_after_kernel:
            emit({"phase": "before_sync", **base})
            _sync()
            emit({"phase": "after_sync", **base})
        return result

    try:
        kernel_runner.SpyreSDSCKernelRunner.run = patched_run
        yield
    finally:
        kernel_runner.SpyreSDSCKernelRunner.run = original_run
        if handle is not None:
            handle.close()


def _run_lx_split_dataop_prototype(
    code_dir: Path,
    launch_args: tuple[Any, ...],
    *,
    emit: Callable[[dict[str, Any]], None],
    base: dict[str, Any],
) -> object | None:
    """Launch producer -> LX data-op restickify -> consumer for one fixture.

    This is intentionally a probe-only path.  It does not alter Torch-Spyre
    lowering; it recognizes an already-generated producer/restickify/consumer
    bundle, builds launchable pieces next to it, and runs those pieces in place
    of the original bundle.
    """

    triplet = _select_restickify_triplet(code_dir)
    if triplet is None:
        return None

    split_root = Path(str(code_dir) + "_lx_split_dataop")
    stages = {
        stage.strip()
        for stage in os.environ.get(
            "SPYRE_RESTICKIFY_LX_SPLIT_STAGES",
            "producer,dataop,consumer",
        ).split(",")
        if stage.strip()
    }
    try:
        summary = _prepare_lx_split_dataop_prototype(
            code_dir,
            triplet=triplet,
            split_root=split_root,
            stages=stages,
        )
    except Exception as exc:  # noqa: BLE001
        emit(
            {
                "phase": "lx_split_dataop_prepare_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                **base,
            }
        )
        raise

    emit({"phase": "lx_split_dataop_launch_start", **summary, **base})
    if os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_PREPARE_ONLY", "0") == "1":
        emit({"phase": "lx_split_dataop_prepare_only_done", **summary, **base})
        return _LX_SPLIT_DATAOP_HANDLED

    from torch_spyre._C import launch_kernel

    producer_args = tuple(launch_args[index] for index in summary["producer_arg_indices"])
    consumer_args = tuple(launch_args[index] for index in summary["consumer_arg_indices"])
    sync_each = os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_SYNC_EACH", "0") == "1"
    if "producer" in stages:
        emit({"phase": "lx_split_dataop_before_producer", **summary, **base})
        launch_kernel(str(split_root / "producer"), producer_args)
        if sync_each:
            _sync()
        emit({"phase": "lx_split_dataop_after_producer", **summary, **base})
    if "dataop" in stages:
        emit({"phase": "lx_split_dataop_before_dataop", **summary, **base})
        launch_kernel(str(split_root / "dataop_launch"), ())
        if sync_each:
            _sync()
        emit({"phase": "lx_split_dataop_after_dataop", **summary, **base})
    if "consumer" in stages:
        emit({"phase": "lx_split_dataop_before_consumer", **summary, **base})
        launch_kernel(str(split_root / "consumer"), consumer_args)
        if sync_each:
            _sync()
        emit({"phase": "lx_split_dataop_after_consumer", **summary, **base})
    emit({"phase": "lx_split_dataop_launch_done", **summary, **base})
    return _LX_SPLIT_DATAOP_HANDLED


def _has_restickify_sdsc(code_dir: Path) -> bool:
    for path in sorted(code_dir.glob("sdsc_*.json"), key=_sdsc_index):
        if "ReStickify" in path.name:
            return True
        try:
            if "ReStickify" in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def _run_lx_bridge_same_artifact_splice_prototype(
    code_dir: Path,
) -> dict[str, Any] | None:
    """Generate an LX bridge frame and splice it into this code dir in place."""

    descriptor = code_dir / "restickify_lx_neighbor_edges.json"
    if not descriptor.exists() or not _has_restickify_sdsc(code_dir):
        return None

    marker = code_dir / ".stage154_lx_bridge_same_artifact_spliced"
    if marker.exists():
        return _read_json_file(marker)

    repo_root = Path(__file__).resolve().parents[1]
    work_root = Path(
        os.environ.get(
            "SPYRE_RESTICKIFY_LX_BRIDGE_SAME_ARTIFACT_WORK_ROOT",
            str(Path(str(code_dir) + "_lx_bridge_same_artifact")),
        )
    )
    shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)
    frame_dir = work_root / "bridge_frame"
    splice_summary = work_root / "splice_summary.json"

    env = {
        **os.environ,
        "SPYRE_RESTICKIFY_LX_DATAOP": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }
    frame_source_dir = code_dir
    consumer_recompile_summary: dict[str, Any] = {}
    if os.environ.get("SPYRE_RESTICKIFY_LX_BRIDGE_PATCH_CONSUMER_SDSC", "0") == "1":
        frame_source_dir = work_root / "frame_source_original"
        shutil.copytree(code_dir, frame_source_dir, dirs_exist_ok=True)
        triplet = _select_restickify_triplet(code_dir)
        if triplet is None:
            raise RuntimeError("LX bridge consumer SDSC patch could not find triplet")
        producer_path, restickify_path, consumer_path = triplet
        producer_payload = _read_json_file(producer_path)
        restickify_payload = _read_json_file(restickify_path)
        consumer_payload = _read_json_file(consumer_path)
        _, producer_dsc = _single_payload_dsc(producer_payload)
        _, restickify_dsc = _single_payload_dsc(restickify_payload)
        _, consumer_dsc = _single_payload_dsc(consumer_payload)
        restickify_input_idx = _first_compute_input_index(restickify_dsc)
        restickify_output_idx = _first_compute_output_index(restickify_dsc)
        restickify_input_hbm = _base_address(
            _alloc_start_map(
                restickify_dsc,
                lds_idx=restickify_input_idx,
                component="hbm",
            )
        )
        restickify_output_hbm = _base_address(
            _alloc_start_map(
                restickify_dsc,
                lds_idx=restickify_output_idx,
                component="hbm",
            )
        )
        producer_output_idx = _find_matching_lds_by_hbm_base(
            producer_dsc,
            candidate_indices=_compute_output_indices(producer_dsc),
            target_base=restickify_input_hbm,
        )
        consumer_input_idx = _find_matching_lds_by_hbm_base(
            consumer_dsc,
            candidate_indices=_compute_input_indices(consumer_dsc),
            target_base=restickify_output_hbm,
        )
        producer_base = int(
            os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_PRODUCER_BASE", "16384")
        )
        consumer_input_name = next(
            str(lds.get("dsName_", f"lds{consumer_input_idx}"))
            for lds in consumer_dsc.get("labeledDs_", []) or []
            if int(lds.get("ldsIdx_", -1)) == int(consumer_input_idx)
        )
        consumer_base = int(
            os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_CONSUMER_BASE", "8192")
        )
        consumer_start = _constant_lx_start_payload(
            num_cores=_core_factor(consumer_payload),
            base=consumer_base,
        )
        producer_start = _constant_lx_start_payload(
            num_cores=_core_factor(producer_payload),
            base=producer_base,
        )
        if (
            os.environ.get(
                "SPYRE_RESTICKIFY_LX_BRIDGE_PATCH_PRODUCER_SOURCE_VIEW",
                "0",
            )
            == "1"
        ):
            restickify_input_lds = next(
                lds
                for lds in restickify_dsc.get("labeledDs_", []) or []
                if int(lds.get("ldsIdx_", -1)) == int(restickify_input_idx)
            )
            restickify_input_primary = restickify_dsc["primaryDsInfo_"][
                str(restickify_input_lds["dsType_"])
            ]
            _patch_producer_output_source_view(
                producer_payload,
                lds_idx=producer_output_idx,
                source_primary=restickify_input_primary,
            )
        _patch_lx_allocation_by_index(
            producer_payload,
            lds_idx=producer_output_idx,
            start_payload=producer_start,
        )
        _patch_consumer_input_lx_map(
            consumer_payload,
            consumer_input_name,
            lds_idx=consumer_input_idx,
            start_payload=consumer_start,
        )
        _force_consumer_corelets(
            consumer_payload,
            factor=_corelet_factor(consumer_start),
        )
        _write_json_file(producer_path, producer_payload)
        _write_json_file(consumer_path, consumer_payload)
        for generated_name in (
            "execute",
            "loadmodel_to_device",
            "loadmodel_to_spad",
            "loadprogram_to_device",
            "loadprogram_to_spad",
            "loadprogram_to_device_dsg.txt",
            "loadprogram_to_spad_dsg.txt",
            "loadmodel_to_device_dsg.txt",
            "loadmodel_to_spad_dsg.txt",
            "execute_dsg.txt",
            "segment_size.json",
            "spyreCodeDir",
        ):
            generated_path = code_dir / generated_name
            if generated_path.is_dir():
                shutil.rmtree(generated_path, ignore_errors=True)
            else:
                generated_path.unlink(missing_ok=True)
        recompile_proc = _run_dxp_bundle(code_dir, cwd=code_dir, verbose=False)
        (work_root / "consumer_sdsc_recompile.stdout.txt").write_text(
            recompile_proc.stdout,
            encoding="utf-8",
        )
        (work_root / "consumer_sdsc_recompile.stderr.txt").write_text(
            recompile_proc.stderr,
            encoding="utf-8",
        )
        if recompile_proc.returncode != 0:
            raise RuntimeError(
                "LX bridge consumer SDSC recompile failed:\n"
                + recompile_proc.stdout[-2000:]
                + recompile_proc.stderr[-4000:]
            )
        consumer_recompile_summary = {
            "consumer_sdsc_patch": "in_place_full_bundle_recompile",
            "consumer_sdsc_recompile_returncode": recompile_proc.returncode,
            "consumer_input_lds_idx": consumer_input_idx,
            "producer_output_lds_idx": producer_output_idx,
            "producer_source_view_patch": os.environ.get(
                "SPYRE_RESTICKIFY_LX_BRIDGE_PATCH_PRODUCER_SOURCE_VIEW",
                "0",
            )
            == "1",
            "producer_lx_unique_starts": _unique_start_values(producer_start),
            "consumer_lx_unique_starts": _unique_start_values(consumer_start),
            "consumer_sdsc_recompile_stdout": str(
                work_root / "consumer_sdsc_recompile.stdout.txt"
            ),
            "consumer_sdsc_recompile_stderr": str(
                work_root / "consumer_sdsc_recompile.stderr.txt"
            ),
        }

    frame_cmd = [
        sys.executable,
        str(repo_root / "tools" / "restickify_lx_bridge_frame.py"),
        "--code-dir",
        str(frame_source_dir),
        "--output-dir",
        str(frame_dir),
        "--mode",
        "stage3b",
        "--fail-on-hbm",
        "--fail-on-missing-senprog",
    ]
    frame_proc = subprocess.run(
        frame_cmd,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (work_root / "bridge_frame.stdout.txt").write_text(
        frame_proc.stdout,
        encoding="utf-8",
    )
    (work_root / "bridge_frame.stderr.txt").write_text(
        frame_proc.stderr,
        encoding="utf-8",
    )
    if frame_proc.returncode != 0:
        raise RuntimeError(
            "LX bridge frame generation failed:\n"
            + frame_proc.stdout[-2000:]
            + frame_proc.stderr[-4000:]
        )

    extra_replacement_args: list[str] = []
    consumer_frame_summary: dict[str, Any] = {}
    if os.environ.get("SPYRE_RESTICKIFY_LX_BRIDGE_PATCH_CONSUMER_FRAME", "0") == "1":
        if (
            os.environ.get(
                "SPYRE_RESTICKIFY_LX_BRIDGE_ALLOW_UNSAFE_CONSUMER_FRAME",
                "0",
            )
            != "1"
        ):
            raise RuntimeError(
                "standalone consumer-frame replacement is disabled because it "
                "can mismatch fused-bundle runtime metadata; use "
                "SPYRE_RESTICKIFY_LX_BRIDGE_PATCH_CONSUMER_SDSC=1 instead"
            )
        triplet = _select_restickify_triplet(code_dir)
        if triplet is None:
            raise RuntimeError("LX bridge consumer-frame patch could not find triplet")
        _producer_path, _restickify_path, consumer_path = triplet
        sdscs = sorted(code_dir.glob("sdsc_*.json"), key=_sdsc_index)
        consumer_pos = sdscs.index(consumer_path)
        consumer_split_root = work_root / "consumer_frame"
        consumer_summary = _prepare_lx_split_dataop_prototype(
            code_dir,
            triplet=triplet,
            split_root=consumer_split_root,
            stages={"consumer"},
        )
        consumer_dir = Path(consumer_summary["consumer_dir"])
        consumer_inits = sorted((consumer_dir / "loadprogram_to_device").glob("*/init.txt"))
        if len(consumer_inits) != 1:
            raise RuntimeError(
                "LX bridge consumer-frame patch expected one consumer init, "
                f"found {len(consumer_inits)} in {consumer_dir}"
            )
        extra_replacement_args.extend(
            ["--replacement-frame", f"{consumer_pos}:{consumer_inits[0]}"]
        )
        consumer_frame_summary = {
            "consumer_frame_position": consumer_pos,
            "consumer_frame_init": str(consumer_inits[0]),
            "consumer_frame_dir": str(consumer_dir),
            "consumer_input_lds_idx": consumer_summary.get("consumer_input_lds_idx"),
            "consumer_lx_start_source": consumer_summary.get("consumer_lx_start_source"),
            "consumer_lx_unique_starts": consumer_summary.get("consumer_lx_unique_starts"),
        }

    splice_cmd = [
        sys.executable,
        str(repo_root / "tools" / "restickify_lx_bridge_same_artifact_splice.py"),
        "--code-dir",
        str(code_dir),
        "--bridge-frame-dir",
        str(frame_dir),
        "--summary",
        str(splice_summary),
        "--require-hbm-free",
        *extra_replacement_args,
    ]
    splice_proc = subprocess.run(
        splice_cmd,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (work_root / "splice.stdout.txt").write_text(splice_proc.stdout, encoding="utf-8")
    (work_root / "splice.stderr.txt").write_text(splice_proc.stderr, encoding="utf-8")
    if splice_proc.returncode != 0:
        raise RuntimeError(
            "same-artifact LX bridge splice failed:\n"
            + splice_proc.stdout[-2000:]
            + splice_proc.stderr[-4000:]
        )

    splice = _read_json_file(splice_summary)
    frame_summary = _read_json_file(frame_dir / "summary.json")
    result = {
        "status": "patched",
        "code_dir": str(code_dir),
        "frame_source_code_dir": str(frame_source_dir),
        "work_root": str(work_root),
        "bridge_frame_dir": str(frame_dir),
        "bridge_frame_returncode": frame_proc.returncode,
        "splice_returncode": splice_proc.returncode,
        "splice_summary": str(splice_summary),
        "patched_bytes": splice.get("patched_bytes"),
        "patched_flits_128b": splice.get("patched_flits_128b"),
        **consumer_recompile_summary,
        **consumer_frame_summary,
        "restickify_start_flit": splice.get("restickify_start_flit"),
        "restickify_original_bytes": splice.get("restickify_original_bytes"),
        "bridge_frame_bytes": splice.get("bridge_frame_bytes"),
        "bridge_hbm_free": (frame_summary.get("frame") or {}).get("hbm_free"),
        "bridge_tokens": (frame_summary.get("frame") or {}).get(
            "senprog_token_counts",
        ),
    }
    _write_json_file(marker, result)
    return result


def _prepare_lx_split_dataop_prototype(
    code_dir: Path,
    *,
    triplet: tuple[Path, Path, Path],
    split_root: Path,
    stages: set[str],
) -> dict[str, Any]:
    ready = split_root / ".ready.json"
    if ready.exists():
        cached = _read_json_file(ready)
        if cached.get("prepare_signature") == _lx_split_prepare_signature(stages):
            return cached

    shutil.rmtree(split_root, ignore_errors=True)
    split_root.mkdir(parents=True, exist_ok=True)

    producer_path, restickify_path, consumer_path = triplet
    producer_payload = _read_json_file(producer_path)
    restickify_payload = _read_json_file(restickify_path)
    consumer_payload = _read_json_file(consumer_path)
    producer_root, producer_dsc = _single_payload_dsc(producer_payload)
    restickify_root, restickify_dsc = _single_payload_dsc(restickify_payload)
    _, consumer_dsc = _single_payload_dsc(consumer_payload)

    restickify_input_idx = _first_compute_input_index(restickify_dsc)
    restickify_output_idx = _first_compute_output_index(restickify_dsc)
    restickify_input_hbm = _base_address(
        _alloc_start_map(restickify_dsc, lds_idx=restickify_input_idx, component="hbm")
    )
    restickify_output_hbm = _base_address(
        _alloc_start_map(restickify_dsc, lds_idx=restickify_output_idx, component="hbm")
    )
    arg_index_by_base = _arg_index_by_hbm_base(
        [producer_dsc, restickify_dsc, consumer_dsc]
    )
    producer_output_idx = _find_matching_lds_by_hbm_base(
        producer_dsc,
        candidate_indices=_compute_output_indices(producer_dsc),
        target_base=restickify_input_hbm,
    )
    consumer_input_idx = _find_matching_lds_by_hbm_base(
        consumer_dsc,
        candidate_indices=_compute_input_indices(consumer_dsc),
        target_base=restickify_output_hbm,
    )

    producer_base = int(os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_PRODUCER_BASE", "16384"))
    consumer_base = int(os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_CONSUMER_BASE", "8192"))
    producer_start_source = "constant"
    consumer_start_source = "constant"
    producer_start = _constant_lx_start_payload(
        num_cores=_core_factor(producer_payload),
        base=producer_base,
    )
    consumer_start = _constant_lx_start_payload(
        num_cores=_core_factor(consumer_payload),
        base=consumer_base,
    )
    if os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_USE_DEBUG_LX", "0") == "1":
        debug_root = code_dir / "debug"
        shutil.rmtree(debug_root, ignore_errors=True)
        debug_result = _run_dxp_bundle(code_dir, cwd=code_dir, verbose=True)
        if debug_result.returncode != 0:
            raise RuntimeError(
                "DXP verbose discovery failed for LX split prototype:\n"
                + debug_result.stdout[-4000:]
                + debug_result.stderr[-4000:]
            )
        producer_debug = (
            debug_root
            / producer_path.stem
            / f"{producer_path.stem}.out.out.json"
        )
        consumer_debug = (
            debug_root
            / consumer_path.stem
            / f"{consumer_path.stem}.out.out.json"
        )
        debug_producer_start = (
            _producer_lxsu_start_payload(
                _read_json_file(producer_debug),
                lds_idx=producer_output_idx,
            )
            if producer_debug.exists()
            else None
        )
        debug_consumer_start = (
            _consumer_lxlu_start_payload(
                _read_json_file(consumer_debug),
                lds_idx=consumer_input_idx,
            )
            if consumer_debug.exists()
            else None
        )
        if debug_producer_start is not None:
            producer_start = debug_producer_start
            producer_start_source = "dxp-debug-lxsu"
        if debug_consumer_start is not None:
            consumer_start = debug_consumer_start
            consumer_start_source = "dxp-debug-lxlu"
        if os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_COLLAPSE_CORELETS", "0") == "1":
            producer_start = _collapse_start_payload_to_one_corelet(producer_start)
            consumer_start = _collapse_start_payload_to_one_corelet(consumer_start)
            producer_start_source += "-collapsed"
            consumer_start_source += "-collapsed"

    if "producer" in stages:
        _patch_lx_allocation_by_index(
            producer_payload,
            lds_idx=producer_output_idx,
            start_payload=producer_start,
        )
    consumer_input_name = next(
        str(lds.get("dsName_", f"lds{consumer_input_idx}"))
        for lds in consumer_dsc.get("labeledDs_", []) or []
        if int(lds.get("ldsIdx_", -1)) == int(consumer_input_idx)
    )
    skip_consumer_patch = (
        os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_SKIP_CONSUMER_PATCH", "0") == "1"
    )
    if "consumer" in stages and not skip_consumer_patch:
        _patch_consumer_input_lx_map(
            consumer_payload,
            consumer_input_name,
            lds_idx=consumer_input_idx,
            start_payload=consumer_start,
        )
        _force_consumer_corelets(
            consumer_payload,
            factor=_corelet_factor(consumer_start),
        )
    _, patched_producer_dsc = _single_payload_dsc(producer_payload)
    _, patched_consumer_dsc = _single_payload_dsc(consumer_payload)
    producer_arg_indices = _dsc_hbm_arg_indices(
        patched_producer_dsc,
        arg_index_by_base,
    )
    consumer_arg_indices = _dsc_hbm_arg_indices(
        patched_consumer_dsc,
        arg_index_by_base,
    )
    if (
        "consumer" in stages
        and os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_KEEP_BRIDGE_ARG", "0") == "1"
    ):
        bridge_arg = arg_index_by_base.get(int(restickify_output_hbm))
        if bridge_arg is not None and bridge_arg not in consumer_arg_indices:
            insert_at = 1 if consumer_arg_indices else 0
            consumer_arg_indices.insert(insert_at, bridge_arg)

    producer_dir = split_root / "producer"
    consumer_dir = split_root / "consumer"
    if "producer" in stages:
        _write_single_sdsc_bundle(producer_dir, producer_path.name, producer_payload)
        _compile_dxp_bundle(producer_dir)
    dataop_summary = (
        _generate_and_package_lx_dataop(code_dir, split_root=split_root)
        if "dataop" in stages
        else {}
    )
    if "consumer" in stages:
        _write_single_sdsc_bundle(consumer_dir, consumer_path.name, consumer_payload)
        _compile_dxp_bundle(consumer_dir)

    summary = {
        "status": "prepared",
        "prepare_signature": _lx_split_prepare_signature(stages),
        "prepared_stages": sorted(stages),
        "split_root": str(split_root),
        "producer_dir": str(producer_dir),
        "consumer_dir": str(consumer_dir),
        "dataop_launch_dir": str(split_root / "dataop_launch"),
        "producer_sdsc": producer_path.name,
        "restickify_sdsc": restickify_path.name,
        "consumer_sdsc": consumer_path.name,
        "producer_output_lds_idx": producer_output_idx,
        "consumer_input_lds_idx": consumer_input_idx,
        "producer_arg_indices": producer_arg_indices,
        "consumer_arg_indices": consumer_arg_indices,
        "producer_lx_base": producer_base,
        "consumer_lx_base": consumer_base,
        "producer_lx_start_source": producer_start_source,
        "consumer_lx_start_source": consumer_start_source,
        "producer_lx_unique_starts": _unique_start_values(producer_start),
        "consumer_lx_unique_starts": _unique_start_values(consumer_start),
        "producer_lx_corelet_factor": _corelet_factor(producer_start),
        "consumer_lx_corelet_factor": _corelet_factor(consumer_start),
        **dataop_summary,
    }
    _write_json_file(ready, summary)
    return summary


def _lx_split_prepare_signature(stages: set[str]) -> dict[str, Any]:
    env_keys = [
        "SPYRE_RESTICKIFY_LX_SPLIT_CONSUMER_BASE",
        "SPYRE_RESTICKIFY_LX_SPLIT_COLLAPSE_CORELETS",
        "SPYRE_RESTICKIFY_LX_SPLIT_DATAOP_MODE",
        "SPYRE_RESTICKIFY_LX_SPLIT_DDL_LIKE_INPUT",
        "SPYRE_RESTICKIFY_LX_SPLIT_DDL_LX_SIZE",
        "SPYRE_RESTICKIFY_LX_SPLIT_PRESERVE_CONSUMER_ROLE",
        "SPYRE_RESTICKIFY_LX_SPLIT_PREPARE_ONLY",
        "SPYRE_RESTICKIFY_LX_SPLIT_PRODUCER_BASE",
        "SPYRE_RESTICKIFY_LX_SPLIT_REQUIRE_MATERIALIZATION_CONTRACT",
        "SPYRE_RESTICKIFY_LX_SPLIT_SKIP_CONSUMER_PATCH",
        "SPYRE_RESTICKIFY_LX_SPLIT_STOP_AFTER_BRIDGE",
        "SPYRE_RESTICKIFY_LX_SPLIT_KEEP_BRIDGE_ARG",
        "SPYRE_RESTICKIFY_LX_SPLIT_USE_DEBUG_LX",
    ]
    return {
        "stages": sorted(stages),
        "env": {key: os.environ.get(key, "") for key in env_keys},
    }


def _select_restickify_triplet(code_dir: Path) -> tuple[Path, Path, Path] | None:
    files = sorted(code_dir.glob("sdsc_*.json"), key=_sdsc_index)
    for index, path in enumerate(files):
        try:
            payload = _read_json_file(path)
            _, dsc = _single_payload_dsc(payload)
        except Exception:  # noqa: BLE001
            continue
        op_names = [
            str(op.get("opFuncName", ""))
            for op in dsc.get("computeOp_", []) or []
        ]
        if not any("ReStickify" in name for name in op_names):
            continue
        if index == 0 or index + 1 >= len(files):
            return None
        return files[index - 1], path, files[index + 1]
    return None


def _sdsc_index(path: Path) -> int:
    match = re.match(r"sdsc_(\d+)_", path.name)
    return int(match.group(1)) if match else 10**9


def _write_single_sdsc_bundle(
    output_dir: Path,
    sdsc_name: str,
    payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_file(output_dir / sdsc_name, payload)
    (output_dir / "bundle.mlir").write_text(
        "module {\n"
        "\tfunc.func @sdsc_bundle() {\n"
        f'\t\tsdscbundle.sdsc_execute () {{sdsc_filename="{sdsc_name}"}}\n'
        "\t\treturn\n"
        "\t}\n"
        "}\n",
        encoding="utf-8",
    )


def _compile_dxp_bundle(code_dir: Path) -> None:
    result = _run_dxp_bundle(code_dir, cwd=code_dir, verbose=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"dxp_standalone failed for {code_dir}:\n"
            + result.stdout[-4000:]
            + result.stderr[-4000:]
        )


def _generate_and_package_lx_dataop(
    code_dir: Path,
    *,
    split_root: Path,
) -> dict[str, Any]:
    gen_dir = split_root / "dataop_gen"
    script = Path(__file__).with_name("restickify_address_preserving_dataop_probe.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--code-dir",
            str(code_dir),
            "--output-dir",
            str(gen_dir),
            "--mode",
            os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_DATAOP_MODE", "stage3b"),
            "--no-run-dataop-standalone",
        ],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "SPYRE_RESTICKIFY_LX_DATAOP": "1",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (split_root / "dataop_gen.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (split_root / "dataop_gen.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            "address-preserving data-op generation failed:\n"
            + proc.stdout[-2000:]
            + proc.stderr[-4000:]
        )

    summary_path = gen_dir / "summary.json"
    dataop_summary = _read_json_file(summary_path)
    endpoint_contract = (
        dataop_summary.get("address_summary", {}).get("endpoint_contract", {})
    )
    contract_source = endpoint_contract.get("source", "")
    require_materialization = (
        os.environ.get(
            "SPYRE_RESTICKIFY_LX_SPLIT_REQUIRE_MATERIALIZATION_CONTRACT",
            "1",
        )
        == "1"
    )
    if require_materialization and contract_source != (
        "schema-v4-lx-materialization-contract"
    ):
        raise RuntimeError(
            "LX split data-op requires schema-v4 materialization contract; "
            f"got {contract_source or '<missing>'}"
        )

    patched_path = Path(dataop_summary["patched"]["path"])
    exporter = os.environ.get(
        "SPYRE_RESTICKIFY_DEEPRT_DATAOP_EXPORTER",
        "/tmp/stage65-deeprt-dataop-probe",
    )
    export_dir = split_root / "dataop_export"
    proc = subprocess.run(
        [exporter, str(patched_path), str(export_dir), "sentient"],
        cwd=split_root,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (split_root / "dataop_export.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (split_root / "dataop_export.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Deeprt data-op export failed with return code {proc.returncode}:\n"
            + proc.stdout[-2000:]
            + proc.stderr[-4000:]
        )

    init_candidates = sorted((export_dir / "execute").glob("*/init.txt"))
    senprog_candidates = sorted((export_dir / "execute").glob("*/senprog.txt"))
    if not init_candidates:
        raise RuntimeError(
            "Deeprt data-op export did not produce execute/*/init.txt:\n"
            + proc.stdout[-2000:]
            + proc.stderr[-4000:]
        )

    launch_dir = split_root / "dataop_launch"
    launch_name = launch_dir.name
    init_target = (
        launch_dir
        / "loadprogram_to_device"
        / f"{launch_name}-SenProgSend"
        / "init.txt"
    )
    init_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(init_candidates[0], init_target)
    (launch_dir / "bundle.mlir").write_text(
        "module {\n"
        "\tfunc.func @sdsc_bundle() {\n"
        "\t\tsdscbundle.sdsc_execute () {sdsc_filename=\"sdsc_0_lx_split_dataop.json\"}\n"
        "\t\treturn\n"
        "\t}\n"
        "}\n",
        encoding="utf-8",
    )
    return {
        "dataop_generation_summary": str(summary_path),
        "dataop_contract_source": contract_source,
        "dataop_materialization_kind": endpoint_contract.get(
            "materialization_kind",
            "",
        ),
        "dataop_intended_sequence": endpoint_contract.get(
            "intended_deeptools_sequence",
            [],
        ),
        "dataop_producer_pieces_patched": dataop_summary.get("patched", {}).get(
            "producer_pieces_patched",
            0,
        ),
        "dataop_consumer_pieces_patched": dataop_summary.get("patched", {}).get(
            "consumer_pieces_patched",
            0,
        ),
        "dataop_patched_sdsc": str(patched_path),
        "dataop_export_dir": str(export_dir),
        "dataop_export_returncode": proc.returncode,
        "dataop_init": str(init_candidates[0]),
        "dataop_senprog": str(senprog_candidates[0]) if senprog_candidates else "",
    }


def _first_compute_input_index(dsc: dict[str, Any]) -> int:
    indices = _compute_input_indices(dsc)
    if not indices:
        raise ValueError("DSC has no compute input LDS")
    return indices[0]


def _first_compute_output_index(dsc: dict[str, Any]) -> int:
    indices = _compute_output_indices(dsc)
    if not indices:
        raise ValueError("DSC has no compute output LDS")
    return indices[0]


def _compute_input_indices(dsc: dict[str, Any]) -> list[int]:
    ops = dsc.get("computeOp_", []) or []
    if not ops:
        return []
    return [_lds_label_index(token) for token in ops[0].get("inputLabeledDs", []) or []]


def _compute_output_indices(dsc: dict[str, Any]) -> list[int]:
    ops = dsc.get("computeOp_", []) or []
    if not ops:
        return []
    return [_lds_label_index(token) for token in ops[0].get("outputLabeledDs", []) or []]


def _lds_label_index(token: str) -> int:
    match = re.search(r"-idx(\d+)$", str(token))
    if not match:
        raise ValueError(f"could not parse LDS index from {token!r}")
    return int(match.group(1))


def _parse_core_key(key: str) -> int | None:
    parts = [part.strip() for part in key.strip("[]").split(",") if part.strip()]
    if not parts:
        return None
    return int(parts[0])


def _alloc_start_map(
    dsc: dict[str, Any],
    *,
    lds_idx: int,
    component: str,
) -> dict[int, int]:
    candidates: list[tuple[str, dict[int, int]]] = []
    for node in dsc.get("scheduleTree_", []) or []:
        if not isinstance(node, dict) or node.get("nodeType_") != "allocate":
            continue
        if int(node.get("ldsIdx_", -1)) != int(lds_idx):
            continue
        if node.get("component_") != component:
            continue
        data = ((node.get("startAddressCoreCorelet_") or {}).get("data_") or {})
        starts = {
            core: int(value)
            for key, value in data.items()
            if (core := _parse_core_key(str(key))) is not None
        }
        if starts:
            candidates.append((str(node.get("name_", "")), starts))
    if not candidates:
        raise ValueError(f"no {component} allocation found for ldsIdx {lds_idx}")
    candidates.sort(key=lambda item: ("allocate_lds" not in item[0], item[0]))
    return candidates[0][1]


def _base_address(starts: dict[int, int]) -> int:
    return int(starts[min(starts)])


def _find_matching_lds_by_hbm_base(
    dsc: dict[str, Any],
    *,
    candidate_indices: list[int],
    target_base: int,
) -> int:
    for index in candidate_indices:
        try:
            starts = _alloc_start_map(dsc, lds_idx=index, component="hbm")
        except ValueError:
            continue
        if _base_address(starts) == int(target_base):
            return index
    raise ValueError(f"no candidate LDS has HBM base {target_base}")


def _arg_index_by_hbm_base(dscs: list[dict[str, Any]]) -> dict[int, int]:
    bases: set[int] = set()
    for dsc in dscs:
        for lds in dsc.get("labeledDs_", []) or []:
            try:
                starts = _alloc_start_map(
                    dsc,
                    lds_idx=int(lds.get("ldsIdx_", -1)),
                    component="hbm",
                )
            except ValueError:
                continue
            bases.add(_base_address(starts))
    return {base: index for index, base in enumerate(sorted(bases))}


def _dsc_hbm_arg_indices(
    dsc: dict[str, Any],
    arg_index_by_base: dict[int, int],
) -> list[int]:
    indices: list[int] = []
    for lds in sorted(
        dsc.get("labeledDs_", []) or [],
        key=lambda item: int(item.get("ldsIdx_", -1)),
    ):
        lds_idx = int(lds.get("ldsIdx_", -1))
        try:
            base = _base_address(
                _alloc_start_map(dsc, lds_idx=lds_idx, component="hbm")
            )
        except ValueError:
            continue
        if base not in arg_index_by_base:
            raise ValueError(f"HBM base {base} missing from split-launch arg map")
        indices.append(arg_index_by_base[base])
    return indices


def _patch_lx_allocation_by_index(
    payload: dict[str, Any],
    *,
    lds_idx: int,
    start_payload: dict[str, Any],
) -> None:
    root, dsc = _single_payload_dsc(payload)
    corelet_factor = _corelet_factor(start_payload)
    root["coreletFoldProp_"] = {"factor_": corelet_factor, "label_": "corelet"}
    dsc["numCoreletsUsed_"] = corelet_factor
    dsc["numCoreletsUsed_DSC2_"] = corelet_factor
    lds_name = None
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            lds_name = str(lds.get("dsName_", f"lds{lds_idx}"))
            lx_meta = dict(lds.get("memOrg_", {}).get("lx", {}))
            lx_meta.update(
                {
                    "isPresent": 1,
                    "isPadded": 0,
                    "isZeroPadded": 0,
                    "zpadGapFront": [0, 0],
                    "gapPerDim": {},
                    "dsOffset": 0,
                    "allocateNode_": f"allocate-{lds_name}_lx",
                }
            )
            lds["memOrg_"] = {"lx": lx_meta}
            lds["hbmStartAddress_"] = -1
            lds["hbmSize_"] = 0
            if int(lds.get("lxSize_", 0) or 0) <= 0:
                lds["lxSize_"] = 2_147_483_647
            if int(lds.get("lxBufferSize_", 0) or 0) <= 0:
                lds["lxBufferSize_"] = 2_147_483_647
            break
    if lds_name is None:
        raise ValueError(f"LDS index {lds_idx} not found")
    for node in dsc.get("scheduleTree_", []) or []:
        if node.get("nodeType_") == "allocate" and int(node.get("ldsIdx_", -1)) == int(lds_idx):
            node["name_"] = f"allocate-{lds_name}_lx"
            node["component_"] = "lx"
            node["startAddressCoreCorelet_"] = start_payload


def _patch_producer_output_source_view(
    payload: dict[str, Any],
    *,
    lds_idx: int,
    source_primary: dict[str, Any],
) -> None:
    """Make a producer output use the restickify source-view layout role."""

    _root, dsc = _single_payload_dsc(payload)
    bridge_role = os.environ.get(
        "SPYRE_RESTICKIFY_LX_BRIDGE_PRODUCER_SOURCE_VIEW_ROLE",
        "LX_BRIDGE_SOURCE",
    )
    dsc.setdefault("primaryDsInfo_", {})[bridge_role] = copy.deepcopy(source_primary)
    layout_order = list(source_primary.get("layoutDimOrder_", []) or [])
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            lds["dsType_"] = bridge_role
            break
    for node in dsc.get("scheduleTree_", []) or []:
        if node.get("nodeType_") != "allocate":
            continue
        if int(node.get("ldsIdx_", -1)) != int(lds_idx):
            continue
        if layout_order:
            node["layoutDimOrder_"] = copy.deepcopy(layout_order)


def _allocation_node_by_lds(
    dsc: dict[str, Any],
    *,
    lds_idx: int,
) -> dict[str, Any] | None:
    for node in dsc.get("scheduleTree_", []) or []:
        if node.get("nodeType_") == "allocate" and int(node.get("ldsIdx_", -1)) == int(lds_idx):
            return node
    return None


def _copy_allocation_shape_metadata(
    *,
    src_dsc: dict[str, Any],
    src_lds_idx: int,
    dst_dsc: dict[str, Any],
    dst_lds_idx: int,
    keys: tuple[str, ...],
) -> None:
    src_node = _allocation_node_by_lds(src_dsc, lds_idx=src_lds_idx)
    dst_node = _allocation_node_by_lds(dst_dsc, lds_idx=dst_lds_idx)
    if src_node is None or dst_node is None:
        raise ValueError(
            f"cannot copy allocation metadata src={src_lds_idx} dst={dst_lds_idx}"
        )
    for key in keys:
        if key in src_node:
            dst_node[key] = copy.deepcopy(src_node[key])


def _canonical_dim_name(name: Any) -> str:
    return str(name).rstrip("_")


def _root_work_splits(root: dict[str, Any]) -> dict[str, int]:
    splits = {
        _canonical_dim_name(dim): int(value)
        for dim, value in (root.get("numWkSlicesPerDim_", {}) or {}).items()
    }
    if splits:
        return splits

    inferred: dict[str, int] = {}
    for raw_slice in (root.get("coreIdToWkSlice_", {}) or {}).values():
        if not isinstance(raw_slice, dict):
            continue
        for dim, value in raw_slice.items():
            dim_name = _canonical_dim_name(dim)
            inferred[dim_name] = max(inferred.get(dim_name, 0), int(value) + 1)
    return inferred


def _root_core_to_slice(root: dict[str, Any]) -> dict[int, dict[str, int]]:
    mapping: dict[int, dict[str, int]] = {}
    for core, raw_slice in (root.get("coreIdToWkSlice_", {}) or {}).items():
        if not isinstance(raw_slice, dict):
            continue
        mapping[int(core)] = {
            _canonical_dim_name(dim): int(value)
            for dim, value in raw_slice.items()
        }
    return mapping


def _dsc_core_chunk_sizes(dsc: dict[str, Any]) -> dict[str, int]:
    for param in (dsc.get("dataStageParam_", {}) or {}).values():
        if not isinstance(param, dict):
            continue
        stage = param.get("ss_") or param.get("el_") or {}
        dims: dict[str, int] = {}
        for dim, value in stage.items():
            dim_name = _canonical_dim_name(dim)
            if dim_name == "name":
                continue
            try:
                dims[dim_name] = int(value)
            except (TypeError, ValueError):
                continue
        if dims:
            return dims
    return {}


def _dim_totals_from_contract(
    root: dict[str, Any],
    dsc: dict[str, Any],
) -> dict[str, int]:
    splits = _root_work_splits(root)
    chunk_sizes = _dsc_core_chunk_sizes(dsc)
    return {
        dim: chunk_sizes[dim] * max(1, splits.get(dim, 1))
        for dim in chunk_sizes
    }


def _work_region_for_core(
    *,
    core: int,
    root: dict[str, Any],
    dsc: dict[str, Any],
) -> dict[str, tuple[int, int]]:
    splits = _root_work_splits(root)
    core_to_slice = _root_core_to_slice(root)
    chunk_sizes = _dsc_core_chunk_sizes(dsc)
    totals = _dim_totals_from_contract(root, dsc)
    region: dict[str, tuple[int, int]] = {}
    dims = sorted(set(splits) | set(chunk_sizes) | set(totals))
    for dim in dims:
        factor = max(1, int(splits.get(dim, 1)))
        index = int(core_to_slice.get(core, {}).get(dim, 0))
        chunk = chunk_sizes.get(dim)
        total = totals.get(dim)
        if chunk is None and total is not None:
            chunk = math.ceil(total / factor)
        if chunk is None:
            continue
        start = index * chunk
        end = start + chunk
        if total is not None:
            end = min(end, total)
            if end < start:
                end = start
        region[dim] = (start, end)
    return region


def _region_volume(region: dict[str, tuple[int, int]]) -> int:
    if not region:
        return 0
    volume = 1
    for start, end in region.values():
        volume *= max(0, int(end) - int(start))
    return volume


def _intersection_volume(
    left: dict[str, tuple[int, int]],
    right: dict[str, tuple[int, int]],
) -> int:
    dims = set(left) | set(right)
    if not dims:
        return 0
    volume = 1
    for dim in dims:
        if dim not in left or dim not in right:
            return 0
        start = max(left[dim][0], right[dim][0])
        end = min(left[dim][1], right[dim][1])
        volume *= max(0, end - start)
    return volume


def _invalid_core_map_entries(root: dict[str, Any]) -> list[dict[str, int | str]]:
    splits = _root_work_splits(root)
    invalid: list[dict[str, int | str]] = []
    for core, raw_slice in _root_core_to_slice(root).items():
        for dim, index in raw_slice.items():
            factor = splits.get(dim)
            if factor is None:
                invalid.append({"core": core, "dim": dim, "index": index, "factor": -1})
            elif index < 0 or index >= factor:
                invalid.append(
                    {"core": core, "dim": dim, "index": index, "factor": factor}
                )
    return invalid


def _format_region(region: dict[str, tuple[int, int]]) -> dict[str, list[int]]:
    return {dim: [int(start), int(end)] for dim, (start, end) in region.items()}


def _stock_lx_alias_ownership_summary(
    *,
    producer_root: dict[str, Any],
    producer_dsc: dict[str, Any],
    restickify_root: dict[str, Any],
    restickify_dsc: dict[str, Any],
) -> dict[str, Any]:
    producer_map = _root_core_to_slice(producer_root)
    restickify_map = _root_core_to_slice(restickify_root)
    cores = sorted(set(producer_map) & set(restickify_map))
    samples: list[dict[str, Any]] = []
    overlap_ratios: list[float] = []
    for core in cores:
        producer_region = _work_region_for_core(
            core=core,
            root=producer_root,
            dsc=producer_dsc,
        )
        restickify_region = _work_region_for_core(
            core=core,
            root=restickify_root,
            dsc=restickify_dsc,
        )
        needed = _region_volume(restickify_region)
        overlap = _intersection_volume(producer_region, restickify_region)
        ratio = (overlap / needed) if needed else 0.0
        overlap_ratios.append(ratio)
        if core in {0, 1, 7, 15, 31}:
            samples.append(
                {
                    "core": core,
                    "producer_slice": producer_map.get(core, {}),
                    "restickify_slice": restickify_map.get(core, {}),
                    "producer_region": _format_region(producer_region),
                    "restickify_needed_region": _format_region(restickify_region),
                    "local_overlap_ratio": ratio,
                }
            )

    invalid_entries = _invalid_core_map_entries(restickify_root)
    min_overlap = min(overlap_ratios, default=0.0)
    avg_overlap = (
        sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0.0
    )
    max_overlap = max(overlap_ratios, default=0.0)
    return {
        "producer_work_splits": _root_work_splits(producer_root),
        "restickify_work_splits": _root_work_splits(restickify_root),
        "producer_core_chunk_sizes": _dsc_core_chunk_sizes(producer_dsc),
        "restickify_core_chunk_sizes": _dsc_core_chunk_sizes(restickify_dsc),
        "producer_dim_totals": _dim_totals_from_contract(producer_root, producer_dsc),
        "restickify_dim_totals": _dim_totals_from_contract(
            restickify_root, restickify_dsc
        ),
        "core_count_compared": len(cores),
        "core_maps_equal": producer_map == restickify_map,
        "restickify_core_map_valid": not invalid_entries,
        "restickify_invalid_core_map_entries": invalid_entries[:12],
        "restickify_invalid_core_map_entry_count": len(invalid_entries),
        "local_overlap_min": min_overlap,
        "local_overlap_avg": avg_overlap,
        "local_overlap_max": max_overlap,
        "direct_lx_alias_safe": (
            bool(cores)
            and not invalid_entries
            and min_overlap == 1.0
            and max_overlap == 1.0
        ),
        "requires_remote_lx_fetch": not (
            bool(cores)
            and not invalid_entries
            and min_overlap == 1.0
            and max_overlap == 1.0
        ),
        "sample_core_regions": samples,
    }


def _apply_lx_boundary_stitch_prototype(code_dir: Path) -> dict[str, Any]:
    """Patch one DDL bridge boundary to share the consumer's LX address map."""

    sdsc_files = sorted(code_dir.glob("sdsc_*.json"))
    bridge_files = [path for path in sdsc_files if "_ddl_bridge" in path.name]
    if not bridge_files:
        if os.environ.get("SPYRE_RESTICKIFY_STOCK_LX_ALIAS", "0") == "1":
            return _apply_stock_restickify_lx_alias_prototype(code_dir, sdsc_files)
        return {"status": "not-applicable", "reason": "no-ddl-bridge"}
    if len(bridge_files) != 1:
        return {
            "status": "not-applicable",
            "reason": "expected-one-ddl-bridge",
            "bridge_count": len(bridge_files),
        }

    bridge_path = bridge_files[0]
    try:
        consumer_path = sdsc_files[sdsc_files.index(bridge_path) + 1]
        producer_path = sdsc_files[sdsc_files.index(bridge_path) - 1]
    except (ValueError, IndexError):
        return {"status": "not-applicable", "reason": "missing-consumer-after-bridge"}

    bridge_payload = _read_json_file(bridge_path)
    producer_payload = _read_json_file(producer_path)
    consumer_payload = _read_json_file(consumer_path)
    _, bridge_dsc = _single_payload_dsc(bridge_payload)
    _, producer_dsc = _single_payload_dsc(producer_payload)
    _, consumer_dsc = _single_payload_dsc(consumer_payload)
    bridge_input_name = _single_input_ds_name(bridge_dsc)
    bridge_output_name = _single_output_ds_name(bridge_dsc)
    consumer_lds_idx = _lds_index_by_name(consumer_dsc, bridge_output_name)
    if consumer_lds_idx is None:
        return {
            "status": "not-applicable",
            "reason": "consumer-ds-name-not-found",
            "bridge_output": bridge_output_name,
        }

    debug_root = code_dir / "debug"
    shutil.rmtree(debug_root, ignore_errors=True)
    discover = _run_dxp_bundle(code_dir, cwd=code_dir, verbose=True)
    producer_debug = debug_root / producer_path.stem / f"{producer_path.stem}.out.out.json"
    consumer_debug = debug_root / consumer_path.stem / f"{consumer_path.stem}.out.out.json"
    if not consumer_debug.exists():
        return {
            "status": "not-applicable",
            "reason": "missing-consumer-debug-json",
            "consumer_debug": str(consumer_debug),
            "dxp_returncode": discover.returncode,
        }

    bridge_debug = debug_root / bridge_path.stem / f"{bridge_path.stem}.out.out.json"
    base_override = os.environ.get("SPYRE_RESTICKIFY_LX_BOUNDARY_BASE")
    match_consumer = os.environ.get("SPYRE_RESTICKIFY_LX_BOUNDARY_MATCH_CONSUMER") == "1"
    consumer_lx_start = _consumer_lxlu_start_payload(
        _read_json_file(consumer_debug),
        lds_idx=consumer_lds_idx,
    )
    if (
        consumer_lx_start is not None
        and os.environ.get("SPYRE_RESTICKIFY_LX_BOUNDARY_COLLAPSE_CORELETS") == "1"
    ):
        consumer_lx_start = _collapse_start_payload_to_one_corelet(consumer_lx_start)
    producer_output_indices = _compute_output_indices(producer_dsc)
    producer_lx_start = None
    if producer_debug.exists() and producer_output_indices:
        producer_lx_start = _producer_lxsu_start_payload(
            _read_json_file(producer_debug),
            lds_idx=producer_output_indices[0],
        )
        if (
            producer_lx_start is not None
            and os.environ.get("SPYRE_RESTICKIFY_LX_BOUNDARY_COLLAPSE_CORELETS") == "1"
        ):
            producer_lx_start = _collapse_start_payload_to_one_corelet(
                producer_lx_start
            )
    bridge_lx_start = None
    bridge_lx_start_source = ""
    if base_override:
        bridge_lx_start = _constant_lx_start_payload(
            num_cores=_core_factor(bridge_payload),
            base=int(base_override),
        )
        bridge_lx_start_source = "constant-override"
    elif match_consumer and consumer_lx_start is not None:
        bridge_lx_start = consumer_lx_start
        bridge_lx_start_source = "consumer-lxlu-input"
    elif bridge_debug.exists():
        bridge_lx_start = _bridge_output_lxsu_start_payload(_read_json_file(bridge_debug))
        bridge_lx_start_source = "bridge-lxsu-output"
    if bridge_lx_start is None:
        return {
            "status": "not-applicable",
            "reason": "missing-bridge-lxsu-output",
            "bridge_debug": str(bridge_debug),
        }

    bridge_changed = False
    input_override = os.environ.get("SPYRE_RESTICKIFY_LX_BOUNDARY_INPUT_BASE")
    bridge_input_start = None
    if input_override:
        bridge_input_start = _constant_lx_start_payload(
            num_cores=_core_factor(bridge_payload),
            base=int(input_override),
        )
        _patch_bridge_input_lx_map(bridge_payload, bridge_input_name, bridge_input_start)
        bridge_changed = True
    elif match_consumer and producer_lx_start is not None:
        bridge_input_start = producer_lx_start
        _patch_bridge_input_lx_map(bridge_payload, bridge_input_name, bridge_input_start)
        bridge_changed = True
    elif os.environ.get("SPYRE_RESTICKIFY_LX_BOUNDARY_PATCH_PRODUCER") == "1":
        bridge_input_start = _constant_lx_start_payload(
            num_cores=_core_factor(bridge_payload),
            base=0,
        )
        if not producer_output_indices:
            return {
                "status": "not-applicable",
                "reason": "missing-producer-output",
                "producer": producer_path.name,
            }
        _patch_lx_allocation_by_index(
            producer_payload,
            lds_idx=producer_output_indices[0],
            start_payload=bridge_input_start,
        )
        _patch_bridge_input_lx_map(bridge_payload, bridge_input_name, bridge_input_start)
        _write_json_file(producer_path, producer_payload)
        bridge_changed = True

    if base_override or (match_consumer and consumer_lx_start is not None):
        _patch_bridge_output_lx_map(bridge_payload, bridge_output_name, bridge_lx_start)
        bridge_changed = True
    if bridge_changed:
        _write_json_file(bridge_path, bridge_payload)
    _patch_consumer_input_lx_map(
        consumer_payload,
        bridge_output_name,
        consumer_lds_idx,
        bridge_lx_start,
    )
    _force_consumer_corelets(consumer_payload, factor=_corelet_factor(bridge_lx_start))
    _write_json_file(consumer_path, consumer_payload)

    final_verbose = os.environ.get("SPYRE_RESTICKIFY_LX_BOUNDARY_STITCH_DEBUG") == "1"
    final = _run_dxp_bundle(code_dir, cwd=code_dir, verbose=final_verbose)
    if final.returncode != 0:
        raise RuntimeError(
            "stage77 LX boundary stitch DXP rerun failed:\n"
            + final.stdout[-4000:]
            + final.stderr[-4000:]
        )

    return {
        "status": "patched",
        "bridge": bridge_path.name,
        "consumer": consumer_path.name,
        "bridge_input": bridge_input_name,
        "bridge_output": bridge_output_name,
        "consumer_lds_idx": consumer_lds_idx,
        "bridge_input_unique_starts": (
            _unique_start_values(bridge_input_start) if bridge_input_start else []
        ),
        "consumer_lx_unique_starts": (
            _unique_start_values(consumer_lx_start) if consumer_lx_start else []
        ),
        "producer_lx_unique_starts": (
            _unique_start_values(producer_lx_start) if producer_lx_start else []
        ),
        "bridge_lx_start_source": bridge_lx_start_source,
        "bridge_lx_unique_starts": _unique_start_values(bridge_lx_start),
        "bridge_lx_corelet_factor": _corelet_factor(bridge_lx_start),
    }


def _apply_stock_restickify_lx_alias_prototype(
    code_dir: Path,
    sdsc_files: list[Path],
) -> dict[str, Any]:
    """Patch a stock ReStickifyOpHBM triplet to use LX aliases.

    This keeps Deeptools' stock restickify DDL/lowering contract, but rewires
    the adjacent producer output, restickify input/output, and consumer input
    to LX-only allocations.  It is intentionally a probe-only path.
    """

    restickify_files: list[Path] = []
    for path in sdsc_files:
        try:
            payload = _read_json_file(path)
            _, dsc = _single_payload_dsc(payload)
        except Exception:  # noqa: BLE001
            continue
        op_names = [
            str(op.get("opFuncName", "")) for op in dsc.get("computeOp_", []) or []
        ]
        if any(name == "ReStickifyOpHBM" for name in op_names):
            restickify_files.append(path)
    if not restickify_files:
        return {"status": "not-applicable", "reason": "no-stock-restickify"}
    if len(restickify_files) != 1:
        return {
            "status": "not-applicable",
            "reason": "expected-one-stock-restickify",
            "restickify_count": len(restickify_files),
        }

    restickify_path = restickify_files[0]
    try:
        restickify_index = sdsc_files.index(restickify_path)
        producer_path = sdsc_files[restickify_index - 1]
        consumer_path = sdsc_files[restickify_index + 1]
    except (ValueError, IndexError):
        return {"status": "not-applicable", "reason": "missing-adjacent-sdsc"}

    producer_payload = _read_json_file(producer_path)
    restickify_payload = _read_json_file(restickify_path)
    consumer_payload = _read_json_file(consumer_path)
    producer_root, producer_dsc = _single_payload_dsc(producer_payload)
    restickify_root, restickify_dsc = _single_payload_dsc(restickify_payload)
    _, consumer_dsc = _single_payload_dsc(consumer_payload)

    restickify_input_idx = _first_compute_input_index(restickify_dsc)
    restickify_output_idx = _first_compute_output_index(restickify_dsc)
    restickify_input_hbm = _base_address(
        _alloc_start_map(restickify_dsc, lds_idx=restickify_input_idx, component="hbm")
    )
    restickify_output_hbm = _base_address(
        _alloc_start_map(restickify_dsc, lds_idx=restickify_output_idx, component="hbm")
    )
    producer_output_idx = _find_matching_lds_by_hbm_base(
        producer_dsc,
        candidate_indices=_compute_output_indices(producer_dsc),
        target_base=restickify_input_hbm,
    )
    consumer_input_idx = _find_matching_lds_by_hbm_base(
        consumer_dsc,
        candidate_indices=_compute_input_indices(consumer_dsc),
        target_base=restickify_output_hbm,
    )

    debug_root = code_dir / "debug"
    shutil.rmtree(debug_root, ignore_errors=True)
    discover = _run_dxp_bundle(code_dir, cwd=code_dir, verbose=True)
    producer_debug = (
        debug_root
        / producer_path.stem
        / f"{producer_path.stem}.out.out.json"
    )
    consumer_debug = (
        debug_root
        / consumer_path.stem
        / f"{consumer_path.stem}.out.out.json"
    )
    if discover.returncode != 0 or not producer_debug.exists() or not consumer_debug.exists():
        return {
            "status": "not-applicable",
            "reason": "debug-discovery-failed",
            "dxp_returncode": discover.returncode,
            "producer_debug": str(producer_debug),
            "consumer_debug": str(consumer_debug),
        }

    producer_start = _producer_lxsu_start_payload(
        _read_json_file(producer_debug),
        lds_idx=producer_output_idx,
    )
    consumer_start = _consumer_lxlu_start_payload(
        _read_json_file(consumer_debug),
        lds_idx=consumer_input_idx,
    )
    if producer_start is None or consumer_start is None:
        return {
            "status": "not-applicable",
            "reason": "missing-debug-lx-address",
            "producer_has_start": producer_start is not None,
            "consumer_has_start": consumer_start is not None,
        }

    ownership_before = _stock_lx_alias_ownership_summary(
        producer_root=producer_root,
        producer_dsc=producer_dsc,
        restickify_root=restickify_root,
        restickify_dsc=restickify_dsc,
    )
    copy_producer_map = (
        os.environ.get("SPYRE_RESTICKIFY_STOCK_LX_ALIAS_COPY_PRODUCER_MAP", "1")
        != "0"
        and producer_root.get("coreIdToWkSlice_")
    )
    if copy_producer_map:
        restickify_root_after = copy.deepcopy(restickify_root)
        restickify_root_after["coreIdToWkSlice_"] = copy.deepcopy(
            producer_root["coreIdToWkSlice_"]
        )
    else:
        restickify_root_after = restickify_root
    ownership_after = _stock_lx_alias_ownership_summary(
        producer_root=producer_root,
        producer_dsc=producer_dsc,
        restickify_root=restickify_root_after,
        restickify_dsc=restickify_dsc,
    )
    if (
        os.environ.get("SPYRE_RESTICKIFY_STOCK_LX_ALIAS_REQUIRE_SAFE", "0") == "1"
        and not ownership_after["direct_lx_alias_safe"]
    ):
        return {
            "status": "not-applicable",
            "reason": "stock-lx-alias-ownership-mismatch",
            "diagnostic_only": True,
            "ownership_before_core_map_copy": ownership_before,
            "ownership_after_core_map_copy": ownership_after,
        }

    # The discovered maps are post-corelet-split.  The pre-scheduler SDSC
    # contract expects one corelet unless corelet split has already run, so use
    # the corelet-0 address per core by default.
    if os.environ.get("SPYRE_RESTICKIFY_STOCK_LX_ALIAS_COLLAPSE", "1") != "0":
        producer_start = _collapse_start_payload_to_one_corelet(producer_start)
        consumer_start = _collapse_start_payload_to_one_corelet(consumer_start)

    _patch_lx_allocation_by_index(
        producer_payload,
        lds_idx=producer_output_idx,
        start_payload=producer_start,
    )
    _patch_lx_allocation_by_index(
        restickify_payload,
        lds_idx=restickify_input_idx,
        start_payload=producer_start,
    )
    input_layout_mode = os.environ.get(
        "SPYRE_RESTICKIFY_STOCK_LX_ALIAS_INPUT_LAYOUT", "keep"
    )
    if input_layout_mode != "keep":
        metadata_keys = {
            "layout": ("layoutDimOrder_",),
            "layout-max": ("layoutDimOrder_", "maxDimSizes_"),
            "producer": ("layoutDimOrder_", "maxDimSizes_", "coordinates_"),
        }.get(input_layout_mode)
        if metadata_keys is None:
            raise ValueError(
                "SPYRE_RESTICKIFY_STOCK_LX_ALIAS_INPUT_LAYOUT must be "
                "keep, layout, layout-max, or producer"
            )
        _copy_allocation_shape_metadata(
            src_dsc=producer_dsc,
            src_lds_idx=producer_output_idx,
            dst_dsc=restickify_dsc,
            dst_lds_idx=restickify_input_idx,
            keys=metadata_keys,
        )
    _patch_lx_allocation_by_index(
        restickify_payload,
        lds_idx=restickify_output_idx,
        start_payload=consumer_start,
    )
    # The stock restickify template can compile this LX-alias shape when the
    # restickify's physical ownership follows the producer.  Keep the selected
    # split factors untouched; this is only a prototype core-map alias.
    if copy_producer_map:
        restickify_root["coreIdToWkSlice_"] = copy.deepcopy(
            producer_root["coreIdToWkSlice_"]
        )
    _patch_lx_allocation_by_index(
        consumer_payload,
        lds_idx=consumer_input_idx,
        start_payload=consumer_start,
    )

    # Remove stale generated artifacts from the discovery pass before rerunning
    # DXP on the patched JSON files.
    for child in ("debug", "execute", "loadprogram_to_device", "profile", "logs"):
        shutil.rmtree(code_dir / child, ignore_errors=True)

    _write_json_file(producer_path, producer_payload)
    _write_json_file(restickify_path, restickify_payload)
    _write_json_file(consumer_path, consumer_payload)

    final_verbose = os.environ.get("SPYRE_RESTICKIFY_STOCK_LX_ALIAS_DEBUG", "0") == "1"
    final = _run_dxp_bundle(code_dir, cwd=code_dir, verbose=final_verbose)
    if final.returncode != 0:
        raise RuntimeError(
            "stock LX alias DXP rerun failed:\n"
            + final.stdout[-4000:]
            + final.stderr[-4000:]
        )

    return {
        "status": "patched-stock-lx-alias",
        "producer": producer_path.name,
        "restickify": restickify_path.name,
        "consumer": consumer_path.name,
        "producer_output_lds_idx": producer_output_idx,
        "restickify_input_lds_idx": restickify_input_idx,
        "restickify_output_lds_idx": restickify_output_idx,
        "consumer_input_lds_idx": consumer_input_idx,
        "producer_lx_unique_starts": _unique_start_values(producer_start),
        "consumer_lx_unique_starts": _unique_start_values(consumer_start),
        "producer_lx_corelet_factor": _corelet_factor(producer_start),
        "consumer_lx_corelet_factor": _corelet_factor(consumer_start),
        "diagnostic_only": True,
        "production_candidate": False,
        "prototype_warning": (
            "stock LX aliasing is a diagnostic probe; correctness requires "
            "per-core producer ownership to match the restickify input need, "
            "or an explicit remote-LX fetch path"
        ),
        "copied_producer_core_map": bool(copy_producer_map),
        "ownership_before_core_map_copy": ownership_before,
        "ownership_after_core_map_copy": ownership_after,
    }


def _run_dxp_bundle(
    code_dir: Path,
    *,
    cwd: Path,
    verbose: bool,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if verbose:
        env["DXP_VERBOSE"] = "1"
    else:
        env.pop("DXP_VERBOSE", None)
    shim = code_dir / "librestickify_ddl_preddc_shim.so"
    if shim.exists():
        old_preload = env.get("LD_PRELOAD")
        env["LD_PRELOAD"] = str(shim) if not old_preload else f"{shim}:{old_preload}"
    return subprocess.run(
        ["dxp_standalone", "--bundle", "-d", str(code_dir)],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _single_payload_dsc(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = next(iter(payload.values()))
    return root, next(iter(root["dscs_"][0].values()))


def _single_output_ds_name(dsc: dict[str, Any]) -> str:
    output_label = dsc["computeOp_"][0]["outputLabeledDs"][0]
    _, idx = output_label.rsplit("-idx", 1)
    output_idx = int(idx)
    for lds in dsc["labeledDs_"]:
        if int(lds["ldsIdx_"]) == output_idx:
            return str(lds["dsName_"])
    raise ValueError(f"could not resolve output LDS {output_idx}")


def _single_input_ds_name(dsc: dict[str, Any]) -> str:
    input_label = dsc["computeOp_"][0]["inputLabeledDs"][0]
    _, idx = input_label.rsplit("-idx", 1)
    input_idx = int(idx)
    for lds in dsc["labeledDs_"]:
        if int(lds["ldsIdx_"]) == input_idx:
            return str(lds["dsName_"])
    raise ValueError(f"could not resolve input LDS {input_idx}")


def _lds_index_by_name(dsc: dict[str, Any], ds_name: str) -> int | None:
    for lds in dsc["labeledDs_"]:
        if lds.get("dsName_") == ds_name:
            return int(lds["ldsIdx_"])
    return None


def _consumer_lxlu_start_payload(
    payload: dict[str, Any],
    *,
    lds_idx: int,
) -> dict[str, Any] | None:
    _, dsc = _single_payload_dsc(payload)
    wanted = f"transfer_lds{lds_idx}_src:lxlu_dst:sfp"
    for node in dsc.get("scheduleTree_", []):
        if node.get("name_") != wanted:
            continue
        start = node.get("srcLdsAndLoopOffsets_", {}).get("startAddr_")
        if isinstance(start, dict) and start.get("data_"):
            return start
    return None


def _producer_lxsu_start_payload(
    payload: dict[str, Any],
    *,
    lds_idx: int,
) -> dict[str, Any] | None:
    _, dsc = _single_payload_dsc(payload)
    wanted = f"transfer_lds{lds_idx}_src:sfp_dst:lxsu"
    for node in dsc.get("scheduleTree_", []):
        if node.get("name_") != wanted:
            continue
        offsets = node.get("dstLdsAndLoopOffsets_", [])
        for offset in offsets:
            start = offset.get("startAddr_") if isinstance(offset, dict) else None
            if isinstance(start, dict) and start.get("data_"):
                return start
    return None


def _bridge_output_lxsu_start_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    _, dsc = _single_payload_dsc(payload)
    for node in dsc.get("scheduleTree_", []):
        name = node.get("name_", "")
        if not (name.startswith("transfer_lds") and "_src:ptrow0_dst:lxsu" in name):
            continue
        offsets = node.get("dstLdsAndLoopOffsets_", [])
        for offset in offsets:
            start = offset.get("startAddr_") if isinstance(offset, dict) else None
            if isinstance(start, dict) and start.get("data_"):
                return start
    return None


def _force_consumer_corelets(payload: dict[str, Any], *, factor: int) -> None:
    root, dsc = _single_payload_dsc(payload)
    root["coreletFoldProp_"] = {"factor_": factor, "label_": "corelet"}
    dsc["numCoreletsUsed_"] = factor
    dsc["numCoreletsUsed_DSC2_"] = factor


def _patch_bridge_output_lx_map(
    payload: dict[str, Any],
    output_name: str,
    start_payload: dict[str, Any],
) -> None:
    root, dsc = _single_payload_dsc(payload)
    corelet_factor = _corelet_factor(start_payload)
    root["coreletFoldProp_"] = {"factor_": corelet_factor, "label_": "corelet"}
    dsc["numCoreletsUsed_"] = corelet_factor
    dsc["numCoreletsUsed_DSC2_"] = corelet_factor
    output_idx = _lds_index_by_name(dsc, output_name)
    if output_idx is None:
        raise ValueError(f"bridge output {output_name!r} not found")
    alloc_name = f"allocate_{output_name}_lx"
    for lds in dsc["labeledDs_"]:
        if int(lds["ldsIdx_"]) == output_idx:
            lx_meta = dict(lds.get("memOrg_", {}).get("lx", {}))
            lx_meta["isPresent"] = 1
            lx_meta["allocateNode_"] = alloc_name
            lds["memOrg_"] = {"lx": lx_meta}
    for node in dsc.get("scheduleTree_", []):
        if node.get("nodeType_") == "allocate" and int(node.get("ldsIdx_", -1)) == output_idx:
            node["name_"] = alloc_name
            node["component_"] = "lx"
            node["startAddressCoreCorelet_"] = start_payload
        if node.get("nodeType_") == "transfer" and node.get("name_") == (
            "transfer_lds1_src:lx_dst:no_component_lx_local"
        ):
            src = node.get("srcLdsAndLoopOffsets_")
            if isinstance(src, dict):
                src["startAddr_"] = start_payload


def _patch_bridge_input_lx_map(
    payload: dict[str, Any],
    input_name: str,
    start_payload: dict[str, Any],
) -> None:
    root, dsc = _single_payload_dsc(payload)
    corelet_factor = _corelet_factor(start_payload)
    root["coreletFoldProp_"] = {"factor_": corelet_factor, "label_": "corelet"}
    dsc["numCoreletsUsed_"] = corelet_factor
    dsc["numCoreletsUsed_DSC2_"] = corelet_factor
    input_idx = _lds_index_by_name(dsc, input_name)
    if input_idx is None:
        raise ValueError(f"bridge input {input_name!r} not found")
    alloc_name = f"allocate_{input_name}_lx"
    for lds in dsc["labeledDs_"]:
        if int(lds["ldsIdx_"]) == input_idx:
            lx_meta = dict(lds.get("memOrg_", {}).get("lx", {}))
            lx_meta["isPresent"] = 1
            lx_meta["allocateNode_"] = alloc_name
            lds["memOrg_"] = {"lx": lx_meta}
    for node in dsc.get("scheduleTree_", []):
        if node.get("nodeType_") == "allocate" and int(node.get("ldsIdx_", -1)) == input_idx:
            node["name_"] = alloc_name
            node["component_"] = "lx"
            node["startAddressCoreCorelet_"] = start_payload
        if node.get("nodeType_") == "transfer" and node.get("name_") == (
            "transfer_lds0_src:no_component_dst:lx_lx_local"
        ):
            offsets = node.get("dstLdsAndLoopOffsets_", [])
            for offset in offsets:
                if isinstance(offset, dict):
                    offset["startAddr_"] = start_payload


def _patch_consumer_input_lx_map(
    payload: dict[str, Any],
    input_name: str,
    lds_idx: int,
    start_payload: dict[str, Any],
) -> None:
    _, dsc = _single_payload_dsc(payload)
    allocate_name = f"allocate-{input_name}_lx"
    ddl_like = os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_DDL_LIKE_INPUT", "0") == "1"
    ddl_lx_size = int(
        os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_DDL_LX_SIZE", "262144")
    )
    preserve_role = (
        os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_PRESERVE_CONSUMER_ROLE", "1")
        != "0"
    )
    primary = dsc.setdefault("primaryDsInfo_", {})
    if not preserve_role and "INPUT" not in primary and "OUTPUT" in primary:
        primary["INPUT"] = copy.deepcopy(primary["OUTPUT"])
    for lds in dsc["labeledDs_"]:
        if int(lds["ldsIdx_"]) == lds_idx:
            original_mem = lds.get("memOrg_", {}) or {}
            lx_meta = dict(original_mem.get("lx", {}))
            lx_meta["isPresent"] = 1
            lx_meta["allocateNode_"] = allocate_name
            if ddl_like:
                lx_meta.update(
                    {
                        "isPadded": 0,
                        "isZeroPadded": 0,
                        "zpadGapFront": [0, 0],
                        "gapPerDim": {},
                        "dsOffset": 0,
                    }
                )
            lds["memOrg_"] = {"lx": lx_meta}
            lds["hbmStartAddress_"] = -1
            if ddl_like:
                lds["segment_"] = "stack"
                lds["isFirstUse_"] = 0
                lds["isExternal_"] = 0
                lds["dataTransfers_"] = []
                lds["lxStartAddress_"] = -1
                lds["hbmSize_"] = 18_446_744_073_709_551_615
                lds["lxSize_"] = ddl_lx_size
                lds["lxBufferSize_"] = 18_446_744_073_709_551_615
                lds["totSlicesPerDim_"] = {}
            else:
                lds["hbmSize_"] = 0
                if int(lds.get("lxSize_", 0) or 0) <= 0:
                    lds["lxSize_"] = 2_147_483_647
                if int(lds.get("lxBufferSize_", 0) or 0) <= 0:
                    lds["lxBufferSize_"] = 2_147_483_647
            lds["coreStateInit_"] = _constant_lx_core_state_init(start_payload)
            if not preserve_role:
                lds["dsType_"] = "INPUT"
    for node in dsc.get("scheduleTree_", []):
        if node.get("nodeType_") == "allocate" and int(node.get("ldsIdx_", -1)) == lds_idx:
            node["name_"] = allocate_name
            node["component_"] = "lx"
            node["startAddressCoreCorelet_"] = start_payload


def _constant_lx_core_state_init(start_payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = start_payload.get("data_", {}) or {}
    core_values: dict[int, dict[int, int]] = {}
    for key, raw_value in data.items():
        try:
            core_str, corelet_str, _time_str = key.strip("[]").split(",")
            core = int(core_str.strip())
            corelet = int(corelet_str.strip())
        except ValueError:
            continue
        core_values.setdefault(core, {})[corelet] = int(raw_value)
    requested_corelets = int(
        os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_CORESTATE_CORELETS", "1")
    )
    corelet_stride = int(
        os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_CORESTATE_CORELET_STRIDE", "8192")
    )
    for values in core_values.values():
        if not values:
            continue
        base = values[min(values)]
        for corelet in range(requested_corelets):
            values.setdefault(corelet, base + corelet * corelet_stride)
    return [
        {
            "ebrInit_": -1,
            "gtr_": {
                "type": "multicast",
                "id": 18446744073709551615,
                "count": 0,
                "sharers": 0,
                "groupInfo_": {},
            },
            "condGtr_": [],
            "lbrInit_": [
                core_values[core][corelet]
                for corelet in sorted(core_values[core])
            ],
            "gapPerDim_": {},
            "lxSizeWithGaps_": 2_147_483_647,
            "lbrInitForwardGap_": 0,
        }
        for core in sorted(core_values)
    ]


def _corelet_factor(start_payload: dict[str, Any]) -> int:
    attrs = start_payload.get("dim_prop_attr", [])
    if len(attrs) > 1:
        return int(attrs[1].get("factor_", 1) or 1)
    return 1


def _unique_start_values(start_payload: dict[str, Any]) -> list[int]:
    return sorted({int(value) for value in start_payload.get("data_", {}).values()})


def _collapse_start_payload_to_one_corelet(start_payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the corelet-0 address for each core and rewrite cardinality to 32x1x1."""

    collapsed = copy.deepcopy(start_payload)
    attrs = collapsed.get("dim_prop_attr", [])
    if len(attrs) >= 2:
        attrs[1] = {**attrs[1], "factor_": 1, "label_": "corelet"}
    funcs = collapsed.get("dim_prop_func", [])
    if len(funcs) >= 2:
        funcs[1] = {"Const": {}}
    data = collapsed.get("data_", {})
    core_to_value: dict[int, str] = {}
    for key, value in data.items():
        try:
            core_str, corelet_str, _time_str = key.strip("[]").split(",")
            core = int(core_str.strip())
            corelet = int(corelet_str.strip())
        except ValueError:
            continue
        if corelet == 0:
            core_to_value[core] = str(value)
    if not core_to_value:
        return collapsed
    collapsed["data_"] = {
        f"[{core}, 0, 0]": core_to_value[core] for core in sorted(core_to_value)
    }
    return collapsed


def _core_factor(payload: dict[str, Any]) -> int:
    root = next(iter(payload.values()))
    return int(root.get("coreFoldProp_", {}).get("factor_", 32) or 32)


def _constant_lx_start_payload(*, num_cores: int, base: int) -> dict[str, Any]:
    return {
        "dim_prop_func": [{"Map": {}}, {"Const": {}}, {"Const": {}}],
        "dim_prop_attr": [
            {"factor_": num_cores, "label_": "core"},
            {"factor_": 1, "label_": "corelet"},
            {"factor_": 1, "label_": "time"},
        ],
        "data_": {f"[{core}, 0, 0]": str(base) for core in range(num_cores)},
    }


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
            "ring_source_kinds": {},
            "ring_locality_assertions": {},
            "ring_locality_certified_rows": 0,
            "ring_certified_byte_hops": 0,
            "ring_exact_rows": 0,
            "ring_skipped_rows": 0,
            "ring_entries": [],
        }

    entries = [json.loads(line) for line in path.read_text().splitlines() if line]
    total_bytes = sum(int(entry.get("bytes_moved") or 0) for entry in entries)
    total_byte_hops = sum(int(entry.get("byte_hops") or 0) for entry in entries)
    max_hops = max((int(entry.get("max_hops") or 0) for entry in entries), default=0)
    skip_reasons: dict[str, int] = {}
    source_kinds: dict[str, int] = {}
    locality_assertions: dict[str, int] = {}
    for entry in entries:
        reason = entry.get("skip_reason")
        if reason:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        source_kind = entry.get("source_kind")
        if source_kind:
            source_kinds[source_kind] = source_kinds.get(source_kind, 0) + 1
        locality_assertion = entry.get("locality_assertion")
        if locality_assertion:
            locality_assertions[locality_assertion] = (
                locality_assertions.get(locality_assertion, 0) + 1
            )
    return {
        "ring_rows": len(entries),
        "ring_total_bytes": total_bytes,
        "ring_total_byte_hops": total_byte_hops,
        "ring_avg_hops": total_byte_hops / total_bytes if total_bytes else 0.0,
        "ring_max_hops": max_hops,
        "ring_skip_reasons": skip_reasons,
        "ring_source_kinds": source_kinds,
        "ring_locality_assertions": locality_assertions,
        "ring_locality_certified_rows": sum(
            1 for entry in entries if entry.get("locality_certified")
        ),
        "ring_certified_byte_hops": sum(
            int(entry.get("certified_byte_hops") or 0) for entry in entries
        ),
        "ring_exact_rows": sum(1 for entry in entries if not entry.get("skip_reason")),
        "ring_skipped_rows": sum(1 for entry in entries if entry.get("skip_reason")),
        "ring_entries": entries,
    }


def _assert_close(actual: Any, expected: Any, atol: float, rtol: float) -> None:
    if isinstance(actual, tuple):
        assert isinstance(expected, tuple) and len(actual) == len(expected)
        tuple_prefix = int(os.environ.get("SPYRE_PROBE_VALIDATE_TUPLE_PREFIX", "0"))
        if tuple_prefix > 0:
            actual = actual[:tuple_prefix]
            expected = expected[:tuple_prefix]
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


def _profiler_value_ms(event: Any, attr: str) -> float:
    return float(getattr(event, attr, 0.0) or 0.0) / 1000.0


def _patch_spyre_trace_device_properties(trace_path: Path) -> bool:
    """Add minimal PrivateUse1 device metadata for acelyzer compatibility.

    PyTorch's Chrome trace exporter currently writes an empty `deviceProperties`
    list for the Spyre PrivateUse1 profiler. `aiu-trace-analyzer` expects one
    device entry for torch-profile inputs, so add a conservative AIU entry when
    the field is present but empty.
    """
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(trace, dict) or trace.get("deviceProperties") != []:
        return False

    cores = int(os.environ.get("SENCORES", "32"))
    trace["deviceProperties"] = [
        {
            "id": 0,
            "name": "IBM Spyre AIU",
            "type": "PrivateUse1",
            "totalGlobalMem": 128 * 1024**3,
            "multiProcessorCount": cores,
            "numSms": cores,
            "computeMajor": 0,
            "computeMinor": 0,
        }
    ]
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    return True


def _profile_compiled(
    compiled_fn: Callable[..., Any],
    dev_args: tuple[Any, ...],
    warmup: int,
    iters: int,
    output_dir: Path,
    profile_memory: bool,
    with_stack: bool,
) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity

    privateuse1 = getattr(ProfilerActivity, "PrivateUse1", None)
    if privateuse1 is None:
        raise RuntimeError("torch.profiler.ProfilerActivity.PrivateUse1 is unavailable")

    for _ in range(warmup):
        compiled_fn(*dev_args)
    _sync()

    output_dir.mkdir(parents=True, exist_ok=True)
    events_json = output_dir / "torch_profiler_events.json"
    events_csv = output_dir / "torch_profiler_events.csv"
    trace_path = output_dir / "torch_profiler_trace.json"

    profiler = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, privateuse1],
        record_shapes=True,
        profile_memory=profile_memory,
        with_stack=with_stack,
        acc_events=True,
    )
    profiler.start()
    for _ in range(iters):
        compiled_fn(*dev_args)
        profiler.step()
    _sync()
    profiler.stop()

    trace_error = ""
    device_properties_patched = False
    try:
        profiler.export_chrome_trace(str(trace_path))
        device_properties_patched = _patch_spyre_trace_device_properties(trace_path)
    except Exception as exc:
        trace_error = f"{type(exc).__name__}: {exc}"

    events: list[dict[str, Any]] = []
    for event in profiler.key_averages():
        key = str(getattr(event, "key", ""))
        count = int(getattr(event, "count", 0) or 0)
        device_total_ms = _profiler_value_ms(event, "device_time_total")
        self_cpu_total_ms = _profiler_value_ms(event, "self_cpu_time_total")
        cpu_total_ms = _profiler_value_ms(event, "cpu_time_total")
        if not (device_total_ms or self_cpu_total_ms or cpu_total_ms):
            continue
        events.append(
            {
                "key": key,
                "count": count,
                "device_time_total_ms": device_total_ms,
                "device_time_avg_ms": device_total_ms / count if count else 0.0,
                "self_cpu_time_total_ms": self_cpu_total_ms,
                "cpu_time_total_ms": cpu_total_ms,
            }
        )
    events.sort(key=lambda item: item["device_time_total_ms"], reverse=True)

    events_json.write_text(json.dumps(events, indent=2, sort_keys=True), encoding="utf-8")
    if events:
        with events_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(events[0].keys()))
            writer.writeheader()
            writer.writerows(events)

    device_events = [event for event in events if event["device_time_total_ms"] > 0]
    interesting_tokens = ("restickify", "ReStickify", "sdsc")
    interesting_events = [
        event
        for event in device_events
        if any(token.lower() in event["key"].lower() for token in interesting_tokens)
    ]
    raw_events = profiler.events()
    return {
        "profiler_event_count": len(events),
        "profiler_device_event_count": len(device_events),
        "profiler_total_device_ms": sum(
            _profiler_value_ms(event, "device_time_total") for event in raw_events
        ),
        "profiler_total_self_cpu_ms": sum(
            _profiler_value_ms(event, "self_cpu_time_total") for event in raw_events
        ),
        "profiler_interesting_event_count": len(interesting_events),
        "profiler_trace_path": str(trace_path) if not trace_error else "",
        "profiler_trace_error": trace_error,
        "profiler_device_properties_patched": device_properties_patched,
        "profiler_events_json": str(events_json),
        "profiler_events_csv": str(events_csv) if events else "",
        "profiler_top_device_events": device_events[:20],
        "profiler_interesting_events": interesting_events,
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
    torch_profiler_dir: Path | None = None,
    torch_profiler_memory: bool = False,
    torch_profiler_with_stack: bool = False,
    sync_after_kernel: bool = False,
    kernel_launch_log_path: Path | None = None,
    copy_kernel_code_dir: Path | None = None,
    lx_boundary_stitch_prototype: bool = False,
    lx_split_dataop_prototype: bool = False,
    lx_bridge_same_artifact_splice: bool = False,
    skip_kernel_launch: bool = False,
) -> dict[str, Any]:
    args, shape_label = case.input_builder(size, dtype)
    dev_args = tuple(arg.to(device) if hasattr(arg, "to") else arg for arg in args)

    import torch_spyre._inductor.insert_restickify as insert_restickify

    insert_restickify.restickify_plan = {}
    previous_capture = os.environ.get("SPYRE_CAPTURE_RESTICKIFY_PLAN")
    previous_ring = os.environ.get("SPYRE_RESTICKIFY_RING_TELEMETRY")
    previous_ring_jsonl = os.environ.get("SPYRE_RESTICKIFY_RING_TELEMETRY_JSONL")
    previous_context = os.environ.get("SPYRE_TELEMETRY_CONTEXT")
    previous_allow_uncertified_descriptor = os.environ.get(
        "SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED"
    )
    spyre_config = None
    previous_config_ring = None
    previous_config_ring_jsonl = None
    os.environ["SPYRE_CAPTURE_RESTICKIFY_PLAN"] = "1"
    os.environ["SPYRE_TELEMETRY_CONTEXT"] = json.dumps(
        {
            "case": case.name,
            "scenario": case.scenario,
            "source_hint": case.source_hint,
            "size": size,
            "shape": shape_label,
        },
        sort_keys=True,
    )
    if lx_bridge_same_artifact_splice:
        os.environ["SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED"] = "1"
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
        with _kernel_launch_debug(
            sync_after_kernel=sync_after_kernel,
            log_path=kernel_launch_log_path,
            copy_code_dir_root=copy_kernel_code_dir,
            lx_boundary_stitch_prototype=lx_boundary_stitch_prototype,
            lx_split_dataop_prototype=lx_split_dataop_prototype,
            lx_bridge_same_artifact_splice=lx_bridge_same_artifact_splice,
            skip_kernel_launch=skip_kernel_launch,
        ):
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

            profiler_summary = {}
            if torch_profiler_dir is not None:
                profiler_summary = _profile_compiled(
                    compiled,
                    dev_args,
                    warmup,
                    iters,
                    torch_profiler_dir,
                    profile_memory=torch_profiler_memory,
                    with_stack=torch_profiler_with_stack,
                )

            if not skip_correctness:
                expected = case.fn(*args)
                actual = _tensor_to_cpu(result)
                _assert_close(actual, expected, atol=atol, rtol=rtol)

        kernel_launch_event_count = 0
        if kernel_launch_log_path is not None and kernel_launch_log_path.exists():
            kernel_launch_event_count = sum(
                1 for line in kernel_launch_log_path.read_text().splitlines() if line
            )

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
            "sync_after_kernel": sync_after_kernel,
            "kernel_launch_log": str(kernel_launch_log_path or ""),
            "kernel_launch_event_count": kernel_launch_event_count,
            "skip_kernel_launch": skip_kernel_launch,
            **ring_summary,
            **timing,
            **profiler_summary,
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
        if previous_context is None:
            os.environ.pop("SPYRE_TELEMETRY_CONTEXT", None)
        else:
            os.environ["SPYRE_TELEMETRY_CONTEXT"] = previous_context
        if previous_allow_uncertified_descriptor is None:
            os.environ.pop(
                "SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED",
                None,
            )
        else:
            os.environ[
                "SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED"
            ] = previous_allow_uncertified_descriptor
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
        "ring_source_kinds": {},
        "ring_exact_rows": 0,
        "ring_skipped_rows": 0,
        "ring_entries": [],
        "sync_after_kernel": False,
        "kernel_launch_log": "",
        "kernel_launch_event_count": 0,
        "profiler_event_count": 0,
        "profiler_device_event_count": 0,
        "profiler_total_device_ms": 0.0,
        "profiler_total_self_cpu_ms": 0.0,
        "profiler_interesting_event_count": 0,
        "profiler_trace_path": "",
        "profiler_trace_error": "",
        "profiler_events_json": "",
        "profiler_events_csv": "",
        "profiler_top_device_events": [],
        "profiler_interesting_events": [],
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
        "ring_source_kinds": json.dumps(row.get("ring_source_kinds", {}), sort_keys=True),
        "ring_exact_rows": row.get("ring_exact_rows", 0),
        "ring_skipped_rows": row.get("ring_skipped_rows", 0),
        "sync_after_kernel": row.get("sync_after_kernel", False),
        "kernel_launch_log": row.get("kernel_launch_log", ""),
        "kernel_launch_event_count": row.get("kernel_launch_event_count", 0),
        "compile_run_ms": f"{row.get('compile_run_ms', 0.0):.3f}" if row.get("compile_run_ms") is not None else "",
        "median_ms": f"{row.get('median_ms', 0.0):.3f}" if row.get("median_ms") is not None else "",
        "p10_ms": f"{row.get('p10_ms', 0.0):.3f}" if row.get("p10_ms") is not None else "",
        "p90_ms": f"{row.get('p90_ms', 0.0):.3f}" if row.get("p90_ms") is not None else "",
        "profiler_event_count": row.get("profiler_event_count", 0),
        "profiler_device_event_count": row.get("profiler_device_event_count", 0),
        "profiler_total_device_ms": f"{row.get('profiler_total_device_ms', 0.0):.3f}",
        "profiler_total_self_cpu_ms": f"{row.get('profiler_total_self_cpu_ms', 0.0):.3f}",
        "profiler_interesting_event_count": row.get("profiler_interesting_event_count", 0),
        "profiler_trace_path": row.get("profiler_trace_path", ""),
        "profiler_trace_error": row.get("profiler_trace_error", ""),
        "profiler_events_json": row.get("profiler_events_json", ""),
        "profiler_events_csv": row.get("profiler_events_csv", ""),
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
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Capture torch.profiler PrivateUse1 events after compile.",
    )
    parser.add_argument(
        "--torch-profiler-memory",
        action="store_true",
        help="Enable torch profiler memory tracking for --torch-profiler.",
    )
    parser.add_argument(
        "--torch-profiler-with-stack",
        action="store_true",
        help="Capture Python stacks for --torch-profiler.",
    )
    parser.add_argument(
        "--sync-after-kernel",
        action="store_true",
        help="Synchronize after each generated Spyre SDSC bundle launch.",
    )
    parser.add_argument(
        "--kernel-launch-log",
        action="store_true",
        help="Write JSONL before/after events for each generated Spyre SDSC bundle launch.",
    )
    parser.add_argument(
        "--copy-kernel-code",
        action="store_true",
        help="Copy each generated Spyre SDSC bundle directory into the probe output.",
    )
    parser.add_argument(
        "--lx-boundary-stitch-prototype",
        action="store_true",
        help="Probe-only launch-time patch for DDL bridge producer/restickify/consumer LX boundary stitching.",
    )
    parser.add_argument(
        "--lx-split-dataop-prototype",
        action="store_true",
        help=(
            "Probe-only launch-time split: producer compute, address-preserving "
            "LX data-op restickify, then consumer compute."
        ),
    )
    parser.add_argument(
        "--lx-bridge-same-artifact-splice",
        action="store_true",
        help=(
            "Probe-only launch-time patch: generate an HBM-free LX bridge frame "
            "and splice it into the normal fused runtime artifact before launch."
        ),
    )
    parser.add_argument(
        "--skip-kernel-launch",
        action="store_true",
        help=(
            "Probe-only codegen mode: generate and optionally copy Spyre SDSC "
            "bundle directories, but do not launch them. Use with "
            "--skip-correctness."
        ),
    )
    parser.add_argument(
        "--validate-tuple-prefix",
        type=int,
        default=0,
        help=(
            "For tuple outputs, compare only the first N values. This is useful "
            "when a launch hook intentionally skips later kernels that only "
            "produce later tuple outputs."
        ),
    )
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
    previous_tuple_prefix = os.environ.get("SPYRE_PROBE_VALIDATE_TUPLE_PREFIX")
    if args.validate_tuple_prefix:
        os.environ["SPYRE_PROBE_VALIDATE_TUPLE_PREFIX"] = str(
            args.validate_tuple_prefix
        )
    try:
        with jsonl_path.open("w", encoding="utf-8") as jsonl:
            for size in sizes:
                for case in selected:
                    telemetry_path = (
                        output_dir / "ring_telemetry" / f"{case.name}_{size}.jsonl"
                        if args.ring_telemetry
                        else None
                    )
                    torch_profiler_dir = (
                        output_dir / "torch_profiler" / f"{case.name}_{size}"
                        if args.torch_profiler
                        else None
                    )
                    kernel_launch_log = (
                        output_dir / "kernel_launches" / f"{case.name}_{size}.jsonl"
                        if args.sync_after_kernel
                        or args.kernel_launch_log
                        or args.copy_kernel_code
                        or args.lx_boundary_stitch_prototype
                        or args.lx_split_dataop_prototype
                        or args.lx_bridge_same_artifact_splice
                        else None
                    )
                    copy_kernel_code_dir = (
                        output_dir / "kernel_code" / f"{case.name}_{size}"
                        if args.copy_kernel_code
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
                            torch_profiler_dir=torch_profiler_dir,
                            torch_profiler_memory=args.torch_profiler_memory,
                            torch_profiler_with_stack=args.torch_profiler_with_stack,
                            sync_after_kernel=args.sync_after_kernel,
                            kernel_launch_log_path=kernel_launch_log,
                            copy_kernel_code_dir=copy_kernel_code_dir,
                            lx_boundary_stitch_prototype=args.lx_boundary_stitch_prototype,
                            lx_split_dataop_prototype=args.lx_split_dataop_prototype,
                            lx_bridge_same_artifact_splice=args.lx_bridge_same_artifact_splice,
                            skip_kernel_launch=args.skip_kernel_launch,
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
                        f"byte_hops={byte_hops} "
                        f"device_events={row.get('profiler_device_event_count', 0)}"
                    )
    finally:
        if args.validate_tuple_prefix:
            if previous_tuple_prefix is None:
                os.environ.pop("SPYRE_PROBE_VALIDATE_TUPLE_PREFIX", None)
            else:
                os.environ["SPYRE_PROBE_VALIDATE_TUPLE_PREFIX"] = (
                    previous_tuple_prefix
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

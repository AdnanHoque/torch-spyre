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

"""Catalog of matmul shapes from popular vLLM-supported model families.

Generates the (model, op, M, N, K) tuples that the production planner
will see across realistic vLLM serving workloads. Output is a CSV-like
listing plus a deduplicated unique-shape view; no hardware needed.

Used downstream by the focused k_fast-essential probe to determine
the production fraction of shapes where (m, n, k>1)+kf is the
empirical optimum.

Usage:
    python tests/diag_vllm_shape_catalog.py
    python tests/diag_vllm_shape_catalog.py --sample 20  # picks 20 representative
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass


# Decode and prefill batch sizes typical for vLLM serving.
M_VALUES = (1, 32, 128, 512, 1024, 2048)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int

    # Computed dims.
    @property
    def kv_proj_out(self) -> int:
        # Combined kv_proj: K+V. (Some models split; we use combined for the
        # GQA case which is the dominant pattern in vLLM serving today.)
        return 2 * self.n_kv_heads * self.head_dim

    @property
    def q_proj_out(self) -> int:
        return self.n_heads * self.head_dim


@dataclass(frozen=True)
class DSV3Config:
    """DeepSeek V3 has MLA — different KV projection structure."""
    name: str
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    q_lora_rank: int       # MLA absorbed Q rank
    kv_lora_rank: int      # MLA absorbed KV rank
    qk_rope_dim: int = 64
    qk_nope_dim: int = 128
    v_head_dim: int = 128

    @property
    def q_a_proj_out(self) -> int:
        return self.q_lora_rank

    @property
    def q_b_proj_out(self) -> int:
        return self.n_heads * (self.qk_rope_dim + self.qk_nope_dim)

    @property
    def kv_a_proj_out(self) -> int:
        return self.kv_lora_rank + self.qk_rope_dim

    @property
    def kv_b_proj_out(self) -> int:
        return self.n_heads * (self.qk_nope_dim + self.v_head_dim)


# Standard transformer architectures (Llama-style block).
STANDARD_MODELS = [
    # Llama 3.1
    ModelConfig("Llama 3.1 8B",   4096, 14336, 32,  8, 128),
    ModelConfig("Llama 3.1 70B",  8192, 28672, 64,  8, 128),
    ModelConfig("Llama 3.1 405B", 16384, 53248, 128, 8, 128),
    # Llama 3.2 (smaller)
    ModelConfig("Llama 3.2 1B",   2048, 8192,  32,  8,  64),
    ModelConfig("Llama 3.2 3B",   3072, 8192,  24,  8, 128),
    # Mistral / Mixtral
    ModelConfig("Mistral 7B",     4096, 14336, 32,  8, 128),
    ModelConfig("Mixtral 8x7B",   4096, 14336, 32,  8, 128),
    ModelConfig("Mixtral 8x22B",  6144, 16384, 48,  8, 128),
    # Qwen 2.5
    ModelConfig("Qwen 2.5 7B",    3584, 18944, 28,  4, 128),
    ModelConfig("Qwen 2.5 14B",   5120, 13824, 40,  8, 128),
    ModelConfig("Qwen 2.5 32B",   5120, 27648, 40,  8, 128),
    ModelConfig("Qwen 2.5 72B",   8192, 29568, 64,  8, 128),
    # Phi 3 medium
    ModelConfig("Phi 3 medium",   5120, 17920, 40, 10, 128),
    # Gemma 2
    ModelConfig("Gemma 2 9B",     3584, 14336, 16,  8, 256),
    ModelConfig("Gemma 2 27B",    4608, 36864, 32, 16, 128),
]


# DeepSeek V3 / R1 (MLA architecture)
DSV3_MODELS = [
    DSV3Config(
        name="DeepSeek V3",
        hidden=7168, intermediate=18432,
        n_heads=128, n_kv_heads=128, head_dim=128,
        q_lora_rank=1536, kv_lora_rank=512,
    ),
]


@dataclass(frozen=True)
class MatmulOp:
    model: str
    op: str
    M: int
    N: int
    K: int

    def shape_tuple(self) -> tuple[int, int, int]:
        return (self.M, self.N, self.K)


def block_matmuls(cfg: ModelConfig, M: int) -> list[MatmulOp]:
    H = cfg.hidden
    I = cfg.intermediate
    Nq = cfg.q_proj_out
    Nkv = cfg.kv_proj_out
    return [
        MatmulOp(cfg.name, "q_proj",    M, Nq,  H),
        MatmulOp(cfg.name, "kv_proj",   M, Nkv, H),
        MatmulOp(cfg.name, "o_proj",    M, H,   Nq),
        MatmulOp(cfg.name, "gate_proj", M, I,   H),
        MatmulOp(cfg.name, "up_proj",   M, I,   H),
        MatmulOp(cfg.name, "down_proj", M, H,   I),
    ]


def block_matmuls_dsv3(cfg: DSV3Config, M: int) -> list[MatmulOp]:
    return [
        MatmulOp(cfg.name, "q_a_proj",  M, cfg.q_a_proj_out,  cfg.hidden),
        MatmulOp(cfg.name, "q_b_proj",  M, cfg.q_b_proj_out,  cfg.q_lora_rank),
        MatmulOp(cfg.name, "kv_a_proj", M, cfg.kv_a_proj_out, cfg.hidden),
        MatmulOp(cfg.name, "kv_b_proj", M, cfg.kv_b_proj_out, cfg.kv_lora_rank),
        MatmulOp(cfg.name, "o_proj",    M, cfg.hidden, cfg.n_heads * cfg.v_head_dim),
        MatmulOp(cfg.name, "gate_proj", M, cfg.intermediate, cfg.hidden),
        MatmulOp(cfg.name, "up_proj",   M, cfg.intermediate, cfg.hidden),
        MatmulOp(cfg.name, "down_proj", M, cfg.hidden, cfg.intermediate),
    ]


def build_catalog() -> list[MatmulOp]:
    out: list[MatmulOp] = []
    for cfg in STANDARD_MODELS:
        for M in M_VALUES:
            out.extend(block_matmuls(cfg, M))
    for cfg in DSV3_MODELS:
        for M in M_VALUES:
            out.extend(block_matmuls_dsv3(cfg, M))
    return out


def deduplicate(catalog: list[MatmulOp]) -> dict[tuple[int, int, int], list[MatmulOp]]:
    """Group ops by (M, N, K) shape — many ops/models share the same shape."""
    by_shape: dict[tuple[int, int, int], list[MatmulOp]] = {}
    for op in catalog:
        by_shape.setdefault(op.shape_tuple(), []).append(op)
    return by_shape


def sample_representative(unique_shapes: dict[tuple[int, int, int], list[MatmulOp]],
                          n: int, seed: int = 0) -> list[tuple]:
    """Pick n shapes with diverse coverage across (size class × M class × op family)."""
    by_size_class = {"small": [], "medium": [], "large": []}
    for shape, ops in unique_shapes.items():
        M, N, K = shape
        # crude size class on K (correlates with model family)
        if K <= 4096:
            cls = "small"
        elif K <= 8192:
            cls = "medium"
        else:
            cls = "large"
        by_size_class[cls].append((shape, ops))

    rng = random.Random(seed)
    result: list[tuple] = []
    per_class = max(1, n // len(by_size_class))
    for cls, items in by_size_class.items():
        rng.shuffle(items)
        result.extend(items[:per_class])
    # Fill remainder
    if len(result) < n:
        extras = [it for items in by_size_class.values() for it in items
                  if it not in result]
        rng.shuffle(extras)
        result.extend(extras[: n - len(result)])
    return result[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="if > 0, print just N representative unique shapes")
    args = parser.parse_args()

    catalog = build_catalog()
    by_shape = deduplicate(catalog)

    print("# vLLM matmul shape catalog\n")
    print(f"Total ops generated:       {len(catalog)}")
    print(f"Unique (M, N, K) shapes:   {len(by_shape)}")
    print(f"Models:                    {len(STANDARD_MODELS) + len(DSV3_MODELS)}")
    print(f"M values per model:        {M_VALUES}\n")

    if args.sample > 0:
        sampled = sample_representative(by_shape, args.sample)
        print(f"## Sampled {len(sampled)} representative shapes for focused probe\n")
        print("| (M, N, K) | example op (model:op_name) | # ops sharing shape |")
        print("|---|---|---:|")
        for shape, ops in sampled:
            example = f"{ops[0].model}:{ops[0].op}"
            print(f"| {shape} | {example} | {len(ops)} |")
        print()
        # Also emit Python literal so the focused probe can import directly
        print("## Python literal (for downstream probe)\n")
        print("SAMPLED_SHAPES = [")
        for shape, ops in sampled:
            example = f"{ops[0].model} {ops[0].op}"
            print(f'    ("{example}", {shape[0]}, {shape[1]}, {shape[2]}),')
        print("]")
        return 0

    # Full catalog dump
    print("## Full catalog (deduplicated by shape)\n")
    print("| (M, N, K) | example op (model:op) | total ops with this shape |")
    print("|---|---|---:|")
    by_shape_sorted = sorted(by_shape.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
    for shape, ops in by_shape_sorted:
        example = f"{ops[0].model}:{ops[0].op}"
        print(f"| {shape} | {example} | {len(ops)} |")

    # Model-by-model summary
    print("\n## Per-model summary\n")
    print("| model | block ops | M values | total | unique shapes contributed |")
    print("|---|---:|---:|---:|---:|")
    for cfg in STANDARD_MODELS:
        ops = []
        for M in M_VALUES:
            ops.extend(block_matmuls(cfg, M))
        unique = {o.shape_tuple() for o in ops}
        print(f"| {cfg.name} | 6 | {len(M_VALUES)} | {len(ops)} | {len(unique)} |")
    for cfg in DSV3_MODELS:
        ops = []
        for M in M_VALUES:
            ops.extend(block_matmuls_dsv3(cfg, M))
        unique = {o.shape_tuple() for o in ops}
        print(f"| {cfg.name} | 8 | {len(M_VALUES)} | {len(ops)} | {len(unique)} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

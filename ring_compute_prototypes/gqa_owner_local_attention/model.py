#!/usr/bin/env python3
"""First-principles ownership, traffic, and numerical model for GQA decode."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DecodeGQAConfig:
    """Granite-shaped owner-local GQA decomposition."""

    batch: int = 1
    query_heads: int = 32
    kv_heads: int = 8
    query_tokens: int = 1
    context: int = 512
    head_dim: int = 128
    key_owners: int = 4
    block_size: int = 64
    dtype_bytes: int = 2

    def __post_init__(self) -> None:
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        if self.query_groups != self.key_owners:
            raise ValueError(
                "the P4 prototype maps one query group and one key owner per lane"
            )
        if self.context % (self.key_owners * self.block_size):
            raise ValueError(
                "context must contain an integer number of blocks per owner"
            )
        if self.batch != 1:
            raise ValueError("the first device schedule targets batch-one decode")

    @property
    def query_groups(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def blocks_per_owner(self) -> int:
        return self.context // (self.key_owners * self.block_size)

    @property
    def cores(self) -> int:
        return self.kv_heads * self.key_owners

    @property
    def q_bytes(self) -> int:
        return (
            self.batch
            * self.query_heads
            * self.query_tokens
            * self.head_dim
            * self.dtype_bytes
        )

    @property
    def unique_k_or_v_bytes(self) -> int:
        return (
            self.batch
            * self.kv_heads
            * self.context
            * self.head_dim
            * self.dtype_bytes
        )

    @property
    def expanded_k_or_v_bytes(self) -> int:
        return self.unique_k_or_v_bytes * self.query_groups

    @property
    def remote_q_delivered_bytes(self) -> int:
        # Every unique Q element is already present at one cohort lane and is
        # delivered to the other P-1 stationary K/V owners.
        return self.q_bytes * (self.key_owners - 1)

    @property
    def q_resident_bytes_per_owner(self) -> int:
        return (
            self.batch
            * self.query_groups
            * self.query_tokens
            * self.head_dim
            * self.dtype_bytes
        )

    @property
    def k_or_v_bytes_per_owner(self) -> int:
        return self.unique_k_or_v_bytes // self.cores

    def core_id(self, kv_head: int, owner: int) -> int:
        if not 0 <= kv_head < self.kv_heads:
            raise ValueError(f"invalid kv_head={kv_head}")
        if not 0 <= owner < self.key_owners:
            raise ValueError(f"invalid owner={owner}")
        return kv_head * self.key_owners + owner

    def q_edges(self) -> list[tuple[int, int]]:
        """Directed remote deliveries for the cohort-local Q all-gather."""

        edges: list[tuple[int, int]] = []
        for kv_head in range(self.kv_heads):
            for query_group in range(self.query_groups):
                source = self.core_id(kv_head, query_group)
                for owner in range(self.key_owners):
                    destination = self.core_id(kv_head, owner)
                    if destination != source:
                        edges.append((source, destination))
        return edges

    def report(self) -> dict[str, object]:
        return {
            "config": asdict(self),
            "query_groups": self.query_groups,
            "blocks_per_owner": self.blocks_per_owner,
            "cores": self.cores,
            "q_bytes": self.q_bytes,
            "unique_k_or_v_bytes": self.unique_k_or_v_bytes,
            "expanded_k_or_v_bytes": self.expanded_k_or_v_bytes,
            "avoided_k_or_v_expansion_bytes": (
                self.expanded_k_or_v_bytes - self.unique_k_or_v_bytes
            ),
            "remote_q_delivered_bytes": self.remote_q_delivered_bytes,
            "q_resident_bytes_per_owner": self.q_resident_bytes_per_owner,
            "k_or_v_bytes_per_owner": self.k_or_v_bytes_per_owner,
            "q_remote_edge_count": len(self.q_edges()),
        }


def owner_local_states(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return associative softmax partials `(m, l, O)` for every key owner.

    Shapes:
      q: ``[Hkv, G, Lq, D]``
      k/v: ``[Hkv, P, NB, BS, D]``
      m/l: ``[Hkv, G, P, Lq]``
      O: ``[Hkv, G, P, Lq, D]`` (unnormalized numerator)
    """

    if q.ndim != 4 or k.ndim != 5 or v.shape != k.shape:
        raise ValueError(f"unexpected shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"incompatible shapes: q={q.shape}, k={k.shape}")

    scale = 1.0 / np.sqrt(q.shape[-1])
    # [Hkv, G, P, NB, Lq, BS]
    scores = np.einsum("hgqd,hoptd->hgopqt", q, k) * scale
    h, g, owners, blocks, lq, block_size = scores.shape
    scores = scores.transpose(0, 1, 2, 4, 3, 5).reshape(
        h, g, owners, lq, blocks * block_size
    )
    values = v.reshape(h, owners, blocks * block_size, v.shape[-1])

    m = scores.max(axis=-1)
    probabilities = np.exp(scores - m[..., None])
    l = probabilities.sum(axis=-1)
    numerator = np.einsum("hg oqt,hotd->hgoqd", probabilities, values)
    return m, l, numerator


def merge_owner_states(
    m: np.ndarray,
    l: np.ndarray,
    numerator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge owner partials with the online-softmax associative identity."""

    global_m = m.max(axis=2)
    owner_scale = np.exp(m - global_m[:, :, None, :])
    global_l = (owner_scale * l).sum(axis=2)
    global_numerator = (owner_scale[..., None] * numerator).sum(axis=2)
    output = global_numerator / global_l[..., None]
    return global_m, global_l, output


def direct_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reference attention after concatenating owner-major K/V shards."""

    h, owners, blocks, block_size, d = k.shape
    keys = k.reshape(h, owners * blocks * block_size, d)
    values = v.reshape(h, owners * blocks * block_size, d)
    scores = np.einsum("hgqd,htd->hgqt", q, keys) / np.sqrt(d)
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return np.einsum("hgqt,htd->hgqd", probabilities, values)

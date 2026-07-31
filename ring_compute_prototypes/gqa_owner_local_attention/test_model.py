from __future__ import annotations

import unittest

import numpy as np

from ring_compute_prototypes.gqa_owner_local_attention.model import (
    DecodeGQAConfig,
    direct_attention,
    merge_owner_states,
    owner_local_states,
)


class TestOwnerLocalGQA(unittest.TestCase):
    def test_granite_decode_traffic_and_ownership(self) -> None:
        config = DecodeGQAConfig()
        self.assertEqual(config.query_groups, 4)
        self.assertEqual(config.blocks_per_owner, 2)
        self.assertEqual(config.cores, 32)
        self.assertEqual(config.q_bytes, 8 * 1024)
        self.assertEqual(config.remote_q_delivered_bytes, 24 * 1024)
        self.assertEqual(config.unique_k_or_v_bytes, 1024 * 1024)
        self.assertEqual(config.expanded_k_or_v_bytes, 4 * 1024 * 1024)
        self.assertEqual(config.q_resident_bytes_per_owner, 1024)
        self.assertEqual(config.k_or_v_bytes_per_owner, 32 * 1024)

        edges = config.q_edges()
        self.assertEqual(len(edges), 96)
        self.assertTrue(all(source // 4 == destination // 4 for source, destination in edges))
        self.assertTrue(all(source != destination for source, destination in edges))

    def test_associative_owner_merge_matches_direct_attention(self) -> None:
        config = DecodeGQAConfig()
        generator = np.random.default_rng(20260731)
        q = generator.normal(
            size=(
                config.kv_heads,
                config.query_groups,
                config.query_tokens,
                config.head_dim,
            )
        ).astype(np.float32)
        k = generator.normal(
            size=(
                config.kv_heads,
                config.key_owners,
                config.blocks_per_owner,
                config.block_size,
                config.head_dim,
            )
        ).astype(np.float32)
        v = generator.normal(size=k.shape).astype(np.float32)

        m, l, numerator = owner_local_states(q, k, v)
        self.assertEqual(
            m.shape,
            (
                config.kv_heads,
                config.query_groups,
                config.key_owners,
                config.query_tokens,
            ),
        )
        self.assertEqual(
            numerator.shape,
            (
                config.kv_heads,
                config.query_groups,
                config.key_owners,
                config.query_tokens,
                config.head_dim,
            ),
        )
        _, _, actual = merge_owner_states(m, l, numerator)
        expected = direct_attention(q, k, v)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()

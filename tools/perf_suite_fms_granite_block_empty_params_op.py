"""Perf-suite custom op for one FMS Granite block with empty Spyre weights.

This wrapper is for compiler/performance investigation.  It constructs the FMS
Granite 3 8B block shape, casts it to fp16, and materializes parameters directly
on Spyre with ``Module.to_empty`` so benchmark runs do not spend time copying
checkpoint or random CPU weights to the device.

Usage:

    python benchmark.py --stack torch-spyre --op fms_granite_block_empty \
        --op-file tools/perf_suite_fms_granite_block_empty_params_op.py \
        --shape 1 512 4096

The single shape is interpreted as ``batch, sequence, embedding``.  The default
scope is a full Granite block with non-causal SDPA to match the existing
``fms_granite_micro`` attention microbench limitation.  Set
``SPYRE_FMS_GRANITE_BLOCK_SCOPE=mlp`` or ``attention`` to isolate the block's
feed-forward or attention submodules without the normalization prefix.
"""

from __future__ import annotations

import os


GRANITE_3_8B_HIDDEN_SIZE = 4096
GRANITE_3_8B_INTERMEDIATE_SIZE = 12800
GRANITE_3_8B_ATTENTION_HEADS = 32
GRANITE_3_8B_KV_HEADS = 8
GRANITE_3_8B_HEAD_DIM = 128
GRANITE_3_8B_ATTENTION_MULTIPLIER = 0.0078125


def _scope() -> str:
    scope = os.environ.get("SPYRE_FMS_GRANITE_BLOCK_SCOPE", "full").strip().lower()
    if scope not in {"full", "attention", "attention_with_norm", "mlp", "mlp_with_norm"}:
        raise ValueError(
            "SPYRE_FMS_GRANITE_BLOCK_SCOPE must be one of 'full', "
            "'attention', 'attention_with_norm', 'mlp', or 'mlp_with_norm', "
            f"got {scope!r}"
        )
    return scope


def _attn_name() -> str:
    return os.environ.get(
        "SPYRE_FMS_GRANITE_BLOCK_ATTN_NAME", "sdpa_bidirectional"
    ).strip()


def _validate_input_shapes(input_shapes) -> tuple[int, int, int]:
    if len(input_shapes) != 1 or len(input_shapes[0]) != 3:
        raise ValueError("FMS Granite block empty-params op expects one --shape: B S E")
    batch, seq_len, emb_dim = (int(value) for value in input_shapes[0])
    if emb_dim != GRANITE_3_8B_HIDDEN_SIZE:
        raise ValueError(
            "FMS Granite 3 8B block benchmark expects E="
            f"{GRANITE_3_8B_HIDDEN_SIZE}, got {emb_dim}"
        )
    return batch, seq_len, emb_dim


def _granite_config(torch):
    from fms.models.granite import GraniteConfig

    return GraniteConfig(
        src_vocab_size=49155,
        emb_dim=GRANITE_3_8B_HIDDEN_SIZE,
        norm_eps=1e-5,
        nheads=GRANITE_3_8B_ATTENTION_HEADS,
        kvheads=GRANITE_3_8B_KV_HEADS,
        head_dim=GRANITE_3_8B_HEAD_DIM,
        nlayers=1,
        hidden_grow_factor=GRANITE_3_8B_INTERMEDIATE_SIZE
        / GRANITE_3_8B_HIDDEN_SIZE,
        max_expected_seq_len=int(
            os.environ.get("SPYRE_FMS_GRANITE_BLOCK_MAX_SEQ_LEN", "8192")
        ),
        rope_theta=10000.0,
        pad_id=0,
        p_dropout=0.0,
        tie_heads=True,
        embedding_multiplier=12.0,
        logits_scaling=16.0,
        residual_multiplier=0.22,
        attention_multiplier=GRANITE_3_8B_ATTENTION_MULTIPLIER,
        fused_weights=os.environ.get("SPYRE_FMS_GRANITE_BLOCK_FUSED_WEIGHTS", "1")
        != "0",
    )


def _make_rotary_embedding(config):
    from fms.modules.positions import RotaryEmbedding

    rope_scaling = {"rope_type": "ntk" if config.ntk_scaling else "regular"}
    return RotaryEmbedding(
        dim=config.head_dim,
        scaling=rope_scaling,
        max_seq_len=config.max_expected_seq_len,
        ratio=config.rope_theta,
    )


def _make_block(torch):
    from fms.models.granite import GraniteBlock

    config = _granite_config(torch)
    return GraniteBlock(config, _make_rotary_embedding(config))


class _GraniteBlockBenchModule:
    """Lazy wrapper to avoid top-level FMS imports in perf-suite validation."""

    def __new__(cls, torch):
        class GraniteBlockBenchModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scope = _scope()
                self.attn_name = _attn_name()
                self.block = _make_block(torch)

            def forward(self, x):
                if self.scope == "attention":
                    return self.block.attn(q=x, attn_name=self.attn_name)
                if self.scope == "attention_with_norm":
                    x = self.block.ln(x)
                    return self.block.attn(q=x, attn_name=self.attn_name)
                if self.scope == "mlp":
                    return self.block.ff_sub_layer(x)
                if self.scope == "mlp_with_norm":
                    x = self.block.ff_ln(x)
                    return self.block.ff_sub_layer(x)
                return self.block(x, attn_name=self.attn_name)

        return GraniteBlockBenchModule()


def create_tensors(torch, input_shapes, op, stack):
    batch, seq_len, emb_dim = _validate_input_shapes(input_shapes)
    return (torch.rand((batch, seq_len, emb_dim), dtype=torch.float16),)


def get_module(op, torch, stack, input_shapes):
    _validate_input_shapes(input_shapes)
    module = _GraniteBlockBenchModule(torch)
    module = module.to(dtype=torch.float16)

    if stack == "tsp" and os.environ.get("SPYRE_FMS_GRANITE_BLOCK_TO_EMPTY", "1") != "0":
        module = module.to_empty(device=torch.device("spyre"))

    module.requires_grad_(False)
    module.eval()
    return module

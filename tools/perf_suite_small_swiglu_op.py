"""Perf-suite custom op for small SwiGLU/MLP coordinate-remap probes.

Usage with spyre-perf-suite benchmark.py:

    python benchmark.py --stack torch-spyre --op small_swiglu \
        --op-file tools/perf_suite_small_swiglu_op.py \
        --shape 1 256 128 512

The single shape is interpreted as ``batch, sequence, embedding, hidden``.
By default this flattens ``batch * sequence`` and emits an ``mm``-based SwiGLU,
which exercises the current coordinate-remap path.  Set
``SPYRE_SMALL_SWIGLU_MODE=bmm`` to keep the batch dimension and emit ``bmm``.
"""

from __future__ import annotations

import os


def _mode() -> str:
    mode = os.environ.get("SPYRE_SMALL_SWIGLU_MODE", "flat_mm").strip().lower()
    if mode not in {"flat_mm", "bmm"}:
        raise ValueError(
            "SPYRE_SMALL_SWIGLU_MODE must be either 'flat_mm' or 'bmm', "
            f"got {mode!r}"
        )
    return mode


def _dims(input_shapes):
    if len(input_shapes) != 1 or len(input_shapes[0]) != 4:
        raise ValueError("small_swiglu expects one --shape: B S E H")
    batch, seq_len, emb_dim, hidden_dim = input_shapes[0]
    return int(batch), int(seq_len), int(emb_dim), int(hidden_dim)


def create_tensors(torch, input_shapes, op, stack):
    batch, seq_len, emb_dim, hidden_dim = _dims(input_shapes)
    if _mode() == "bmm":
        x = torch.randn(batch, seq_len, emb_dim, dtype=torch.float16)
        gate = torch.empty(batch, emb_dim, hidden_dim, dtype=torch.float16)
        up = torch.empty(batch, emb_dim, hidden_dim, dtype=torch.float16)
        down = torch.empty(batch, hidden_dim, emb_dim, dtype=torch.float16)
    else:
        x = torch.randn(batch * seq_len, emb_dim, dtype=torch.float16)
        gate = torch.empty(emb_dim, hidden_dim, dtype=torch.float16)
        up = torch.empty(emb_dim, hidden_dim, dtype=torch.float16)
        down = torch.empty(hidden_dim, emb_dim, dtype=torch.float16)
    torch.nn.init.kaiming_uniform_(gate)
    torch.nn.init.kaiming_uniform_(up)
    torch.nn.init.kaiming_uniform_(down)
    return x, gate, up, down


def get_function(op, torch, stack):
    silu = torch.nn.functional.silu

    def small_swiglu(x, gate, up, down):
        gate_out = torch.matmul(x, gate)
        up_out = torch.matmul(x, up)
        return (up_out * silu(gate_out)) @ down

    return small_swiglu

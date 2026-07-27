"""Identify Granite prefill relayout edges by structure instead of by name.

Buffer names (``buf10``, ``buf52``, ...) are assigned during lowering. They shift
whenever the surrounding stack inserts or fuses an op, so a matcher keyed to a
literal name silently stops matching and its edge is dropped -- while the model
still produces the correct token, because dropping a relayout only costs time.
That failure mode is invisible to the correctness gate.

This module names tensors by what they *are* rather than where they happened to
land. The signatures use only shape and graph position, both of which survive
renumbering:

    attn_norm_out   [1, S, H] pointwise read by three matmuls whose output widths
                    are H, H*kv/nh, H*kv/nh   (grouped-query: K and V are narrower)
    q_proj/k_proj/v_proj    those three matmul consumers, widest first
    mlp_norm_out    [1, S, H] pointwise read by two matmuls of equal width I > H
    gate_proj/up_proj       those two consumers
    down_proj       the matmul taking [1, S, I] to [1, S, H]
    out_proj        the matmul taking attention output to [1, S, H]
    attn_out        the rank-5 attention output feeding out_proj

Nothing here decides policy; it only answers "which buffer is this".
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Resolved role -> buffer name, per graph. Keyed by id(graph): the pass runs once
# per compiled graph and we only need the map to survive within that pass.
_ROLE_CACHE: dict[int, dict[str, str]] = {}


def _shape(op) -> list[int]:
    out = []
    for size in op.get_size():
        try:
            out.append(int(str(size)))
        except (TypeError, ValueError):
            return []
    return out


def _resolve(graph, is_matmul, reads_of) -> dict[str, str]:
    """Build the role -> buffer-name map for one graph."""
    from torch._inductor.ir import ComputedBuffer  # local: keep import cost off the hot path

    ops = [op for op in graph.operations if isinstance(op, ComputedBuffer)]
    by_name = {op.get_name(): op for op in ops}

    # consumers[name] = matmul ops that read `name`
    consumers: dict[str, list] = {}
    for op in ops:
        for dep_name in reads_of(op):
            consumers.setdefault(dep_name, []).append(op)

    roles: dict[str, str] = {}

    def width(op):
        s = _shape(op)
        return s[-1] if s else 0

    for name, op in by_name.items():
        s = _shape(op)
        if len(s) != 3 or s[0] != 1:
            continue
        seq, hidden = s[1], s[2]
        mm = [c for c in consumers.get(name, []) if is_matmul(c)]
        if len(mm) < 2:
            continue
        widths = sorted((width(c) for c in mm), reverse=True)

        # Q/K/V: three matmul consumers, the two narrow ones equal and narrower
        # than the wide one. That inequality IS grouped-query attention.
        if (
            len(mm) == 3
            and widths[0] == hidden
            and widths[1] == widths[2]
            and widths[1] < widths[0]
        ):
            roles["attn_norm_out"] = name
            ordered = sorted(mm, key=lambda c: (-width(c), c.get_name()))
            roles["q_proj"] = ordered[0].get_name()
            roles["k_proj"] = ordered[1].get_name()
            roles["v_proj"] = ordered[2].get_name()
            continue

        # gate/up: two matmul consumers of equal width wider than hidden.
        if len(mm) == 2 and widths[0] == widths[1] and widths[0] > hidden:
            roles["mlp_norm_out"] = name
            ordered = sorted(mm, key=lambda c: c.get_name())
            roles["gate_proj"] = ordered[0].get_name()
            roles["up_proj"] = ordered[1].get_name()
            roles["mlp_intermediate_width"] = str(widths[0])
            continue

    # down projection: matmul [1,S,I] -> [1,S,H], I from the gate/up pair above.
    inter = roles.get("mlp_intermediate_width")
    if inter:
        inter_w = int(inter)
        for op in ops:
            if not is_matmul(op):
                continue
            s = _shape(op)
            if len(s) != 3 or s[0] != 1:
                continue
            srcs = [by_name.get(n) for n in reads_of(op)]
            if any(
                src is not None and _shape(src)[-1:] == [inter_w] for src in srcs
            ):
                roles["down_proj"] = op.get_name()
                break

    # output projection: matmul producing [1,S,H] whose input is not the MLP
    # intermediate -- i.e. the attention side.
    norm_name = roles.get("attn_norm_out")
    if norm_name:
        h = _shape(by_name[norm_name])[-1]
        for op in ops:
            if not is_matmul(op) or op.get_name() == roles.get("down_proj"):
                continue
            s = _shape(op)
            if len(s) == 3 and s[0] == 1 and s[-1] == h:
                srcs = [by_name.get(n) for n in reads_of(op)]
                if any(
                    src is not None and len(_shape(src)) >= 4 for src in srcs
                ):
                    roles["out_proj"] = op.get_name()
                    for n in reads_of(op):
                        src = by_name.get(n)
                        if src is not None and len(_shape(src)) >= 4:
                            roles["attn_out"] = n
                            break
                    break

    return roles


def roles_for(graph, is_matmul, reads_of) -> dict[str, str]:
    key = id(graph)
    cached = _ROLE_CACHE.get(key)
    if cached is None:
        try:
            cached = _resolve(graph, is_matmul, reads_of)
        except Exception:  # never let identification break compilation
            logger.exception("granite role resolution failed; falling back to none")
            cached = {}
        _ROLE_CACHE[key] = cached
        logger.info("granite roles resolved: %s", cached)
        if cached:
            print(f"GRANITE_ROLES {cached}", flush=True)
    return cached


def name_of(graph, role, is_matmul, reads_of) -> str | None:
    return roles_for(graph, is_matmul, reads_of).get(role)

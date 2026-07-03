# Grouped All-Gather / Sub-Stick Probe - 2026-07-03

## Purpose

This probe narrows the next Granite/attention communication class after PR1 scatter.
The target edge is the attention matmul RHS/KERNEL operand all-gather in the generated
Granite block bundle, especially `sdsc_8.json`.

The question was whether the current DLDSC relayout machinery can realize this as
plain `STCDPOpLx` or existing `ReStickifyOpLx`.

## Source Bundle

```text
/home/adnan/codex-isolated/dldsc_granite_clean_relayout_20260703_163108/runs/granite_relayout_s512_failclosed_20260703_173131/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_e0v78nrw
```

Relevant SDSCs:

- `sdsc_8.json`: attention batchmatmul RHS/KERNEL operand collective.
- `sdsc_16.json`: another matmul RHS/KERNEL operand collective.
- `sdsc_7.json`: working unary `ReStickifyOpLx` shape for comparison.

## What The Frontend Correctly Exposes

For `sdsc_8`, the Torch metadata classifies the edge as:

```text
kind = matmul_operand_broadcast
communication_class = all_gather
communication_pattern = all_gather_replicate
consumer_operand_ds_type = KERNEL
read_index = 1
```

The KERNEL operand layout is:

```text
layoutDimOrder_ = [mb, out, in]
stickDimOrder_ = [out]
stickSize_ = [64]
N.out = 512
```

The producer has 32 shards across the KERNEL `out` dimension, so each producer
chunk covers `512 / 32 = 16` `out` elements. The consumer batchmatmul has an
`out:2` compute split, so this is not a single global all-gather. It is a grouped
all-gather/restickify:

- consumer group 0 needs producer chunks 0..15;
- consumer group 1 needs producer chunks 16..31;
- four `out=16` producer chunks are needed to assemble one full `out=64` stick.

## Probe Patch

Deeptools fork branch:

```text
github.ibm.com/Adnan-Hoque1/deeptools:ah/comms-collectives
commit 375837d9a [DXP] Filter matmul operand collective fanout by overlap
```

The patch changes the diagnostic `matmul_operand_broadcast` IFN/STCDP path to
filter destination cores by coordinate overlap instead of blindly fanning every
producer shard to every consumer core. This is still diagnostic; it does not add
sub-stick assembly.

## Replay Result

Replay directory:

```text
/home/adnan/codex-isolated/dldsc_granite_clean_relayout_20260703_163108/runs/dxp_replay_grouped_ifn_deeptoolspath_20260703_220056
```

The replay uses `DEEPTOOLS_PATH` pinned to the source checkout and reaches the
real backend limitation:

```text
DtException: op->inpSP_.at(inpSPIdx).dimToSize_.at(dimNameOuter) >= stickDim,
file .../deeptools/dcg/dcg_fe/pcfg_gen/stcdpOp.cpp line 4342
```

Logs are archived in `logs/`.

## Why This Fails Today

Existing `STCDPOpLx` and `ReStickifyOpLx` are full-stick movement/restickify
mechanisms. They do not currently assemble multiple source pieces that are
smaller than the target stick dimension.

Evidence from Deeptools:

- `dcg/dcg_fe/pcfg_gen/stcdpOp.cpp:4342` requires an STCDP input piece's relevant
  stick dimension to be at least the stick size.
- `dcg/dcg_fe/pcfg_gen/restickifyOp.cpp:353-355` requires each input piece to be
  at least the output stick size when the dimension is an output stick dim.
- `dcg/dcg_fe/pcfg_gen/apeOp.cpp:927-930` requires equal input/output piece counts
  for non-HBM `ReStickifyOpLx`; it is not an N-source-pieces-to-1-output-piece
  gather assembler.

For this edge, the producer piece is `out=16` and the target stick is `out=64`,
so both existing carriers reject the shape for the right reason.

## Current Conclusion

PR1 scatter is not the limiting issue here. The next Granite/attention class is:

```text
grouped all-gather + sub-stick assembly + KERNEL operand binding
```

A production backend path needs one of these shapes:

1. Loop-scoped matmul operand fetch that accepts a grouped list of producer
   chunks and binds them as the RHS/KERNEL transfer loop consumes them.
2. A stick-aware LX restickify/all-gather primitive that packs `N x sub-stick`
   source pieces into full target sticks before matmul.

The second option is not existing `ReStickifyOpLx`; it would require relaxing the
piece-size/equal-piece assumptions and adding explicit intra-stick destination
offsets plus tests such as `4 x out=16 -> 1 x out=64`.

## Status

- Frontend classification: present and useful.
- Backend grouped-overlap filtering: diagnostic patch pushed to fork.
- Backend physical realization: still blocked at sub-stick assembly.
- Scope: non-weight attention activation/RHS operand spill, not weight restickify.

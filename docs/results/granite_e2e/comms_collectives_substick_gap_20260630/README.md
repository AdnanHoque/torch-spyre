# Granite Comms Collectives: Sub-Stick Attention Gap

Date: 2026-06-30

This directory records the current state of the exploratory `ah/comms-collectives`
work after splitting the implementation into two lanes:

1. A staged DLDSC/LX-planner lane.
2. A direct `STCDPOpLx`/data-op carrier lane.

The concrete failing edge is in the attention part of the Granite block:

- Bundle: `stcdp_clean_substick_gap_20260630_211149.tgz`
- SDSC: `bundle_input/sdsc_10.json`
- Root op: `10_batchmatmul`
- Input: `Tensor1`
- Torch classification:
  - `kind = layout_restickify_activation`
  - `communication_pattern = layout_transform_then_operand_broadcast`
  - producer distribution: `out: 32`
  - consumer compute distribution: `x: 16, out: 2`

## What We Learned

PR1-style scatter handles fully materialized LX relayouts where the consumer-shaped
piece fits in scratchpad. This attention edge is different. A full consumer-shaped
post-relayout piece for `Tensor1` is about 2 MiB per core:

```text
[relayout] capacity sdsc=10_batchmatmul lds=Tensor1
           out_piece_size=2.09715e+06
           lx_space_found=0 first_missing_lx_core=0
```

So this edge cannot be fixed by simply materializing another resident LX view.
It needs streamed movement into the consumer loop.

When routed to loop-scoped movement, Deeptools fails in `STCDPOpLx`:

```text
DtException:
op->inpSP_.at(inpSPIdx).dimToSize_.at(dimNameOuter) >= stickDim
file .../dcg/dcg_fe/pcfg_gen/stcdpOp.cpp line 4374
```

That assertion is the important backend gap. The producer fragments are smaller
than a full stick in the destination layout. Current `STCDPOpLx` assumes the
source subpiece spans at least one whole stick along the relevant stick dimension.
The attention layout-transform-plus-broadcast case needs to assemble a destination
stick from multiple source fragments.

## Rejected Diagnostic

We briefly relaxed the `stcdpOp.cpp:4374` whole-stick assertion only for compact
input-neighbor broadcast. That did not produce a valid compile; the replay ran
until timeout with no useful emitted artifact. This is diagnostic evidence that
the whole-stick assumption is deeper than that single collapse-factor check.

Do not treat that relaxation as a candidate fix.

## Agent B Staged DLDSC Patch

`staged_dldsc_agent_20260630_205912.tgz` contains the isolated Torch-side patch
from the staged DLDSC lane. It adds:

- `realization_strategy` metadata.
- A staged strategy:
  `staged_lx_restickify_then_loop_scoped_input_fetch`.
- Scratchpad reservation logic that skips extra resident reservations for
  loop-scoped strategies.
- Focused unit coverage.

Agent B validation:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
  tests/inductor/test_lx_relayout_dldsc.py -q
```

Result: `12 passed`.

The staged Torch patch is aligned with the North Star contract: Torch classifies
the edge and records the tensor distribution versus consumer compute distribution,
while Deeptools owns physical movement synthesis.

## Agent A Direct STCDPOpLx Finding

`agentA_stcdp_direct_findings_20260630/` contains the independent direct-carrier
lane. It reached the same conclusion: existing `STCDPOpLx` cannot represent this
attention edge correctly without additional fragment/range metadata.

Agent A also produced `grouped_destination_fix.diff`, a small backend bugfix for
grouped L3LU destination address selection. That patch may be worth keeping, but
it is not sufficient for this communication class. It fixes destination lane
selection; it does not add sub-stick source/destination offsets.

Direct-carrier replay evidence:

```text
compact direct:      fails at stcdpOp.cpp:4374
noncompact stage 8: fails coverage check at stcdpOp.cpp:440
noncompact stage 16: fails coverage check at stcdpOp.cpp:440
noncompact stage 32: fails program verification with LX_MODLRFIMM lrfimm:-2101120
compact stage 32:   times out
```

## Clean Replay Command

On `adnan-cdx-spyre-dev-pf`:

```bash
ROOT=/home/adnan-cdx/codex-isolated/comms_collectives_stcdp_agent_20260630_190747
SRC=$ROOT/runs/stcdp_agent_validation_20260630_191625/dxp_replay_fresh_attention_subpiece_20260630_193832/bundle_input
RUN=$ROOT/runs/stcdp_clean_substick_gap_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
cp -a "$SRC" "$RUN/bundle_input"

timeout 90s env \
  DXP_LX_RELAYOUT_BROADCAST_STAGE_CORES=4 \
  DXP_LX_RELAYOUT_BROADCAST_COMPACT=1 \
  DXP_LX_RELAYOUT_BROADCAST_SUBPIECE_REUSE=0 \
  DXP_BACKEND_LX_FRAC_AVAIL=1 \
  DXP_LX_FRAC_AVAIL=0 \
  "$ROOT/tools/dxp-split-wrapper-stcdp-agent/dxp_standalone" \
  --bundle -d "$RUN/bundle_input" \
  > "$RUN/dxp.stdout" 2> "$RUN/dxp.stderr"
```

Expected current result:

```text
rc=134
stcdpOp.cpp line 4374
```

## Next Backend Feature

To remove this non-weight attention HBM spill, Deeptools needs a real sub-stick or
ranged LX transfer contract. The missing capability is not "many whole-stick
transfers"; it is:

```text
source core + source LX address + source intra-stick range
destination core + destination LX address + destination intra-stick offset
element or byte count
logical coverage metadata
```

This can be exposed either as:

- An extension to the backend-synthesized DLDSC relayout path.
- A ranged mode of `STCDPOpLx`.

Either way, the backend must preserve intra-stick offsets through:

- subpiece construction,
- placement/address derivation,
- collapse-factor planning,
- L3 ring transfer node generation.

## Classification

This edge is not PR1 scatter. It is a fused communication class:

```text
layout transform + broadcast / input-neighbor fetch
```

In gather/scatter/reduce terms, it is closest to a fragment gather into each
consumer's operand tile, with broadcast/replication across the consumer work
division. It should stay separate from:

- pure 1:1 scatter,
- whole-tensor all-gather replicate,
- reductions/all-reductions,
- offline weight prelayout.

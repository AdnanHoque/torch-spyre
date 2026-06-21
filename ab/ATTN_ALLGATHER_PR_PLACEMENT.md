# Where the attention all-gather sits in the coordinate-remap PR plan (2026-06-21)

Companion to `ATTN_ALLGATHER_HANDOFF.md`. Reconciles our stuck attention all-gather
with Codex's `LXCoordinateRemapOp` work and the PR1/PR2/PR3 scope, after the
device-verified 2026-06-20 SwiGLU snapshot.

## What the 2026-06-20 snapshot changes

The coordinate-remap pass now has a **device-verified** win and **compiles through
deeptools at prefill** — both of which an earlier (stale-branch) read had wrong:

- FMS fused SwiGLU prefill (`B=1 S=512 E=4096`), Kineto `kernel_ms_per_iter`:
  **15.14 → 12.18 ms = 19.5%**. `memory_ms` rises 0.193 → 0.285 (the relay cost).
- SDSC proof: projection output `2_hbm` → `2_lx@0x0`; `neg`/`realdiv`/`mul` inputs
  flip `*_hbm` → `*_lx`. The proj→pointwise HBM reads are provably gone.
- The fix that made prefill compile is the **Same-Core Local Relay** (same-core
  cells routed through a neighbor-core scratch region).
- Decode is correctly **inert** (9.395 → 9.368, pass finds no eligible remap) —
  evidence the pass does not invent work.

Source: `swiglu-ws-co-remap` torch `3ac4c1ed`, deeptools `83f9320c`,
`docs/source/compiler/lx_coordinate_remap_swiglu_snapshot_2026_06_20.md` (`b9b8f30`).

## The attention all-gather is NOT Codex's current relayout

They are two members of the same `LXCoordinateRemapOp` family, separated on **two
independent axes**. The attention all-gather is harder on **both**.

| axis | Codex's current relayout (PR1) | Attention QK^T all-gather |
| --- | --- | --- |
| **cardinality** | **disjoint / one-to-one** ownership *permutation* (`{mb:4,out:8}` → `{mb:32}`). Enforced in code: `_owner_lookup` bails `"duplicate-owner"` if any consumer slice maps to >1 core (`onchip_move.py:162`); coverage validator rejects `coverage-cell-overlap`. No replication, by construction. | **read-only fan-out / multicast** — one K-band replicated to a 4-core Lq cohort. This *is* the "duplicate-owner" the current planner refuses. |
| **bundle locality** | **intra-bundle.** Projection matmul output stays in LX (`sdsc_1 2_lx@0x0`) and the remap+`neg` read it in place — the fused region is one program, LX persists, no HBM round-trip. | **cross-bundle.** `mul(K)` is a *separate device program* from QK^T; LX does not persist, so QK^T re-reads K from HBM via a ReStickify (`x:32`, not the producer's `Lk:32`) and the gather operates on a re-materialization. This is the wall that stuck the all-gather at ~0.02-wrong. |

Corroboration that this is purely SwiGLU work today: Codex's first-principles doc
and planner **never mention attention, QK^T, or all-gather**; the only "K" is the
`producer-k-split-partial-output` skip (matmul reduction split), which PR1 rejects.

## Mapping onto the PR1/PR2/PR3 scope

- **PR1 — exact resharding.** This is exactly Codex's *shipped* relayout: disjoint,
  rejects fan-out/multicast/reductions/split-K partials. Device-verified 19.5%.
  **Done in spirit**, and SwiGLU is a *good* proof vehicle for it.
- **PR2 — read-only fan-out.** Relaxes the `onchip_move.py:162` duplicate-owner bail
  to allow consumer duplicate ownership. Targets the SwiGLU `mul → down_projection`
  edge, which is **intra-bundle**. This is the **same primitive family** as the
  attention all-gather — but a different edge with no program boundary to cross.
- **PR3 — streaming.** Tiled movement/consumption; a scheduler feature, off the
  all-gather's critical path.

**Where attention sits:** the all-gather = **PR2's multicast carrier *plus* a
cross-program co-bundle prerequisite that is in *none* of PR1/PR2/PR3.** PR2 can
land fully on the intra-bundle `mul→down_proj` edge and still not deliver attention,
because attention's producer (`mul(K)`) lives in a different program. That missing
prerequisite sits below PR2, adjacent to the "weight preload/restickify" item that
is currently out-of-scope.

## Two gaps in the scope plan

1. **The cross-bundle boundary is the unscoped item, and it is exactly what
   attention needs.** Make a "co-bundle the producer across the program boundary"
   prerequisite explicit rather than letting PR2 imply attention falls out for free
   — it will not. (Same wall the SwiGLU split-K substrate hit; see
   `SPLITK_VS_COORDINATE_REMAP.md`.)
2. **PR2's capacity-aware split selection collides with the cost-model planner.**
   Rejecting `{mb:4,out:8}` and trying `{mb:8,out:4}`/`{mb:16,out:2}` on activation-LX
   overflow re-derives a split *for capacity*, but the cost model already picks splits
   *for matmul throughput* — and those objectives have been seen to invert each other
   (the narrow-N mis-rank). Design that coupling deliberately, not as an independent
   capacity guard bolted onto the planner output.

## Recommendation — one carrier, two sequenced tracks

Treat the current relayout as the validated **disjoint mover (PR1) with a real
number**. Sequence everything else onto the *same* carrier instead of two
half-finished movers:

1. **PR2 first, on `mul→down_proj`** (intra-bundle): generalize the carrier to
   fan-out, prove the multicast value-correct where there is no program boundary to
   fight.
2. **Then attention**, as its own track depending on PR2: point the same fan-out
   carrier at QK^T *together with* co-bundling `mul(K)` into the QK^T program, so the
   gather operates on the producer's true LX shards (the model the geometry already
   assumes) and genuinely replaces the HBM broadcast.

Warp specialization stays last and gated on a stable movement baseline: the
classifier is **audit-only** today (`codegen/onchip_move.py:1097`), the remap is a
**sequential barrier** before the consumer, and silu*mul-into-matmul epilogue fusion
is **refuted** (deeptools matmul-epilogue whitelist is `stridedadd` only). "Warp-spec
last" and "no epilogue fusion in scope" are both correct calls.

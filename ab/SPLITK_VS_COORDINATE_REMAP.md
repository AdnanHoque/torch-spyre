# Split-K down_proj vs the coordinate-remap LX data-op

Two ways to give the Granite SwiGLU MLP a **genuine on-chip core-to-core reduce**
for the `down_proj`, compared head-to-head. Companion to
`reshard/MECHANISM_AND_BLOCKER.md` (why the data-op move is deeptools-blocked),
`coassign/README.md` (the no-move element-wise win), and the compile-time
validation `/tmp/swiglu-ws-v2/torch_spyre/_inductor/reshard/SPLITK_VALIDATION.md`.

## The two approaches

Both target the same edge: `gate/up matmul → silu/mul → down_proj`, where
`down_proj` is a `[*, 12800] · [12800, 4096]` contraction over the 12800 hidden
dim. The cost model splits gate/up wide-N `{mb:4, out:8}`; the question is how the
12800-wide reduction the `down_proj` performs gets divided across cores and
combined.

| | **coordinate-remap** (codex / `reshard/`) | **split-K** (`swiglu-ws-v2`) |
|---|---|---|
| What it is | A new `STCDPOpLx` gather/scatter data-op that transports the activation across cores (LX → RIU ring → LX) so each core holds the band the next op wants | Pure work-division steering: the `down_proj` is given a `{M:4, K:8}` split (K in the reduction slot); each core reduces its own K-band, the **existing** PSUM ring reduce combines partials |
| New primitive? | **Yes** — a coordinate-remap data-op (2-D `mb×out` scatter) | **No** — `align_downproj_split_k` + `coassign_elementwise`, then the PSUM ring reduce already in `codegen/superdsc.py:177` |
| Data-op / mixed SDSC | yes | **no** |
| dxp gate | yes (`SdscTree.cpp:152` patched) | **no — stock dxp** |
| DCG EBR packer | yes — **the blocker** (assumes `core == column-band`, false for 2-D co-splits; ~0 output) | **no — never touched** |
| Reduce machinery | a transported re-tile, then a plain matmul | the inherent PSUM ring reduce of any K-split matmul (mathematically exact) |

## Mechanism weight

This is the decisive axis. The coordinate-remap path **moves data**: the SwiGLU
producer is the first 2-D `{mb:4, out:8}` producer in the codebase, four cores
share each column band (`core p` owns band `p // 4`), and the DCG EBR packer's
baked-in `EBR = core × stride` assumption is wrong by exactly the `mb_split = 4`
factor — cores 8–31 store out of bounds, output ≈ 0
(`reshard/MECHANISM_AND_BLOCKER.md` §3). Every move primitive (scatter STCDP,
gather `InputFetchNeighbor`, standalone vs mixed SDSC) routes through the **same**
`computeMulticastOptMetadata` chokepoint, so the only landing path is a deeptools
packer generalisation + dxp rebuild + device re-validation. The frontend is
exhausted.

Split-K **moves nothing as a primitive**. It only changes how the `down_proj`
reads its activation and which slot the 12800 dim's split lands in. The reduction
is performed by the PSUM ring reduce that fires for *any* K-split matmul — no new
microcode, no data-op, no dxp gate, no EBR packer. The compile-time validation
confirms the split commits exactly band-for-band on the shared 12800 dim:

| op | role | committed split |
|---|---|---|
| gate/up matmul | wide-N | `{mb:4, out:8}` (cost model) |
| silu / mul tail | element-wise | co-assigned `{mb:4, out:8}` (not pointwise pure-M) |
| **down_proj** | reduction | `{M:4, K:8}` — **K in the reduction slot** |

So split-K is strictly lighter on every axis the coordinate-remap path is heavy
on. The cost it trades for that lightness is in the PSUM chain length (below), not
in deeptools surface.

## In-place locality hinges on k_fast (CPU-measured)

A K-split matmul is value-correct regardless of which core owns which K-band — the
PSUM ring reduce sums them all. The *locality* question is separate: does each core
reduce the band it **produced** (zero cross-core activation move), or does the band
ownership permute between the `mul` (producer) and the `down_proj` (consumer)?

That hinges entirely on the `core_to_slot` mapping the two ops take on the shared
12800 dim. The `down_proj` is a K-split matmul → takes `_k_fast_core_to_slice_mapping`
when `SPYRE_CORE_ID_K_FAST_EMISSION=1`; the `mul` is a pointwise with an N-split →
always takes the plain mapping. Measured (`SPLITK_VALIDATION.md` §2):

**k_fast ON (default):** mappings diverge → **NOT in-place**

```
down_proj  k_fast=True   core_to_slot = ((0, Mod(core//8, 4)), (1, Mod(core, 8)))
mul                      core_to_slot = ((0, Mod(core, 4)),    (1, Mod(core//4, 8)))
```

**k_fast OFF:** mappings identical → **in-place** (each core reduces its own band)

```
down_proj  k_fast=False  core_to_slot = ((0, Mod(core, 4)), (1, Mod(core//4, 8)))
mul                      core_to_slot = ((0, Mod(core, 4)), (1, Mod(core//4, 8)))
```

This gives split-K **two** in-place options, both strictly lighter than the
coordinate-remap 2-D scatter:

1. **k_fast OFF** — in-place for free, one config flag. Cost: a **longer PSUM
   reduction chain** (hops `1 → m·n`; the 8 K-collaborators per M-band are
   ring-strided, not adjacent), so the combine is slower per-step than the
   1-hop k_fast layout.
2. **k_fast ON + a 1-D same-stick re-ownership** — keep the 1-hop PSUM chain, but
   the producer bands are permuted relative to the k_fast K-band order, so align
   them with a **1-D, same-shape, same-stick `STCDPOpLx` re-ownership** that just
   re-assigns which core owns which already-correctly-shaped band. This is a pure
   row redistribution at constant column (`src_col == dst_col`): **`core == column`
   is preserved, so it never hits the EBR packer bug** that blocks the 2-D
   coordinate-remap. This is the **correct home for the 1-D re-ownership primitive**
   — the same primitive that gave no SwiGLU win when applied as a 2-D *re-tiling*
   (it could not realign tiles across the `mb×out` co-split) finds its proper role
   here as a band-alignment step that re-owns identically-shaped slices.

Option 1 is the lightest in-place path and needs no data-op at all; option 2
recovers the 1-hop chain at the cost of a single 1-D re-own move that the packer
*can* express.

## What each still needs

**Coordinate-remap (codex):**

- A deeptools DCG EBR-packer generalisation: derive the L3SU dest column from the
  subpiece `out_` coordinate (`core // mb_split`) instead of the raw core index
  (`reshard/MECHANISM_AND_BLOCKER.md` §5, `RESHARD_DESIGN.md:87`). One function,
  contained blast radius, but a **deeptools** change.
- A `dxp` rebuild against the harvest flex (flex-skew risk) + device `max_err`
  re-validation. Until the packer lands, the move is value-broken (≈0 output).

**Split-K:**

- **Co-bundle the `mul → down_proj` into one device program.** Without it the
  activation round-trips HBM between two separate programs and split-K only changes
  the `down_proj`'s read tiling, not the round-trip. The co-bundle infrastructure
  already exists (`plan_reduction_reshard_edges` records the edge;
  `can_fuse_vertical` / `spyre_fuse_nodes` keep the two ops in one bundle) but is
  gated on `config.onchip_reduction_reshard`. The plan: **widen the edge-recording
  and co-bundle gates to also fire under `config.onchip_splitk_downproj`, but
  WITHOUT invoking the reshard mixed-fold splice** (`realize_reduction_reshard_bundle`,
  `bundle.py:116`) — split-K needs the **plain co-bundle only**. The
  `bundle.py:116` realize gate stays `onchip_reduction_reshard`-only, so the STCDP
  splice never fires under pure split-K. (If `onchip_reduction_reshard` were used
  to *get* the co-bundle, the splice would fire on the now-in-bundle `down_proj` —
  it passes `_is_reduction_consumer` — and corrupt; hence the gate widening is
  required, not optional.)
- **LX-residency pinning of the `mul` output across the bundle edge.** The plain
  co-bundle is necessary but not sufficient: LX residency is decided independently
  by the graph-level scratchpad allocator. For the multi-user `mul`-output buffer
  the allocator pins LX only if `get_ncores_for_buffers` does **not** return -1,
  which requires the `mul`'s per-core N-band view and the `down_proj`'s per-core
  K-band view to **agree** on the 12800 dim — i.e. the **same k_fast-OFF
  identical-mapping** condition above. With k_fast ON the views diverge → -1 →
  HBM, even with a perfect co-bundle. Unlike the reshard path, **no `apply_lx_flip`
  / STCDP data-op is needed**: the k_fast-OFF edge is same-shard same-core, so the
  allocator just renames the HBM address to an LX address — nothing moves. Confirm
  on a CPU compile (post-`scratchpad_planning`) that the `mul`-output buffer carries
  an `'lx'` allocation **and** `get_ncores_for_buffers != -1` before any device
  claim; tune `dxp_lx_frac_avail` / `onchip_reduction_reshard_region0` only if the
  dump shows the band rejected.
- **Device value-correctness** of the co-bundled K-split SwiGLU (a K-split + PSUM
  ring reduce is exact, so only fp16 rounding from the longer chain is expected;
  near-zero output = corruption).
- **Device perf** vs the cost model's natural `down_proj` split, and the
  k_fast-OFF PSUM-chain cost vs the k_fast-ON 1-D re-own alignment cost.

## Recommendation

**Pursue split-K (k_fast OFF) as the primary on-chip reduce.** It reaches a genuine
in-place core-to-core reduce with **zero new primitives, zero data-ops, stock dxp,
and no DCG EBR packer dependency** — it is strictly lighter than the coordinate-remap
path on every mechanism axis, and the CPU validation already confirms the splits
commit exactly and the k_fast-OFF mappings align in-place. Its remaining work is all
**Inductor-side** (widen the co-bundle gate, verify LX pinning, device-validate),
with no deeptools blocker on the critical path.

Keep the **coordinate-remap data-op parked** until the deeptools EBR-packer fix
lands — it is the only path for a *genuine* cross-core move on a non-co-assignable,
non-K-splittable edge, but for the `down_proj` reduction specifically, split-K
delivers the same on-chip-reduce outcome without paying the data-op's deeptools
cost.

If the k_fast-OFF PSUM chain proves too long on device (the in-place win must beat
the `1 → m·n`-hop chain latency), the **1-D same-stick re-ownership** (k_fast ON +
band-alignment STCDP) is the natural fallback: it recovers the 1-hop chain and,
because it preserves `core == column`, it sidesteps the 2-D EBR packer bug entirely
— making it a *much* smaller deeptools ask than the full coordinate-remap, and the
correct application of the 1-D re-ownership primitive that found no role as a 2-D
re-tiling.

---

## DEVICE-MEASURED VERDICT (2026-06-19) — split-K is a perf dead-end for the SwiGLU

Everything above (the "Recommendation" to pursue split-K) was reasoned from CPU
compile evidence. It has now been **device-measured**, and the recommendation is
**overturned for this SwiGLU**. Three corrections:

### 1. The co-bundle lever is `spyre_fuse_nodes`, NOT `can_fuse_vertical`

"What each still needs" above named `can_fuse_vertical` as the co-bundle lever.
That hook is **never reached** on this torch: `torch._inductor.scheduler.Scheduler.can_fuse`
short-circuits on `V.choices.can_fuse(...)`, and `torch_spyre/_inductor/choices.py`
returns `False` unconditionally for all of `can_fuse` / `can_fuse_vertical` /
`can_fuse_horizontal`. The actual bundling is done by **`fusion.spyre_fuse_nodes`**
(a greedy packer bounded by the 5-non-intermediate-tensor SDSC budget, issue #827).
The working co-bundle (swiglu-ws-v2 `04932ec`) forces a bundle boundary **before**
the recorded edge producer (the mul) in `spyre_fuse_nodes`, splitting
`{gate,silu,up} | {mul,down}`. Both fit the budget (4/5 and 2/5 non-intermediate
tensors). The widened `can_fuse_vertical` gate is kept only as documented intent.

### 2. The co-bundle is HBM-NEUTRAL (CPU accounting) — no traffic win

`max_tensors = 5`. The natural lowering already packs `gate+silu+up+mul+weight-prep`
into a **full 5/5** bundle, so the down_proj spills to its own. The budget forces
**exactly one `[512,12800]` (25.6 MB) intermediate to cross HBM either way** —
natively the mul output; in the `{gate,silu,up}|{mul,down}` co-bundle the silu and
up outputs cross instead. So co-bundling the mul→down reduce on-chip yields **no net
HBM saving** for this SwiGLU; it only moves which intermediate crosses.

### 3. Device perf (wedge-free, empty-on-device weights, median of 20 forwards)

| config | prefill `[1,512,4096]` | decode `[4,1,4096]` |
|---|---|---|
| baseline (split-K off, cost-model split) | **18.885 ms** | **8.175 ms** |
| split-K only (no co-bundle) | 22.521 ms (**+19%**) | — |
| co-bundle (on-chip reduce, k_fast off) | 22.296 ms (**+18%**) | — |
| split-K only, decode | — | 8.299 ms (neutral) |

Split-K `{M:4,K:8}` **hurts** prefill wide-N (the cost model's `{mb,out}` split has
more output parallelism + less reduction overhead) and is **neutral** on decode at
the full-SwiGLU level. The co-bundle (which carries the split-K split) does not
recover the penalty — consistent with its HBM-neutrality.

**Proven:** the co-bundle codegens a valid `Pointwise → K-split-matmul` SDSC bundle
**without the reshard splice** and executes on device (the forward completes). The
substrate mechanism (co-bundle + in-place k_fast-off split-K + PSUM ring reduce) is
**real and runs end-to-end**. Value-correctness is the only piece pending (a
standard K-split + PSUM reduce is mathematically exact, so it is expected to pass;
blocked only by the recurring H2D weight-upload wedge).

### Conclusion

The on-chip core-to-core reduce **works**, but the SwiGLU is the **wrong vehicle**:
the 5-tensor SDSC budget makes the co-bundle HBM-neutral, and split-K hurts the
prefill matmul it targets. This sharpens `reshard/MECHANISM_AND_BLOCKER.md`'s
cross-bundle conclusion — even *with* co-bundling, the budget prevents a net win
for the SwiGLU. The substrate's real perf home is a **different op**: the 1-D
cross-core **re-ownership** case (attention QK^T→softmax, the proven `STCDPOpLx`)
or the **MoE expert-FFN bmm** M×N co-split (2.6–3.1× in prior work) — not the
SwiGLU mul→down 2-D edge.

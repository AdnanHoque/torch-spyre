# Tier-1 Eligibility — MoE Token Routing Data Movement (Dispatch / Combine)

Offline classification (2026-05-24). No device, no compile, no dxp run. Decides
whether the two MoE routing data-movement pieces — **dispatch** (gather tokens per
expert) and **combine** (scatter expert outputs back to tokens) — are addressable
by the proven Tier-1 same-stick cross-core `STCDPOpLx` primitive, or whether they
are layout-changing (blocked, needs the Compute-CB-faulting transpose) or otherwise
not Tier-1-addressable.

Grounded in `CoreToCoreDataMovementRecipe.md` (the proven primitive; §4 the `memId`
mechanism; §12 applicability) and `/tmp/real_edge_analysis.md` (the same-stick
classification method via `stickDimOrder_`). Inferences are flagged **[INFER]**.

---

## 0. The one-line verdict

**The activation MOVEMENT in both dispatch and combine is SAME-STICK (Tier-1
shaped): the stick lives on the hidden dim, routing permutes only the token dim, so
stick orientation is preserved end-to-end.** BUT there is a hard caveat that makes
routing *not* a drop-in `STCDPOpLx` today: the permutation is **data-dependent**
(the destination core for a token is the runtime router output), whereas `STCDPOpLx`
moves logical slice *i* to a **statically-encoded** core via `memId`. So routing is
"Tier-1 eligible in layout, blocked in addressing" — see §4. This is the honest,
load-bearing distinction.

---

## 1. The MoE activation layout on Spyre

MoE activations are 2-D `[tokens, hidden]`. On Spyre the stick is **64 fp16
elements (128 B)** along the contiguous innermost dim, which is the **hidden** dim.
In the work-division / SDSC vocabulary used by `real_edge_analysis.md`:

- `out_` (the stick dim) = **hidden** H.  `stickDimOrder_ = ['out']`.
- `mb_` = the **token** axis T (the batch-like, row axis).

A token's hidden vector of length H spans `ceil(H/64)` whole sticks (H=2048 → 32
sticks/token; H=4096 → 64 sticks/token). **A token = an integer number of whole
sticks.** This is the crucial geometric fact for the verdict.

---

## 2. Dispatch (gather): is it same-stick?

Dispatch builds, for each expert, the buffer of tokens routed to it:

```
dispatched[r, :] = x[token_of_slot(r), :]      # r in [0, E*C),  : over hidden H
```

It selects **whole rows** of `x` (whole hidden vectors = whole sticks) and re-places
them at new row positions indexed by the router. Reasoning out the stick question
exactly as the task frames it:

- The stick is on the **hidden** dim. The gather permutes the **token** dim.
- A whole hidden vector (an integer number of sticks) is moved as a unit; **no stick
  is ever split, merged, or re-oriented.** `stickDimOrder_` of the source row and the
  destination row are both `['out']` (hidden).

**=> Dispatch is SAME-STICK.** It is a pure cross-core (and cross-row) re-placement
of whole sticks along the token axis — structurally identical to the same-stick move
`STCDPOpLx` performs (recipe §4: pieces matched by logical coordinate, placement set
by `memId`; the stick dim is untouched).

This matches the `real_edge_analysis.md` rule directly: an edge is same-stick iff
`prod stick == cons stick`. Here both are `['out']` (hidden) → same-stick, exactly
like the matmul-output → elementwise edges that classification found Tier-1-eligible.

**Contrast — what would make it layout-changing:** if the stick were on the *token*
dim (e.g. a `[hidden, tokens]` transposed layout, `stickDimOrder_ = ['mb']`), then
permuting tokens would permute *within* sticks, requiring a restickify — the blocked
`ReStickifyOpWithPTLx` transpose. Spyre's token-major `[T, H]` activation layout is
what keeps dispatch same-stick. **[INFER]** that the MoE activation reaches routing
in token-major `[T,H]` (hidden-stick) layout — the `x @ wg` router GEMM and the
expert FFNs both consume `[*, H]`, and `real_edge_analysis.md` consistently shows
post-matmul elementwise activations carrying `stickDimOrder_ = ['out']`, so the
hidden dim being the stick is the established convention.

---

## 3. Combine (scatter-add): is it same-stick?

Combine is the transpose-shaped operation: scatter each expert-slot output back to
its token row, weighted by the gate, accumulating across the top-k slots a token
landed in:

```
combined[t, :] = sum_{r : token_of_slot(r) == t} gate[r] * y[r, :]
```

Two parts:

1. **The MOVE** — `y[r, :]` (a whole hidden vector = whole sticks) goes to row `t`.
   Identical geometry to dispatch in reverse: whole sticks re-placed along the token
   dim, hidden untouched. **=> the move is SAME-STICK.**
2. **The ACCUMULATE** — for top-k > 1, two (or k) slots map to the same token and
   their hidden vectors are summed. This is an **elementwise add reduction**, which is
   a *compute* op, not a data move. In the proven mechanism this is the **DL op the
   STCDP feeds** inside the mixed SuperDSC (recipe §3: `dscs_` = the consumer DL op,
   `datadscs_` = the STCDP moves before it). For top-k = 1 there is no accumulation
   at all (pure scatter) — combine reduces to exactly the dispatch geometry inverted.

**=> Combine's data movement is SAME-STICK.** The scatter-add is a same-stick move
feeding an elementwise-add DL op — the canonical mixed-SuperDSC shape the recipe
proves (move-then-compute in one SDSC).

---

## 4. The load-bearing caveat: data-dependent vs static placement

This is where I must be explicit and honest. `STCDPOpLx` is same-stick-only **and**
its routing is **static**: each `PieceInfo` hard-codes `dimToStartCordinate` (logical
slice *i*) and `PlacementInfo.memId` (the physical core that holds slice *i*). The
move is "slice *i* → slice *i*, on cores set at compile time" (recipe §4). The
proof's reversed-ownership trick `i → 31-i` is a **compile-time** permutation baked
into the JSON.

MoE routing is **data-dependent**: which expert (and therefore which destination row,
and — once sharded — which destination core) a token goes to is the runtime output of
the router top-k. The mapping `token → slot → core` is not known at compile time.

So the two questions decouple:

| Question | Dispatch | Combine | Verdict |
|---|---|---|---|
| Is the activation move same-stick (stick orientation preserved)? | yes | yes (move part) | **Tier-1 SHAPED** |
| Crosses an SDSC boundary as an HBM round-trip Tier-1 eliminates? | yes (see §5) | yes (see §5) | **Tier-1 RELEVANT** |
| Can it be expressed as `STCDPOpLx` with **static** `memId` today? | **no** | **no** | **BLOCKED on addressing** |

**Conclusion:** dispatch and combine are **same-stick** — they do NOT need the
blocked transpose primitive, which is the good news and the direct answer to the core
question. But they are **not** addressable by `STCDPOpLx` as-built, because that
primitive moves *statically-placed* slices, and routing placement is *data-dependent*.
The missing capability is an **index-driven same-stick move** (a same-stick STCDP
whose per-piece `memId` / `dimToStartCordinate` is supplied by a runtime index buffer
rather than a compile-time constant). This is a *strictly smaller* gap than the
layout-changing/transpose frontier (§11a of the recipe): the data path (the RIU ring,
the `L3_LDU`/`L3_STU` units, the mixed SuperDSC) is exactly the proven one; only the
*addressing* needs to become dynamic. **[INFER]** that no dynamic-`memId` STCDP exists
today — the recipe describes only static `PieceInfo`, and the open-frontier list
(§11) does not mention index-driven moves, so I treat it as not-yet-available.

### 4.1 The matmul-permutation realization sidesteps the addressing gap (but loses Tier-1)

The microbench (`moe_routing_workload.py`) realizes dispatch/combine as **one-hot
permutation matmuls** (`perm @ x`, `perm_w.T @ y`) because that is what compiles on
Spyre today (index_select/embedding fall back to CPU — see the bench's docstring and
`torch_spyre/ops/fallbacks.py`). As a *matmul*, the permutation is no longer a pure
data move — it is a GEMM whose output goes to HBM and whose *output → next op* edge is
the Tier-1-addressable handoff (same-stick `['out']`, exactly the matmul-output →
elementwise pattern in `real_edge_analysis.md`). So:

- The **baseline** the orchestrator measures is the matmul-permutation cost (a real,
  device-runnable number).
- The **Tier-1 win** the projection estimates is for the *true* same-stick move
  (index-driven STCDP), which would replace the permutation-matmul-through-HBM with an
  on-chip same-stick re-placement — once the dynamic-`memId` capability exists.

These are two different things and the projection (`projection.md`) labels them.

---

## 5. SDSC boundary / HBM round-trip — the thing Tier-1 eliminates

Tier-1 eliminates a producer→consumer handoff that crosses an **SDSC boundary** and
therefore round-trips through HBM by default (recipe §2: LX *does* persist across
`sdsc_execute` in PF / single-user VF — measured; the round-trip is the planner
conservatively evicting to HBM at SDSC boundaries, a scheduling choice, not a hardware
wipe). Routing creates exactly such boundaries:

```
router GEMM  ->  [HBM]  ->  dispatch (gather)  ->  [HBM]  ->  expert FFN (bmm)
expert FFN   ->  [HBM]  ->  combine (scatter)  ->  [HBM]  ->  residual add
```

Each arrow into/out of dispatch/combine is a separate SDSC launch, so the activation
that dispatch gathers and the buffer it produces both transit HBM. The bytes that
cross these boundaries (per §6) are the activation tensors — `EC * H * 2` bytes for the
dispatch buffer, `T * H * 2` for the combined output — which are large (MB-scale) at MoE
shapes, putting them well above the recipe's ~1 MB net-positive threshold. So the
handoff is genuinely an HBM round-trip Tier-1 would target; the only blocker is the
data-dependent addressing (§4), not the layout and not the size.

---

## 6. How routing shards, and the activation bytes moved

MoE shards the same way `real_edge_analysis.md` and `project_bmm_aware_split.md`
describe: activations split along a work dim across the 32 cores; the dispatch buffer
is `[E*C, H]`, the expert FFNs are the true bmm (batch = experts, recipe §12.3 /
`project_bmm_aware_split`).

- **Dispatch produces** `[EC, H]` = `E*C*H*2` bytes. The cross-core *re-ownership*
  delta is whatever fraction of tokens land on a different core than they started on
  — for a uniform random router that is ~`(num_cores-1)/num_cores` ≈ 97% of rows → a
  genuine cross-core ring move (not a degenerate same-core copy). **[INFER]** the
  ~97% from a uniform-routing assumption; real routers are skewed but still
  predominantly cross-core.
- **Combine produces** `[T, H]` = `T*H*2` bytes, same re-ownership argument.

Representative byte volumes (fp16, `cap_fac=1.0`, top-k=1 so `EC ≈ T`):

| E | T | H | dispatch buf `EC*H*2` | combine out `T*H*2` |
|---|---|---|---|---|
| 8  | 2048 | 2048 | ~8.4 MB | ~8.4 MB |
| 8  | 2048 | 4096 | ~16.8 MB | ~16.8 MB |
| 64 | 4096 | 4096 | ~33.6 MB | ~33.6 MB |

(For top-k=2, the dispatch buffer doubles: `EC ≈ 2T`.) All are far above the ~1 MB
net-positive threshold — the regime where on-chip wins (recipe §0 anchor).

---

## 7. Summary table

| piece | move same-stick? | needs blocked transpose? | crosses HBM SDSC boundary? | `STCDPOpLx`-as-built today? | gap |
|---|---|---|---|---|---|
| **dispatch (gather)** | **YES** (hidden stick, token permute) | no | yes (router→dispatch→FFN) | **no** | data-dependent `memId` (index-driven move) |
| **combine (scatter-add)** | **YES** (move part) | no | yes (FFN→combine→residual) | **no** | data-dependent `memId` + the add is the DL op |

The verdict the task asks for: **dispatch and combine are same-stick (Tier-1
layout-eligible), NOT layout-changing — they avoid the Compute-CB transpose wall.
They are blocked only on a smaller, distinct gap: a dynamic/index-driven same-stick
move, since the proven `STCDPOpLx` places slices statically.** Honest bottom line:
the routing data movement is the *best-shaped* Tier-1 candidate by layout, but is one
capability (dynamic addressing) short of being addressable today — whereas the
attention score→softmax edge (`real_edge_analysis.md` §best target) is same-stick AND
statically addressable, so it remains the first demo.

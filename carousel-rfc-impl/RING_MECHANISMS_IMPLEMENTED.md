# Turning Three Ring-Aware Mechanisms Into Real Code

## 1. What we set out to do

Residency — pinning working tensors to LX scratchpad — is the trivial floor of Spyre performance. The interesting layer sits on top of it: **ring-aware algorithms** that move data across cores along the fabric in the shape the fabric is fast at. This session took three ranked ring-aware mechanisms and drove each as far toward real, committed code as the evidence honestly allowed.

The methodology was the same for all three:

1. **Scout the live code sites** and the exact device-free test recipe — not a sketch, the actual file/symbol and the test that exercises it.
2. **Implement in isolated git worktrees**, one per bet, so nothing cross-contaminates.
3. **Verify device-free where possible** (unit tests, reference-math checks, byte-identical ablations).
4. **Be honest-partial at the device / deeptools boundary** — where the compiler pipeline or the accelerator blocks end-to-end observation, say so and stop, rather than claim a win.
5. **Push to the `AdnanHoque` fork as branches, never PRs.**

The three bets, ranked:

- **Bet 2 — per-link contention cost term.** Make ring-aware plans *selectable at all* by teaching the matmul cost model the difference between a multicast and a scatter. **Done, device-free-verified, structurally safe.**
- **Bet 1 — flash-in-a-loop.** Keep online-softmax state LX-resident by exploiting the coarse-tile-loop-is-one-bundle property, so flash never materializes the full scores tensor. **Partially landed: the transpose demonstrates the win; the resident-scratch guard is still inert.**
- **Bet 3 — LSE ring-fold merge.** A genuinely novel collective: merge per-core KV-shard softmax partials with a neighbor reduce-scatter on the fast band instead of a linear fold. **Math + schedule device-free-verified; not compiler-wired.**

## 2. The device physics these play by (measured)

> **The rules of the fabric.** These are the invariants every bet is built against. The first is device-measured from prior sessions; the rest are established Spyre architecture.
>
> - **Uniform shift is fast; scatter is slow.** A uniform `p -> p+1` ring shift moves **1 transfer per link (~130 GB/s)**. An all-to-all scatter or a linear fold-to-one root piles **many transfers on one link (~36 GB/s floor)**. The variable that sets the band is **per-link transfer count**, not burst size.
> - **Multicast is live and proven.** `STCDPOpLx` one-to-many multicast is a real, exercised primitive on the fast band.
> - **HBM has no channel affinity.** It is one flat `memId=-1` space; you cannot cheat contention by picking a channel.
> - **Matmul is weight-stationary.** M streams past resident weights; M is never a stick.
> - **The cross-bundle wall.** LX does not persist across SDSC bundles. But **a coarse-tile loop is one bundle by construction**, so ops inside it co-bundle — and stay resident — for free, with no deeptools primitive.
> - **The cost model is contention-blind.** It prices every operand move at flat `bytes/128` with one lumped cohort penalty, so it **cannot tell a fast shift from a slow scatter.** Bet 2 fixes exactly this.

## 3. Bet 2 — per-link contention cost term

**Branch `ah/ring-cost-term`, commit `2dfb2cc`. Device-free-verified. Structurally no-regression.**

### The problem, first principles

The matmul cost model in `work_division.py` priced every operand move at flat `bytes/128`, then multiplied the *whole byte total* by a single lumped `cohort_penalty = max(1, max(m,n)/8)`. That prices the device-proven multicast (one-to-many, ~130 GB/s) identically to a scatter (~36 GB/s). With no way to express that a wide cohort can be *cheap* when the data is shared, the planner could never select a ring-aware wide-cohort plan. This is the concrete cause of the earlier-found mispricing where wide-N `(m2, n16)` cost *more* than `(m8, n4)`.

### The fix

Decompose the HBM term into its three real streams and price each by who actually shares it:

- **LHS `[M,K]`** has no N index → it is **identical across the n-cohort** → a one-to-many multicast.
- **RHS `[K,N]`** has no M index → it is **identical across the m-cohort** → a one-to-many multicast.
- **Output tile** is distinct per `(m,n)` → no sharing.

A new `_cohort_penalty(cohort, identical=True)` caps the shared streams at the fast band: `min(max(1, cohort/8), peak/130)`. A distinct-data scatter caps at `peak/36`. The output tile pays no penalty.

### What verification ran, and its result

- **7/7 new unit tests pass**, including the decisive one: wide-N `(m2, n16)` now costs *less* than `(m8, n4)` — the exact inversion found earlier is gone.
- **153 device-free inductor tests pass. Ruff clean.**
- **No-regression is structural, not empirical:** `_cohort_penalty` returns exactly `1.0` for `cohort <= 8`, so every already-banked case (shared-weight, min-cores) is **byte-identical** to before.

**Status: device-free-verified and structurally safe.** Not yet device-measured on a matmul sweep — that is the next step (§5).

## 4. Bet 1 — flash-in-a-loop

**Branch `ah/flash-ring`, commit `e47f268` + an uncommitted follow-up. Partially landed; the resident-scratch guard does not yet fire.**

This is a two-part story and both parts must be told straight.

### The problem, first principles

`test_flash.py:17` disallows coarse-tiling in `Lk` (it sets `kv_block_size = Lk // 1`). With `Lk` tiled by 1, the online-softmax loop runs **once** and materializes the full `[1, 32, 4096, 4096]` scores = **32 MB/head**, a ~12 ms HBM round-trip. That defeats the entire point of flash attention, whose value is *never* holding the full scores.

### Part (a) — the transpose (demonstrated the win)

An off-stick transpose — mirroring paged attention's `scores.transpose(-1,-2).contiguous()` "avoid stick reduction" move — clears the pre-codegen stick-incompatibility error and, on its own, **collapses the scores from `[1,32,4096,4096]` to `[1,4,1024,4096]`.** This part works and shows the mechanism is real.

### Part (b) — the resident-scratch guard (committed, but inert)

A committed inductor guard `_reduction_consumed_only_inside` in `coarse_tile.py` was meant to divert flash-style reductions off the fill+combine accumulation path and mark the per-tile `block_max`/`block_sum` as `per_tile_fixed` LX-resident scratch, so the running `(m, l, O)` online-softmax state co-bundles for free (the loop-is-one-bundle property from §2).

**The review caught that the committed guard was inert.** `_hints_levels` (`coarse_tile.py:95`) read `is_reduction` from only the *first* op in the group — the pointwise `keys*scale`, which has `is_reduction=False`. So the `Lk` level was never classified as a reduction level, and the guard's path was never taken.

### The follow-up fix, and whether it fires

The follow-up makes two coupled edits in `coarse_tile.py`:

1. **`_hints_levels`** now OR-reduces `is_reduction` across *every* op in the group. A level is a reduction level if the hinted dim reduces in *any* op (so `block_max`/`block_sum` reducing `Lk` counts, even though the leading pointwise op does not).
2. **`_stamp_group`** now chooses the tile branch **per-op, not from the global level flag.** Each op looks up both its reduction-range position and its output-range position; a reduction op tiles its reduction range, a pointwise op tiles its output range. This second edit was *required*, not optional: without it, once the level flips to reduction, the leading pointwise op falls into the reduction branch, finds no reduction range, and is left **un-tiled** — iterating full `Lk` every loop. This is exactly the fallback the task anticipated.

**Fix result, reported honestly: it does NOT fire. `committed: false` — no commit was created; the edit sits uncommitted in the worktree.**

The fix is logically correct for the stated root cause and passes **146 tests (0 failed)** across `test_coarse_tiling.py` and `test_unroll_loop_specs.py` — no regression. But it could not be observed firing on any reachable flash graph, because both vehicles are blocked upstream of the guard:

- **The faithful probe** (`tiles={"Lk"}`) aborts far upstream at `optimize_restickify_locations` (pass index 2) with `buf12 (Pointwise): no mechanism to resolve stick incompatibility` — the sparse `real_max`/`denominator` `amax(dim=-1)` gather. `_maybe_coarse_tile` is pass index 10, so **coarse_tile never runs.** This abort is **identical on base `b36236a`, the guard commit `e47f268`, and the fix** → pre-existing, cross-commit, independent of this change. (It is *not* the known end-of-pipeline deeptools/VFIO abort; it is much earlier.)
- **The reachable e2e test** `test_coarse_tile_e2e.py::test_hint_flash_attention` compiles fully and *does* reach coarse_tile — but its group's representative op has `Lk range=0 / loop_var=None`, so **`Lk` is never a level at all.** `_hints_levels` only reclassifies *existing* levels, so the OR-reduce has nothing to act on. **Ablation: base vs fix stamps are byte-identical.**

Per the task's own rule — *if it still does not fire, commit nothing* — nothing was committed. The premise (that the faithful probe reaches coarse_tile with only a deeptools/VFIO abort at the very end) did not hold in this environment; the probe dies at restickify instead, suggesting the author's confirming run used a different LD/senlib env where restickify succeeded.

**Status: transpose device-free-verified (scores collapse observed); resident-scratch guard unverified / inert; follow-up fix logically correct and test-green but not exercisable end-to-end here.**

## 5-context. Bet 3 — LSE ring-fold merge

**Branch `ah/flash-ring`, commit `b36236a`. Math + schedule device-free-verified; not compiler-wired.**

### The problem, first principles

Once `Lk` splits across **cores** (not just time), each core owns a KV shard and produces a partial `(m, l, O)`. Merging those partials by a **linear fold-to-one root** piles the full payload on one link and hits the **36 GB/s contention floor** — the exact scatter pathology from §2.

### The fix

Use a **neighbor reduce-scatter** instead: 1 transfer per link, staying on the fast ~130 GB/s band. It also lands the output **head-split** — precisely the layout the O-projection wants, so the merge is **relayout-free**.

Frontend composition (buildable now): ring order + per-hop pair = an `STCDPOpLx` cross-core move plus a separate SFP `lse_combine` sub-run; the exact fp32 `(m, l, A)` combine with dead-lane guard; reduce-scatter as `P-1` combining hops plus an optional all-gather.

The **deeptools ask** — a single fused move-then-reduce hop, *only if* the composed per-hop overhead proves too high — was **written to `deeptools-change-log.md`, not hacked in.**

### What verification ran, and its result

- **13/13 reference checks pass**, including: the fold equals a single-pass softmax (`rtol 1e-4`), every ring round is distance-1, and the endpoint is head-split.

**Status: math and schedule device-free-verified; the collective itself is modeled — not yet wired into the compiler and not device-measured.**

## 4-throughline. The throughline

Residency is the floor: pinning to LX is table stakes and mechanically easy. The interesting engineering is the **ring-aware algorithm** layered on top.

- **Bet 1** exploits the coarse-tile-loop-is-one-bundle property to keep online-softmax state resident with **no new primitive** — pure inductor.
- **Bet 2** is the enabler: without a contention-aware cost term, no ring-aware plan is even *selectable*.
- **Bet 3** is the genuinely novel collective — a neighbor reduce-scatter fold that stays on the fast band *and* lands the layout the next op wants.

Two of the three are prototype-now, pure-inductor. One is design-complete with a precise, minimal deeptools ask.

## 5. Status and next verifiable step

| Bet | Branch / commit | Verification level | Honest status |
|---|---|---|---|
| **2 — per-link contention cost term** | `ah/ring-cost-term` `2dfb2cc` | **Device-free-verified** (7/7 unit + 153 inductor tests, ruff); structural no-regression | Done; selects the wide-N plan the old model mispriced |
| **1 — flash-in-a-loop (transpose)** | `ah/flash-ring` `e47f268` | **Device-free-verified** (scores collapse `[1,32,4096,4096]→[1,4,1024,4096]`) | Landed; the win is demonstrated pre-codegen |
| **1 — flash resident-scratch guard + fix** | `ah/flash-ring` `e47f268` + uncommitted | **Unverified** (guard inert; fix test-green at 146 passed but does not fire e2e) | **Not committed** — blocked upstream, per the commit-nothing rule |
| **3 — LSE ring-fold merge** | `ah/flash-ring` `b36236a` | **Device-free-verified math** (13/13 reference checks); collective **modeled**, not wired | Design-complete; precise deeptools ask logged, not hacked |

**Underlying physics** (~130 GB/s shift vs ~36 GB/s scatter) is **device-measured** from prior sessions and is the shared premise all three rest on.

### Next verifiable step per bet

- **Bet 2:** Device-measure a matmul sweep (KV-proj `[4096,1024]` wide-N and the shared-weight banked cases) to confirm the newly-selected plan is faster on-device and the `cohort<=8` cases are unchanged. This is the one bet ready for a device number.
- **Bet 1:** Resolve the two upstream blockers before the guard can fire e2e — (1) `optimize_restickify` must handle the faithful probe's sparse `real_max`/`denominator` `amax(dim=-1)` multi-stick→single-stick gather (a restickify/layout task); (2) `assign_dim_hints` + `hints_to_coarse_tile_groups` must surface an `Lk` tiling *level* whose representative op iterates `Lk` (`loop_var != None`) while a peer reduces it. Only then does the OR-reduce classification have something to reclassify. Real `Lk` tiling is also still gated by the in-code "current limitation disallows coarse tiling in Lk."
- **Bet 3:** Wire the composed frontend hop (`STCDPOpLx` move + SFP `lse_combine`) into a real cross-core `Lk`-split flash graph and measure the per-hop overhead — the number that decides whether the fused-hop deeptools ask is worth filing.

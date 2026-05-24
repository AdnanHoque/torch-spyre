# MoE Block Activation-Handoff Eligibility (on-chip core-to-core)

Offline analysis (2026-05-24). No device, no compile, no dxp. Classifies the
producer->consumer activation handoffs in `moe_block_workload.py` /
`moe_ffn_workload.py` by whether the proven same-stick `STCDPOpLx` core-to-core
primitive can carry them on-chip, vs whether they are layout-changing (blocked on
the Compute-CB-faulting `ReStickifyOpWithPTLx` transpose), vs graph-input/weight
prelayout.

Classification is **analytical**, derived from (a) the op structure of the
workloads, (b) Agent C's real-edge classification of the SAME op patterns in
already-compiled granite/SDPA bundles (`/tmp/real_edge_analysis.md`), and (c) the
bmm M×N co-split sharding from `project_bmm_aware_split`. It is **not** traced
from a compiled bundle of these specific workloads (we did not compile). Each
inferred sharding is flagged. Tensor sizes are exact (fp16, 2 B/elem).

## How the classification rule maps to the recipe

From `CoreToCoreDataMovementRecipe.md` §3.2/§7c and `real_edge_analysis.md`:

- **same-stick + same-shard** -> producer and consumer own the same per-core
  slice. `STCDPOpLx` eliminates the HBM round-trip as a degenerate same-core
  LX->LX copy; the RIU ring ops are dead-code-eliminated (zero `L3_LDU`/`L3_STU`).
  Pure HBM-elimination, cheapest, no ring.
- **same-stick + cross-core** -> same `stickDimOrder_`, but producer and consumer
  shard differently (different `memId` ownership). Genuine RIU-ring `STCDPOpLx`
  (the proven `i -> 31-i -> i` class). This is the headline addressable case.
- **layout-changing** -> `stickDimOrder_` flips (e.g. `out`->`in` entering a
  matmul on the contracted axis, or a reduction-reshape `out`<->`x`). Needs the
  transpose op, which faults Compute-CB today -> **BLOCKED**.
- **prelayout** -> source is a graph weight/input (16-GiB-aligned segment
  marker). Solved by input/weight prelayout in inductor, not a runtime primitive.

The structural rule that decides same-stick-but-cross-core: **a bmm shards M×N
co-split, so its output is owned per-core along `out`/`in`; the immediately
following elementwise/softmax/combine op reshards along the token (`mb`/`x`)
axis.** Stick orientation is preserved (`['out']` on both) but ownership differs
-> same-stick cross-core ring. This is exactly the dominant pattern Agent C found
(matmul-output -> elementwise/softmax). It recurs at every bmm output in the MoE
expert FFN.

---

## Activation tensor sizes (the handoff payloads)

Default block shape `B=1, Sq=128, H=2048, INTER=8192, E=8, K=2` (Tk = B*Sq = 128
tokens, NH=16):

| activation | shape | size |
|---|---|---|
| attn norm out / qkv-proj (each) / attn-out | `[B,Sq,H]` | 0.524 MB |
| sdpa scores (fused, per QK^T) | `[B,NH,Sq,Sq]` | ~0.524 MB |
| router logits | `[Tk,E]` | 0.002 MB |
| combine/dispatch matrix | `[Tk,E]` | 0.002 MB |
| **dispatch xe (expert token stack)** | `[E,Tk,H]` | **4.194 MB** |
| **gate bmm out** | `[E,Tk,INTER]` | **16.78 MB** |
| **up bmm out** | `[E,Tk,INTER]` | **16.78 MB** |
| **act (silu(gate)*up)** | `[E,Tk,INTER]` | **16.78 MB** |
| **ye (down bmm out) -> combine** | `[E,Tk,H]` | **4.194 MB** |
| moe_out (combined) | `[Tk,H]` | 0.524 MB |

The MoE MLP activations dwarf the attention activations by 1-2 orders of
magnitude (the E× expert replication + INTER expansion). At prefill
(`Sq=512, H=4096, INTER=14336`) the gate/up/act tensors are **117 MB each**. This
is why MoE is the high-leverage on-chip target: it moves a *lot* of activation
data between the bmm-output-owning cores and the downstream consumers.

---

## Per-edge classification — MoE MLP path (the core of the ask)

| # | producer -> consumer | tensor | size (default) | stick (prod->cons) | shard (prod -> cons) | class | on-chip? |
|---|---|---|---|---|---|---|---|
| M1 | rmsnorm(h) -> router matmul (`xf @ w_router`) | `[Tk,H]` | 0.524 MB | `['out']` -> `['in']` | `{mb:32}` -> bmm `{out,in}` | **layout-changing** | BLOCKED (out->in into matmul) |
| M2 | rmsnorm(h) -> dispatch xe (`expand`) | `[Tk,H]`->`[E,Tk,H]` | 0.524->4.19 MB | `['out']` -> `['out']` | `{mb:32}` -> `{?}` | **same-stick cross-core** (INFER) | YES (broadcast remap) |
| M3 | router logits -> softmax | `[Tk,E]` | 0.002 MB | `['out']` -> `['out']` | same | same-stick same-shard | yes but tiny (<1 MB) |
| M4 | softmax -> topk | `[Tk,E]` | 0.002 MB | `['out']` -> `['out']` | same | same-stick same-shard | yes but tiny |
| M5 | topk gates/idx -> scatter (combine matrix) | `[Tk,K]`,`[Tk,E]` | 0.002 MB | mixed | reshard | **layout-changing** (scatter) | BLOCKED, but tiny |
| **M6** | **dispatch xe -> gate bmm** | `[E,Tk,H]` | **4.19 MB** | `['out']` -> `['in']` | `{?}` -> bmm `{out,in}` | **layout-changing** (into matmul) | BLOCKED (transpose-before-bmm) |
| **M7** | **dispatch xe -> up bmm** | `[E,Tk,H]` | **4.19 MB** | `['out']` -> `['in']` | `{?}` -> bmm `{out,in}` | **layout-changing** (into matmul) | BLOCKED |
| **M8** | **gate bmm out -> silu (act)** | `[E,Tk,INTER]` | **16.78 MB** | `['out']` -> `['out']` | bmm `{out,in}` -> elemwise `{mb}` | **same-stick cross-core** (INFER) | **YES (best target)** |
| **M9** | **up bmm out -> mul (act)** | `[E,Tk,INTER]` | **16.78 MB** | `['out']` -> `['out']` | bmm `{out,in}` -> elemwise `{mb}` | **same-stick cross-core** (INFER) | **YES** |
| **M10** | **act -> down bmm** | `[E,Tk,INTER]` | **16.78 MB** | `['out']` -> `['in']` | elemwise `{mb}` -> bmm `{out,in}` | **layout-changing** (into matmul) | BLOCKED (transpose-before-bmm) |
| **M11** | **down bmm out (ye) -> combine mul** | `[E,Tk,H]` | **4.19 MB** | `['out']` -> `['out']` | bmm `{out,in}` -> elemwise `{mb}` | **same-stick cross-core** (INFER) | **YES (combine edge)** |
| M12 | combine matrix -> combine mul (`cw`) | `[E,Tk,1]` | 0.001 MB | `['out']` | reshard (transpose `.t()`) | **layout-changing** (transpose) | BLOCKED, but tiny |
| M13 | weighted ye -> sum over experts (reduce dim 0) | `[E,Tk,H]`->`[Tk,H]` | 4.19->0.52 MB | `['out']` -> `['out']`/`['x']` | reduce over E (batch) | **same-stick cross-core** (reduce-combine) (INFER) | PARTIAL (see note) |
| M14 | moe_out -> residual add | `[Tk,H]` | 0.524 MB | `['out']` -> `['out']` | `{mb}` -> `{mb}` | same-stick same-shard | YES |

### The two edges the ask specifically flags

- **Router -> dispatch (M1 / M2).** Two sub-edges. The router *matmul* input
  (M1: rmsnorm out -> `@ w_router`) is **layout-changing** — it enters a matmul
  on the contracted axis (`out`->`in`), the same blocked pattern Agent C saw at
  every `activation -> bmm.in` edge. The *dispatch* itself (M2: the `expand` to
  `[E,Tk,H]`) is same-stick (a pure broadcast/re-ownership, no stick flip) and is
  **on-chip-eligible** as a cross-core remap — but it is small (0.5->4 MB) and the
  scatter that builds the combine matrix (M5) is layout-changing (and tiny). Net:
  **the router->dispatch handoff is dominated by a layout-changing matmul-input
  edge (BLOCKED) plus a small eligible broadcast.**

- **Expert-FFN bmm output -> combine (M11).** This is the down-projection bmm
  output `ye[E,Tk,H]` feeding the combine multiply. It is **same-stick**
  (`['out']` on both producer and the elementwise combine consumer) and
  **cross-core** (bmm shards `{out,in}`, combine elementwise reshards `{mb}`) ->
  genuine RIU-ring `STCDPOpLx`. **4.19 MB at default, 33.5 MB at prefill — well
  above the ~1 MB net-positive threshold. This is on-chip-eligible and one of the
  two best targets.** The reduce-over-experts that follows (M13) keeps the same
  stick but contracts the batch (E) axis; the per-slice move is same-stick
  cross-core, eligible in the same way the elementwise combine is, though the
  cross-batch reduction adds a partial-sum accumulation the recipe's pure-data
  move does not by itself express (flagged PARTIAL).

### Honest blocked-edge flag (the uncomfortable part)

**The three biggest MoE tensors (M6/M7 gate&up bmm inputs, M10 down bmm input —
the act tensor at 16.78 MB) are all `activation -> bmm.in` edges and are
layout-changing -> BLOCKED.** They flip `out`->`in` to feed the contracted axis
of the next matmul. These are precisely the transpose-before-bmm edges Agent C
flagged as needing `ReStickifyOpWithPTLx`, which faults Compute-CB. So the single
largest data movers in the MoE FFN (the 16.78 MB act tensor entering the down
projection, and the dispatch entering gate/up) are **not** addressable today.

What **is** addressable today are the bmm-OUTPUT -> elementwise edges:
**M8, M9 (16.78 MB each, gate/up -> SwiGLU act), and M11 (4.19 MB, down ->
combine).** These are the same-stick cross-core class proven on device.

---

## Per-edge classification — attention + norm path

These reuse Agent C's already-traced classifications of the identical op patterns
(SDPA bundle, granite RMSNorm block in `/tmp/real_edge_analysis.md`); sizes are
from this block's default shape.

| # | producer -> consumer | tensor | size | class | on-chip? |
|---|---|---|---|---|---|
| A1 | x -> rmsnorm (mean/rsqrt chain) | `[B,Sq,H]` | 0.524 MB | mostly same-stick same-shard; one `out`<->`x` reduction reshape is layout-changing (RMSNorm tail) | mixed (norm chain mostly yes; reduction reshape BLOCKED) |
| A2 | rmsnorm out -> q/k/v proj matmul | `[B,Sq,H]` | 0.524 MB | **layout-changing** (`out`->`in` into matmul) | BLOCKED |
| A3 | qkv proj out -> sdpa | `[B,NH,Sq,HD]` | 0.524 MB | layout-changing (transpose/view marshalling into fused SDPA) | BLOCKED |
| A4 | sdpa scores (QK^T) -> softmax (`max`/`sub`) | `[B,NH,Sq,Sq]` | ~0.524 MB | **same-stick cross-core** (Agent C's single best target; bmm `{mb}` -> softmax `{x}`) | **YES** (but fused inside one SDPA SDSC here) |
| A5 | softmax internal chain (max/sub/exp/sum/div) | `[B,NH,Sq,Sq]` | ~0.524 MB | same-stick (mostly same-shard) | yes (but intra-SDPA-fusion) |
| A6 | sdpa out (PV) -> output proj `@ wo` | `[B,Sq,H]` | 0.524 MB | layout-changing (`out`->`in` into matmul) | BLOCKED |
| A7 | attn out -> residual add | `[B,Sq,H]` | 0.524 MB | same-stick same-shard | YES |
| A8 | rmsnorm/residual elementwise chain | `[Tk,H]` | 0.524 MB | same-stick same-shard | YES |

Caveat: on Spyre, attention is a **single fused SDPA SDSC** (per
`project_bmm_aware_split` and Agent C), so A4/A5 are *inside* one SDSC and never
cross an SDSC boundary at runtime — they are not separate HBM handoffs in the
fused kernel. The score->softmax same-stick cross-core edge is real and the best
attention target *if* SDPA were decomposed, but in the fused form there is no
inter-SDSC HBM handoff there to eliminate. The addressable attention handoffs are
the residual/norm same-stick edges (A1 partial, A7, A8).

---

## Summary table

| class | MoE MLP edges | attn+norm edges | addressable today? |
|---|---|---|---|
| same-stick same-shard (HBM-elim, no ring) | M3, M4, M14 | A1 (partial), A7, A8 | YES (cheapest) |
| **same-stick cross-core (RIU ring `STCDPOpLx`)** | **M2, M8, M9, M11, M13** | A4 (if SDPA split) | **YES (proven primitive)** |
| layout-changing (transpose, Compute-CB BLOCKED) | M1, M5, M6, M7, M10, M12 | A2, A3, A6, RMSNorm reshape | NO (blocked) |
| prelayout (weight/input) | w_router, w_gate/up/down restickify | wq/wk/wv/wo restickify | inductor prelayout |

### The big-tensor verdict (what actually moves the needle)

Ranked by payload at default shape, the on-chip-**eligible** same-stick edges:

| edge | tensor | default | prefill (Sq=512,H=4096,INTER=14336) |
|---|---|---|---|
| M8 gate-bmm-out -> act | `[E,Tk,INTER]` | 16.78 MB | 117 MB |
| M9 up-bmm-out -> act | `[E,Tk,INTER]` | 16.78 MB | 117 MB |
| M11 down-bmm-out (ye) -> combine | `[E,Tk,H]` | 4.19 MB | 33.5 MB |
| M2 rmsnorm -> dispatch (expand) | `[E,Tk,H]` | 4.19 MB | 33.5 MB |

The on-chip-**blocked** layout-changing big tensors:

| edge | tensor | default | prefill |
|---|---|---|---|
| M10 act -> down-bmm.in | `[E,Tk,INTER]` | 16.78 MB | 117 MB |
| M6/M7 dispatch -> gate/up-bmm.in | `[E,Tk,H]` | 4.19 MB each | 33.5 MB each |

**The MoE FFN's largest activation (the 16.78 MB act tensor, 117 MB at prefill)
appears on BOTH an eligible edge (M8/M9, bmm-out->silu) and a blocked edge (M10,
act->bmm.in).** So roughly half the act-tensor traffic is addressable today
(produce-side) and half is blocked (consume-into-matmul side). The combine edge
(M11) is fully eligible.

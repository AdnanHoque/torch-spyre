# Real-Model Activation Handoff Classification (Spyre AIU)

Offline analysis (2026-05-24). No device, no compile, no dxp run. Classifies producer->consumer HBM handoffs in three real compiled fused kernels by whether the proven same-stick cross-core `STCDPOpLx` primitive can address them today, vs whether they need the (Compute-CB-faulting) `ReStickifyOpWithPTLx` transpose, vs whether they are graph-input/weight restickifies better solved by prelayout.

## Method and a load-bearing schema note

The cached inductor SDSC JSONs do **NOT** carry `hbmStartAddress_` on their DL `labeledDs_` entries (verified: `grep hbmStartAddress` returns nothing on any real bundle SDSC, and even on the recipe's own `sdsc_fused_add_mm_t` baseline cache). The `hbmStartAddress_ = 8388608` matching described in `splice_2048_*.py` operates on the **dxp-resolved** spliced files, not the cache. In the cache, the only place a resolved HBM address appears is the `scheduleTree_` `allocate` node's `startAddressCoreCorelet_.data_["[0, 0, 0]"]` (the per-core HBM base). Edges were therefore traced by matching a producer's OUTPUT allocate-node HBM base to a consumer's INPUT allocate-node HBM base (latest-prior-producer wins, to handle buffer reuse). Stick orientation comes from `primaryDsInfo_[role].stickDimOrder_` (keyed by the tensor's `dsType_` role INPUT/KERNEL/OUTPUT), and sharding from the SuperDSC `numWkSlicesPerDim_`.

**Address ranges:** activation scratch buffers live at low offsets (~0.5-37 MB, the `output` segment); addresses that are exact multiples of 16 GiB (17179869184=16GiB, 34359738368=32GiB, ... = `2^34, 2^35, ...`) are **symbolic graph-segment base markers** for graph inputs / weights / consts / graph outputs (not intra-bundle activations).

## Bundle: Granite RMSNorm + linear block

`/tmp/granite_inductor/inductor-spyre/sdsc_fused_add_linear_mul_rms_norm_6_m56h1rzb`

Execution order: [0]ReStickifyOpHBM -> [1]batchmatmul -> [2]mul -> [3]ReStickifyOpHBM -> [4]batchmatmul -> [5]mul -> [6]add -> [7]mul -> [8]mean -> [9]add -> [10]rsqrt -> [11]mul -> [12]mul

| producer | consumer | via ReStickifyOpHBM? | prod stick | cons stick | same-stick? | prod shard | cons shard | same-shard? | verdict |
|---|---|---|---|---|---|---|---|---|---|
| [0]ReStickifyOpHBM | [1]batchmatmul | yes | ['mb'] | ['out'] | no | {'mb': 25, 'out': 1} | {'x': 1, 'out': 8, 'in': 4} | no | prelayout-bucket (weight/input restickify) |
| [1]batchmatmul | [2]mul | no | ['out'] | ['out'] | YES | {'x': 1, 'out': 8, 'in': 4} | {'mb': 1, 'out': 25} | no | STCDP-today |
| [2]mul | [4]batchmatmul | no | ['out'] | ['in'] | no | {'mb': 1, 'out': 25} | {'x': 1, 'out': 16, 'in': 2} | no | needs-transpose (layout-changing activation handoff) |
| [3]ReStickifyOpHBM | [4]batchmatmul | yes | ['mb'] | ['out'] | no | {'mb': 1, 'out': 25} | {'x': 1, 'out': 16, 'in': 2} | no | prelayout-bucket (weight/input restickify) |
| [4]batchmatmul | [5]mul | no | ['out'] | ['out'] | YES | {'x': 1, 'out': 16, 'in': 2} | {'mb': 32, 'out': 1} | no | STCDP-today |
| [5]mul | [6]add | no | ['out'] | ['out'] | YES | {'mb': 32, 'out': 1} | {'mb': 32, 'out': 1} | YES | STCDP-today |
| [6]add | [7]mul | no | ['out'] | ['out'] | YES | {'mb': 32, 'out': 1} | {'mb': 32, 'out': 1} | YES | STCDP-today |
| [7]mul | [8]mean | no | ['out'] | ['out'] | YES | {'mb': 32, 'out': 1} | {'mb': 32, 'out': 1} | YES | STCDP-today |
| [8]mean | [9]add | no | ['out'] | ['x'] | no | {'mb': 32, 'out': 1} | {'out': 32, 'x': 1} | no | needs-transpose (layout-changing activation handoff) |
| [9]add | [10]rsqrt | no | ['x'] | ['x'] | YES | {'out': 32, 'x': 1} | {'out': 32, 'x': 1} | YES | STCDP-today |
| [6]add | [11]mul | no | ['out'] | ['out'] | YES | {'mb': 32, 'out': 1} | {'mb': 32, 'out': 1} | YES | STCDP-today |
| [10]rsqrt | [11]mul | no | ['x'] | ['out'] | no | {'out': 32, 'x': 1} | {'mb': 32, 'out': 1} | no | needs-transpose (layout-changing activation handoff) |
| [11]mul | [12]mul | no | ['out'] | ['out'] | YES | {'mb': 32, 'out': 1} | {'mb': 32, 'out': 1} | YES | STCDP-today |

## Bundle: SDPA attention

`/tmp/torchinductor_adnan/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable_0_451ht_5h`

Execution order: [0]identity -> [1]mul -> [2]mul -> [3]ReStickifyOpHBM -> [4]batchmatmul -> [5]max -> [6]sub -> [7]exp -> [8]sum -> [9]realdiv -> [10]batchmatmul -> [11]identity

| producer | consumer | via ReStickifyOpHBM? | prod stick | cons stick | same-stick? | prod shard | cons shard | same-shard? | verdict |
|---|---|---|---|---|---|---|---|---|---|
| [0]identity | [1]mul | no | ['out'] | ['out'] | YES | {'mb': 1, 'x': 32, 'out': 1} | {'mb': 1, 'x': 32, 'out': 1} | YES | STCDP-today |
| [2]mul | [3]ReStickifyOpHBM | no | ['x'] | ['x'] | YES | {'mb': 1, 'x': 1, 'out': 32} | {'mb': 32, 'x': 1, 'out': 1} | no | STCDP-today |
| [1]mul | [4]batchmatmul | no | ['out'] | ['in'] | no | {'mb': 1, 'x': 32, 'out': 1} | {'x': 1, 'mb': 32, 'out': 1, 'in': 1} | no | needs-transpose (layout-changing activation handoff) |
| [3]ReStickifyOpHBM | [4]batchmatmul | yes | ['out'] | ['out'] | YES | {'mb': 32, 'x': 1, 'out': 1} | {'x': 1, 'mb': 32, 'out': 1, 'in': 1} | YES | STCDP-today |
| [4]batchmatmul | [5]max | no | ['out'] | ['out'] | YES | {'x': 1, 'mb': 32, 'out': 1, 'in': 1} | {'mb': 1, 'x': 32, 'out': 1} | no | STCDP-today |
| [4]batchmatmul | [6]sub | no | ['out'] | ['out'] | YES | {'x': 1, 'mb': 32, 'out': 1, 'in': 1} | {'mb': 1, 'x': 32, 'out': 1} | no | STCDP-today |
| [5]max | [6]sub | no | ['out'] | ['out'] | YES | {'mb': 1, 'x': 32, 'out': 1} | {'mb': 1, 'x': 32, 'out': 1} | YES | STCDP-today |
| [6]sub | [7]exp | no | ['out'] | ['out'] | YES | {'mb': 1, 'x': 32, 'out': 1} | {'mb': 1, 'x': 32, 'out': 1} | YES | STCDP-today |
| [7]exp | [8]sum | no | ['out'] | ['out'] | YES | {'mb': 1, 'x': 32, 'out': 1} | {'mb': 1, 'x': 32, 'out': 1} | YES | STCDP-today |
| [7]exp | [9]realdiv | no | ['out'] | ['out'] | YES | {'mb': 1, 'x': 32, 'out': 1} | {'mb': 1, 'x': 32, 'out': 1} | YES | STCDP-today |
| [8]sum | [9]realdiv | no | ['out'] | ['out'] | YES | {'mb': 1, 'x': 32, 'out': 1} | {'mb': 1, 'x': 32, 'out': 1} | YES | STCDP-today |
| [9]realdiv | [10]batchmatmul | no | ['out'] | ['in'] | no | {'mb': 1, 'x': 32, 'out': 1} | {'x': 1, 'mb': 32, 'out': 1, 'in': 1} | no | needs-transpose (layout-changing activation handoff) |
| [10]batchmatmul | [11]identity | no | ['out'] | ['out'] | YES | {'x': 1, 'mb': 32, 'out': 1, 'in': 1} | {'mb': 1, 'x': 32, 'out': 1} | no | STCDP-today |

## Bundle: Full attention + RMSNorm block (with transpose)

`/tmp/granite_inductor/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_transpose_unsqueeze_view_2_jfvth_by`

Execution order: [0]ReStickifyOpHBM -> [1]batchmatmul -> [2]identity -> [3]batchmatmul -> [4]identity -> [5]ReStickifyOpHBM -> [6]batchmatmul -> [7]mul -> [8]add -> [9]mul -> [10]mean -> [11]add -> [12]rsqrt -> [13]mul

| producer | consumer | via ReStickifyOpHBM? | prod stick | cons stick | same-stick? | prod shard | cons shard | same-shard? | verdict |
|---|---|---|---|---|---|---|---|---|---|
| [0]ReStickifyOpHBM | [1]batchmatmul | yes | ['mb'] | ['out'] | no | {'mb': 1, 'out': 32} | {'x': 1, 'out': 16, 'in': 2} | no | prelayout-bucket (weight/input restickify) |
| [1]batchmatmul | [2]identity | no | ['out'] | ['out'] | YES | {'x': 1, 'out': 16, 'in': 2} | {'mb': 1, 'x': 1, 'y': 32, 'out': 1} | no | prelayout-bucket (graph-segment marker) |
| [2]identity | [3]batchmatmul | no | ['out'] | ['out'] | YES | {'mb': 1, 'x': 1, 'y': 32, 'out': 1} | {'x': 1, 'mb': 32, 'out': 1, 'in': 1} | no | STCDP-today |
| [3]batchmatmul | [4]identity | no | ['out'] | ['out'] | YES | {'x': 1, 'mb': 32, 'out': 1, 'in': 1} | {'mb': 1, 'x': 32, 'out': 1} | no | STCDP-today |
| [4]identity | [6]batchmatmul | no | ['out'] | ['in'] | no | {'mb': 1, 'x': 32, 'out': 1} | {'x': 1, 'out': 16, 'in': 2} | no | needs-transpose (layout-changing activation handoff) |
| [5]ReStickifyOpHBM | [6]batchmatmul | yes | ['mb'] | ['out'] | no | {'mb': 32, 'out': 1} | {'x': 1, 'out': 16, 'in': 2} | no | prelayout-bucket (weight/input restickify) |
| [6]batchmatmul | [7]mul | no | ['out'] | ['out'] | YES | {'x': 1, 'out': 16, 'in': 2} | {'mb': 32, 'out': 1} | no | STCDP-today |
| [7]mul | [8]add | no | ['out'] | ['out'] | YES | {'mb': 32, 'out': 1} | {'mb': 32, 'out': 1} | YES | STCDP-today |
| [8]add | [9]mul | no | ['out'] | ['out'] | YES | {'mb': 32, 'out': 1} | {'mb': 32, 'out': 1} | YES | STCDP-today |
| [9]mul | [10]mean | no | ['out'] | ['out'] | YES | {'mb': 32, 'out': 1} | {'mb': 32, 'out': 1} | YES | STCDP-today |
| [10]mean | [11]add | no | ['out'] | ['x'] | no | {'mb': 32, 'out': 1} | {'out': 32, 'x': 1} | no | needs-transpose (layout-changing activation handoff) |
| [11]add | [12]rsqrt | no | ['x'] | ['x'] | YES | {'out': 32, 'x': 1} | {'out': 32, 'x': 1} | YES | STCDP-today |
| [8]add | [13]mul | no | ['out'] | ['out'] | YES | {'mb': 32, 'out': 1} | {'mb': 32, 'out': 1} | YES | STCDP-today |
| [12]rsqrt | [13]mul | no | ['x'] | ['out'] | no | {'out': 32, 'x': 1} | {'mb': 32, 'out': 1} | no | needs-transpose (layout-changing activation handoff) |

## Summary across all three real bundles

| class | count | addressable how |
|---|---|---|
| **same-stick (STCDP-today)** | 27 | `STCDPOpLx` cross-core ring move, proven on device |
| **layout-changing (needs-transpose, BLOCKED)** | 8 | `ReStickifyOpWithPTLx` — faults Compute-CB today |
| **graph-input/weight/marker (prelayout-bucket)** | 5 | input/weight prelayout in inductor; no runtime primitive |
| total handoff edges | 40 | |

### Same-stick edges split by ring need (the count that actually matters)

A "same-stick" verdict means `STCDPOpLx` is the right primitive, but two sub-cases differ in cost:

- **same-stick + same-shard** -> degenerate same-core LX->LX copy. STCDP eliminates the HBM round-trip but emits **zero** `L3_LDU`/`L3_STU` (ring transfers dead-code-eliminated, per recipe section 7c). Pure HBM-elimination, no ring.
- **same-stick + different-shard** -> genuine **cross-core ring** STCDP (producer and consumer own different slices, so data must hop cores). This is exactly the proven `i -> 31-i -> i` round-trip class.

Counting the 27 STCDP-today edges this way (excluding graph-segment-marker reuse, latest-prior-producer dedup):

| bundle | same-stick same-shard (same-core, HBM-elim only) | same-stick diff-shard (cross-core ring) |
|---|---|---|
| Granite RMSNorm+linear | 6 | 2 (`[1]bmm->[2]mul`, `[4]bmm->[5]mul`) |
| SDPA attention | 7 | 4 (`[4]bmm(QK)->[5]max`, `[4]bmm(QK)->[6]sub`, `[10]bmm(PV)->[11]identity`, `[2]mul->[3]restickify`) |
| Attn+RMSNorm | 5 | 3 (`[2]id->[3]bmm`, `[3]bmm->[4]id`, `[6]bmm->[7]mul`) |

The cross-core-ring same-stick edges are almost all **matmul-output -> elementwise/softmax** edges: the bmm shards along `out`/`in`, the downstream elementwise op reshards along `mb` or `x`, so the same-stick activation must travel cores.

## Honest assessment

**Does the same-stick STCDP primitive apply to real transformer blocks? Yes, decisively, and for the majority of handoffs.** Across three distinct real fused kernels (a Granite RMSNorm+linear block, a standalone SDPA attention, and a full attention+RMSNorm block) **27 of 40** traced producer->consumer handoff edges are same-stick (`stickDimOrder_` identical on both endpoints) and are addressable today by `STCDPOpLx` once the Foundation contract (the minimal dxp gate+dispatch patch) is productionized. Only **8 of 40** are genuinely layout-changing (the stick dim flips, e.g. `out`->`in` before a matmul, or the `out`->`x` reduction-reshape in RMSNorm), which require the `ReStickifyOpWithPTLx` transpose that currently faults Compute-CB. The remaining **5** are graph-input/weight restickifies (a `ReStickifyOpHBM` whose source is a 16-GiB-aligned segment marker = a graph weight), which belong to the prelayout bucket and need no runtime primitive at all. This reproduces codex's earlier bucketing (~52% graph-input/weight, small fundamental-post-compute tail) on the activation side: the post-compute activation handoffs are dominated by same-stick edges, and the layout-changing ones cluster tightly at the two structural points where the stick orientation must flip (entering a matmul on the contracted axis, and the RMSNorm reduction).

**Crucial structural finding about the `ReStickifyOpHBM` ops.** Of the 5 `ReStickifyOpHBM` SDSCs across these bundles, **4 restickify a graph WEIGHT** (input address is a 16/32-GiB segment marker; output feeds a `batchmatmul` KERNEL/idx1) -- these are weight prelayout, not activation handoffs, and they are the ones the original recipe's "eliminate the in-graph restickify" framing was aimed at. Only **1** `ReStickifyOpHBM` (SDPA `[3]`) restickifies an **activation** (the K projection, low address, `mul[2]` output) -- and notably *that* one is same-stick on its producer edge (`mul[2] stick=x -> restickify stick=x`) but layout-changing on its consumer edge (the restickify itself flips `x -> out` for the QK^T matmul). So the activation restickify is the transpose; the weight restickifies are prelayout. This is the sharpest practical takeaway: **eliminating the in-graph weight ReStickifyOpHBM is a prelayout problem, not an STCDP problem.**

### The single best same-stick edge to target first for a real-model demo

**Target: the SDPA `[4]batchmatmul (QK^T) -> [6]sub` edge (equivalently `-> [5]max`), stick `['out']`, in `sdsc_fused__scaled_dot_product_fused_attention_overrideable_0_451ht_5h`.**

Why this one:
- **Same-stick, proven-safe primitive.** Producer output `stickDimOrder_=['out']` == consumer input `stickDimOrder_=['out']`. It is exactly the same-stick STCDP case proven value-correct and HBM-free on device; it does NOT touch the Compute-CB-faulting transpose.
- **Genuine cross-core ring, not a degenerate copy.** Producer (QK^T bmm) shards `{mb:32}`; consumer (softmax `max`/`sub`) shards `{x:32}`. Different ownership -> a real RIU-ring transfer, so the demo exercises `L3_LDU`/`L3_STU` (the thing the round-trip proof validated), not a dead-code-eliminated same-core copy.
- **Highest workload recurrence.** This is the attention-score -> softmax handoff. It occurs in **every attention layer of every transformer** in the roadmap (Llama/Mistral/Granite/GPT-OSS). The producer is a heavy matmul whose output is immediately re-read by the softmax, so the HBM round-trip is squarely on the critical path and recurs O(layers) times per forward.
- **Self-contained within one already-cached real attention bundle**, so a demo needs no new model plumbing.

Runner-up: Granite `[4]batchmatmul -> [5]mul` (the linear-layer output -> elementwise scale), same-stick `['out']`, also cross-core (bmm `{out:16,in:2}` -> mul `{mb:32}`). Good as a non-attention second data point (MLP path) for the same demo.

## Schema ambiguities / inferences flagged

1. **No `hbmStartAddress_` in the cache (load-bearing).** Edge tracing in the recipe's `splice_*.py` matches `hbmStartAddress_`, but that field is absent from every cached real-bundle SDSC (and from the recipe's own `add_mm_t` baseline cache). I traced edges via the `scheduleTree_` allocate node's `startAddressCoreCorelet_.data_["[0, 0, 0]"]` per-core HBM base instead. This is the correct resolved address (the splice's `hbmStartAddress_` is a post-dxp materialization), but the two schemas are different and I had to infer the equivalence. **Verified self-consistent**: every traced chain forms a clean, acyclic, op-sensible dataflow.

2. **Buffer reuse / address aliasing.** Low scratch addresses (especially `0` in SDPA, and `13377536`/`69214208` reused across SDSCs) are recycled by the allocator. I resolved this with a *latest-prior-producer* rule (the closest preceding SDSC that wrote the address is the true producer). I could not cross-check against an explicit liveness/SSA buffer-name map because the cache does not record one (every SDSC names its tensors `Tensor0/1/2` locally, with no global buffer identity). The dataflow it yields is internally consistent, but a buffer that is written, fully consumed, then independently reused at the same address would be indistinguishable from a real edge by address alone.

3. **`primaryDsInfo_` is per-ROLE, not per-tensor.** Stick/layout come from `primaryDsInfo_[dsType_]` where `dsType_` is INPUT/KERNEL/OUTPUT. When two inputs share a role (e.g. binary `add`/`mul` with both inputs `dsType_=OUTPUT`) they are reported with the same stick/layout. For these elementwise ops both operands genuinely share the layout, so this is correct in practice, but it is an inference: I cannot distinguish per-operand stick if a binary op ever had operands with different sticks (none observed here).

4. **"layout-changing" lumps two physically different costs.** The 8 needs-transpose edges include both heavy pre-matmul transposes (`activation -> bmm.in`, e.g. `mul[2]->bmm[4]` in Granite, `[4]id->[6]bmm` in Attn) and lightweight reduction-induced restickifies (`mean/rsqrt out<->x` in the RMSNorm tail, where the stick flips because the reduction changes which axis survives). Both flip `stickDimOrder_` so both need the transpose primitive, but the RMSNorm ones move far less data. I flag them together as needs-transpose (correct for "does it need the blocked primitive?") but they are not equal in value.

5. **`identity` SDSCs.** SDPA `[0]`/`[11]` and Attn `[2]`/`[4]` are `identity` ops (from `_unsafe_view`/`clone`/`expand`/`unsqueeze` view marshalling). They are real SDSC boundaries and I counted their edges, but several read or write a graph-segment marker (e.g. Attn `[1]bmm->[2]identity` reads an 80-GiB marker = a graph output buffer), which I bucketed as prelayout/marker rather than a true intra-block activation handoff.

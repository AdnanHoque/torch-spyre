# Cross-call weight preload investigation — project kickoff

## Hypothesis

The IBM AIU has a documented cross-call weight preload mechanism
(architecture doc slides 86-94) that should keep frequently-reused
weights in LX scratchpad across many inference calls. Empirically
this mechanism doesn't fire for `torch.compile`-driven matmul on
torch_spyre — every kernel call re-streams weights from DRAM (our
LX-budget probe showed first-iter == median across all configs,
ratio 1.00×).

If we can route `torch.compile` matmul through the preload path, we
get a **per-inference DRAM-elimination win for weights** that
compounds with everything else we've shipped (`output_element_priority`,
`DXP_LX_FRAC_AVAIL` in-call staging).

For production deployment with many tokens per inference, this could
be the biggest single lever remaining in the matmul performance
space.

## What the AIU stack already does (vs. what torch_spyre uses)

Code-reading findings from the deeptools and torch_spyre source:

### The mechanism IS fully supported in DSM and dxp

`/home/adnan/dt-inductor/deeptools/dsm/dsmds.h:939-940` — DSM tracks
two sets of dsengraph node indices:
```cpp
std::set<size_t> precomputeDsgNidx;  // preload node indices (dsengraph)
std::set<size_t> preloadDsgNidx;     // preload node indices (dsengraph)
```

`/home/adnan/dt-inductor/deeptools/dsm/graphOptimizer.cpp:1598-1676`
— DSM identifies tensors as static by checking the `_OUT_IS_STATIC=1`
attribute. Static tensors get STCDP preload nodes generated and
separated into the `loadmodel_to_spad` dsengraph.

`/home/adnan/dt-inductor/deeptools/dxp/dxp.cpp:354-358` — dxp creates
the `loadmodel_to_spad_dsg.txt` file, ready to receive preload nodes.

### Torch_spyre never marks anything as static

Searched `torch_spyre/_inductor/` and `torch_spyre/_inductor/codegen/`
for any of: `_OUT_IS_STATIC`, `_IS_PRECOMPUTE_OP`,
`_IS_ITER_DEPENDENT_NODE`, weight-registration APIs.

**Nothing.** The torch_spyre Inductor backend compiles matmul ops to
OpSpec, hands them to `dxp_standalone`, and never communicates which
inputs are weights vs. activations. dxp's `loadmodel_to_spad_dsg.txt`
is therefore created empty on every compile, and DSM has no static
tensors to preload.

### Why DXP_LX_FRAC_AVAIL helps but is the wrong mechanism

`torch_spyre/_inductor/scratchpad.py:45-48` shows
`DXP_LX_FRAC_AVAIL` reserves a fraction of the 2 MB LX scratchpad
for backend runtime use **within a single kernel call**. It doesn't
touch cross-call preload. This explains our LX-budget probe result:
20% wall-time win on L3-70B q_proj from in-call staging, but no
first-iter speedup because each call still independently fetches.

The architecture doc's `DSM_STATICDS_LXFRAC` (slide 86-87) is a
different env var that controls the static-LX partition for
preloaded tensors. It doesn't exist in torch_spyre's config.

## Architectural mismatch

The mechanism is **AOT** (ahead-of-time) — weights must be statically
identified before the inference loop begins, transferred once during
`loadmodel_to_spad`, and reused across many `execute` calls.

`torch.compile` is **JIT** (just-in-time) — it traces a graph and
generates code without prior knowledge of which inputs are "weights"
vs. transient activations.

Bridging these requires:

1. A torch_spyre API for the user to declare which tensors are
   weights at compile time
2. Codegen changes to emit `_OUT_IS_STATIC=1` on declared tensors
3. Verifying DSM picks up the annotations and generates preload nodes
4. Ensuring the runtime executes `loadmodel_to_spad` before the first
   inference iteration (and skips it on subsequent iterations within
   the same process)

## Investigation plan

### Phase 1 — Confirm code-reading findings empirically

Probe: compile a single matmul, capture the dxp output bundle,
inspect `loadmodel_to_spad_dsg.txt`. **Predicted: empty.** If empty,
confirms the agent's reading and validates the gap.

If non-empty, the agent's reading is wrong and we need to figure out
why preload isn't firing despite nodes being generated — different
failure mode, different fix.

### Phase 2 — Identify the smallest viable annotation hook

Read the torch_spyre codegen path from compile time to OpSpec
generation. Find where tensor metadata flows and identify the
narrowest place to inject an `_OUT_IS_STATIC=1` attribute on
designated tensors.

Two sub-questions:
1. Does Inductor's tracing-time graph already distinguish constant /
   parameter inputs from data inputs? (Probably yes — `torch.compile`
   knows about `nn.Parameter`.) If so, we may have ready-made
   metadata to use.
2. Is there an existing pass that could annotate inputs without
   changing the user-facing API? Or do we need a new opt-in API?

### Phase 3 — Hack a single-tensor preload to validate the path

Pick one matmul, manually inject `_OUT_IS_STATIC=1` on the weight
tensor's OpSpec, recompile, run, measure. **Predicted: first-iter
substantially slower than median, all subsequent iterations fast.**

If preload works for one tensor, the path is proven. Then it's a
question of plumbing — exposing the annotation through the right
level of the compile API and getting it to apply to all weights in
a model.

### Phase 4 — Production API + measurement

If Phase 3 works:
- Design the user-facing API (e.g., `torch.compile(weight_tensors=...)`
  or auto-detect from `nn.Parameter`)
- Ship with measurements showing per-token cost reduction in a
  realistic prefill workload

## Why this project is worth doing

Comparing to the other projects we've explored:

| project | layer | impact ceiling | self-contained from torch_spyre? |
|---|---|---|---|
| `output_element_priority` (shipped) | planner | medium-broad | ✓ |
| LX budget (mixed) | runtime config | 20% peak | mostly ✓ |
| Core-ordering (closed) | planner | dead | ✓ |
| K-split (closed, narrow) | planner | 13% on 1 shape | ✓ |
| Bidirectional ring (closed) | runtime/HW | unknown | ✗ (hidden) |
| Per-corelet (closed, multi-repo) | codegen+backend | 2× compute | ✗ |
| **Preload (this project)** | **codegen+runtime** | **per-call DRAM elim** | **mostly ✓** |

The preload investigation is the **biggest remaining lever where
torch_spyre is the natural place to make most of the changes**. DSM
and dxp already support the mechanism — they're waiting for
torch_spyre to feed them the right metadata.

It's bigger than per-corelet in a sense: per-corelet is 2×
compute *parallelism* but the per-call cost still includes weight
streaming. Preload eliminates the per-call weight streaming entirely
for repeated inference, which is the deployment pattern.

## What I'm doing first

Phase 1 — confirm the empirical baseline. I'll write a probe that
compiles a matmul, captures the dxp output bundle, inspects the
`loadmodel_to_spad_dsg.txt`, and reports whether it contains preload
nodes. ~30 min to write, ~5 min to run.

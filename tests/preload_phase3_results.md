# Phase 3 results — codegen wire works, DSM preload doesn't fire

## Headline

The five-touchpoint codegen change emits `isStatic_=1` correctly all the
way to the SDSC JSON for graph-input weights. **DSM still produces an
empty `loadmodel_to_spad_dsg.txt` and wall time is unchanged.** The
torch_spyre per-matmul bundle structure does not satisfy DSM's
preconditions for cross-call weight preload.

This is a clean negative result with high diagnostic value. The
project-scope assumption from Phase 2 ("zero DSC/DSM/dxp changes
required") was wrong. The real barrier is architectural, not codegen.

## What changed in the code

Spike branch: `AdnanHoque/preload-static-codegen-spike`. Five touchpoints:

| # | file | change |
|---|---|---|
| 1 | [`config.py`](../torch_spyre/_inductor/config.py) | new knob `preload_static` (env `SPYRE_PRELOAD_STATIC`, default off) |
| 2 | [`op_spec.py`](../torch_spyre/_inductor/op_spec.py) | new `TensorArg.is_static: bool = False` |
| 3 | [`spyre_kernel.py`](../torch_spyre/_inductor/spyre_kernel.py) | `_is_static_graph_input(name)` reads placeholder `tensor_meta.requires_grad` from `V.graph.orig_gm`; `create_tensor_arg` sets `is_static` |
| 4 | [`superdsc.py`](../torch_spyre/_inductor/codegen/superdsc.py) | `SDSCArgs.is_static` + copy from `TensorArg` in `_create_sdsc_tensors` |
| 5 | [`compute_ops.py`](../torch_spyre/_inductor/codegen/compute_ops.py) | emit `"isStatic_": int(tensor.is_static)` in `labeledDs_` |

**Plus a sixth touchpoint discovered during execution:** the OpSpec gets
serialized to Python source and re-executed downstream
([`spyre_kernel.py:692`](../torch_spyre/_inductor/spyre_kernel.py#L692)).
The serializer needed an `is_static={arg.is_static},` line to round-trip
the field. Phase 2 missed this — the four touchpoints were not
sufficient.

## What works

With `SPYRE_PRELOAD_STATIC=1`:

- The discriminator fires correctly. Instrumented logging:
  ```text
  _is_static_graph_input('arg0_1'): requires_grad=True  -> is_static=True
  _is_static_graph_input('arg1_1'): requires_grad=False -> is_static=False
  _is_static_graph_input('buf1'):  early-out (not a graph input)
  ```
- `is_static=True` flows from `TensorArg` → `SDSCArgs` → `labeledDs_`.
- The freshly-emitted SDSC JSON for the `ReStickifyOpHBM` op shows
  `isStatic_=1` on the weight input:
  ```json
  "labeledDs_": [
    { "dsName_": "Tensor0", "isStatic_": 1, ... },   // weight
    { "dsName_": "Tensor1", "isStatic_": 0, ... }    // restickified output
  ]
  ```

## What doesn't work

Two distinct gaps:

### Gap 1 — layout transform breaks the chain

The torch_spyre lowering pipeline doesn't hand the matmul kernel a
direct reference to the graph-input weight. It first runs a
**ReStickify** pass to convert the weight's tile layout, producing an
intermediate buffer (`buf0`):

```
arg0_1 (graph input weight)  --ReStickify-->  buf0  --matmul-->  buf1
```

The matmul OpSpec's input arg is `buf0`, not `arg0_1`. `buf0` is an
internal kernel buffer, not a graph input — so our heuristic
(`name in V.graph.graph_input_names`) returns False. **The matmul SDSC
shows `isStatic_=[0, 0, 0]` even though the weight upstream is flagged.**

DSM's static-ancestor pass at
[`graphOptimizer.cpp:1620`](file:///home/adnan/dt-inductor/deeptools/dsm/graphOptimizer.cpp)
*does* propagate static-ness across "DataOp/DimOp/Identity" parents. So
in principle, marking the ReStickify input as static *should* let DSM
infer that the matmul's input is also static. We confirmed below that
this propagation isn't happening — but the mechanism is at least
plausible if (Gap 2) didn't also block it.

### Gap 2 — DSM doesn't fire preload on single-matmul bundles

Even for the ReStickify input where `isStatic_=1` reaches the JSON,
`loadmodel_to_spad_dsg.txt` stays at the 18-byte empty skeleton.

Tracing the DSM side:

| step | location | requires |
|---|---|---|
| `pinStaticDsToLx()` populates `mi.preloadInsertNodes` | `dsmperf.cpp:2332` | `dscGlobal->doWeiPreload && lds.isStatic_ && lds.isHbmPinned()` |
| `insertStcdpForWeiPreload()` adds Stcdp nodes to dsg | `sen_data_ops.cpp:4291` | `mi.preloadInsertNodes` non-empty |
| `gi.preloadDsgNidx.insert(node)` | `sen_data_ops.cpp:4402` | per-Stcdp |

`doWeiPreload` defaults to true. `isHbmPinned()` returns true for our
weight (it has both HBM and LX in `memOrg_`). So the gate appears open.
But `mi.preloadInsertNodes` stays empty — meaning either (a) the static
DS doesn't get added to `staticDsAndFom`, (b) the per-form fitting
check fails, or (c) some upstream condition we haven't traced filters
it out.

Without a deeptools-side debug build, we can't tell exactly which.
What we CAN say from the empty output: torch_spyre's bundle structure
(one matmul wrapped in a `SenPreparedOp` node, sharing zero ops with
neighbors) doesn't expose the preload-eligible pattern that DSM was
designed to recognize.

## Wall-time A/B

Run via [`diag_preload_phase3_walltime_ab.py`](diag_preload_phase3_walltime_ab.py)
with the knob on and off, 1 warmup + 20 iters, fresh subprocess for each
mode (so env var takes effect at config-import time):

```text
mode        first     median       mean        min
OFF       4725.9μs    4585.9μs    4594.4μs    4571.5μs
ON        4702.7μs    4585.5μs    4592.1μs    4573.5μs

Δ first  (ON - OFF): -23.2 μs   (positive = preload setup cost)
Δ median (ON - OFF): -0.3 μs   (negative = preload paid off)
```

Statistically identical. No first-iter setup cost. No per-call savings.
Confirms the empty-preload-dsg observation: the runtime never executes
a separate `loadmodel_to_spad` phase because there's nothing in it to
execute.

## What this means for the project

The Phase 2 plan said: "zero DSC/DSM/dxp changes required." That was
based on tracing the parser/propagation/emission paths individually.
**Reality: the torch_spyre bundle structure also has to satisfy DSM's
upstream preconditions, and it doesn't.**

Three ways the project could move forward, in increasing order of effort:

### Option A — give up the per-bundle approach

If DSM only fires preload on multi-op bundles, torch_spyre would need
to either (a) emit multi-op bundles (large rework of `async_compile`),
or (b) accept that JIT-style per-op compilation forfeits this
optimization.

### Option B — hand-build the preload graph

Instead of trying to convince DSM to generate the preload graph, write
torch_spyre code that emits a non-empty `loadmodel_to_spad_dsg.txt`
directly alongside the dxp-generated bundle, with appropriate
`Stcdp` / `XrfPreload` nodes for each static weight. Requires
understanding the full dsengraph format and the runtime that consumes
it. Higher torch_spyre effort, no deeptools changes.

### Option C — enable a different runtime path

Maybe deeptools has a separate "model load" entry point that we're not
calling at all. The fact that the `loadmodel_to_spad_dsg.txt` *file*
exists (as an empty skeleton) suggests dxp generates it as a
placeholder — but whether the runtime ever reads it for torch_spyre's
single-bundle invocations is unclear. Investigating the runtime side
would clarify whether B is even worth attempting.

### Recommended

**Don't merge this spike.** The codegen change is technically correct
but operationally inert. Three follow-ups make sense:

1. Engage the deeptools team. Ask whether (a) torch_spyre's per-bundle
   structure is intentional / expected, (b) the preload mechanism was
   ever meant to fire for JIT-style clients, (c) what additional
   bundle-shape preconditions DSM needs to detect preload-eligible
   tensors.
2. If they confirm that single-bundle preload is in scope, dig into
   why `mi.preloadInsertNodes` stays empty even when `isStatic_=1` and
   `isHbmPinned()=1` are both satisfied. This requires either a debug
   build or detailed cooperation from the deeptools team.
3. If it turns out preload requires multi-op bundles, the project pivots
   from "wire up an existing mechanism" to "redesign the bundle
   boundary." That's a much larger conversation that probably needs an
   RFC and broader buy-in.

## Type of contribution this would have been

If the wire had fired end-to-end:

- **Feature addition**, opt-in via config knob — exposing dormant
  deeptools functionality to torch.compile users.
- Low correctness risk (off by default, easy revert).
- Open production-readiness questions: cache invalidation when weights
  mutate, multi-bundle LX coordination, user-facing API design.

As-is:

- **Diagnostic-only contribution.** The five touchpoints are correct
  and reusable; the negative result expands what we know about why
  preload doesn't fire today. Good RFC fodder, not a shippable patch.

## Files

- [`tests/diag_preload_phase3_verify_wire.py`](diag_preload_phase3_verify_wire.py)
  — single-compile verification (was the JSON populated? was the dsg
  populated?)
- [`tests/diag_preload_phase3_walltime_ab.py`](diag_preload_phase3_walltime_ab.py)
  — subprocess-isolated A/B comparison with first/median/mean/min
- five-touchpoint codegen patch (preserved on this branch)

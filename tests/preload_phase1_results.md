# Phase 1 results — `loadmodel_to_spad_dsg.txt` is empty

## Verdict

**Hypothesis validated.** A single `torch.compile`d matmul on torch_spyre
produces a dxp bundle in which `loadmodel_to_spad_dsg.txt` contains
zero preload nodes — only the 18-byte skeleton `I { } / O { } / T { }`.

This empirically confirms the code-reading finding from
[`preload_investigation_plan.md`](preload_investigation_plan.md): the
torch_spyre Inductor backend never marks any tensor as
`_OUT_IS_STATIC=1`, so DSM has nothing to preload, and the cross-call
weight preload mechanism never fires.

## What we ran

[`tests/diag_preload_phase1.py`](diag_preload_phase1.py): compile and
execute one matmul `(M=128, N=4096, K=4096)` fp16 on the spyre device,
snapshot the inductor-spyre bundle dir before and after, then inspect
the new bundle's five `*_dsg.txt` files.

## What we found

For the bundle `sdsc_fused_mm_0_ox96d6dy` produced by this run:

| dsengraph file | bytes | lines | N | E | A | populated? |
|---|---|---|---|---|---|---|
| `execute_dsg.txt` | 116 | 6 | 2 | 1 | 0 | **yes — the actual matmul kernel** |
| `loadprogram_to_device_dsg.txt` | 111 | 6 | 1 | 0 | 2 | **yes — binary upload** |
| `loadmodel_to_device_dsg.txt` | 18 | 3 | 0 | 0 | 0 | empty skeleton |
| **`loadmodel_to_spad_dsg.txt`** | **18** | **3** | **0** | **0** | **0** | **empty skeleton** |
| `loadprogram_to_spad_dsg.txt` | 18 | 3 | 0 | 0 | 0 | empty skeleton |

`execute_dsg.txt` carries `SenPreparedOp` + `SenProgAddrGenerate` for
the matmul. `loadprogram_to_device_dsg.txt` carries a `SenProgSend` for
the kernel binary. So two of the five graphs *do* get populated — the
empty state of `loadmodel_to_spad_dsg.txt` is not an artifact of dxp
producing skeletons everywhere, it is specific to the preload path.

## Why the empty skeleton is the smoking gun

Both populated graphs (`execute_dsg.txt`,
`loadprogram_to_device_dsg.txt`) show the *exact same* `I { } / O { } /
T { }` trailer that the empty graphs consist entirely of. So the
skeleton is the dxp-side literal "no nodes" representation. Three
zero-skeleton graphs in this bundle:

- `loadmodel_to_device_dsg.txt` — would carry static-tensor
  device-side allocations
- `loadmodel_to_spad_dsg.txt` — **the cross-call preload graph**
- `loadprogram_to_spad_dsg.txt` — would carry pre-staged program data

All three are infrastructure that exists to receive preload nodes if
torch_spyre ever marked a tensor as static. None of them ever do today.

## What this means

The architectural mismatch from the kickoff doc is confirmed at the
binary-bundle level:

1. DSM and dxp **wired up** the entire AOT preload pipeline:
   filenames, schema, generation logic, `_OUT_IS_STATIC` recognition.
2. torch_spyre **never reaches the wire** — it doesn't know which of
   its OpSpec inputs are weights vs. activations, so it doesn't
   annotate anything.
3. dxp dutifully creates an empty `loadmodel_to_spad` dsengraph on
   every compile. The runtime then has nothing to execute during
   "model load," so every inference call independently re-streams
   weights from DRAM through HMI.

This is consistent with the earlier LX-budget probe finding that
first-iter and median wall times match across all `DXP_LX_FRAC_AVAIL`
configurations: there's no warm-cache effect because there's no warm
cache. Every iteration starts cold.

## Generality check

Beyond the freshly-produced bundle, we sampled an existing fused_mm
bundle from the inductor cache (`sdsc_fused_mm_0_00u5yp_q`) and got
the identical 18-byte skeleton in `loadmodel_to_spad_dsg.txt`. The
cache holds 1,280+ compiled bundles from prior sessions. Spot-checking
suggests every matmul bundle has the same empty preload graph.

## Implications for impact ceiling

For LLM prefill matmul, the per-call HMI weight transfer is one of
the dominant wall-time components. If preload were wired through:

- **First inference**: pays the one-time `loadmodel_to_spad` cost
  (weights fetched DDR→LX once).
- **Subsequent inferences**: no DRAM weight fetch; just operand
  shuffling on the data ring + compute.

For a model that's reused across many prompts (the deployment case),
this approaches **per-token DRAM-elimination for weights**. The exact
saving depends on what fraction of measured wall time is weight
streaming vs. compute vs. launch floor — but with the largest LLM
matmul shapes already HMI-bound (see
[`element_priority_theory.md`](element_priority_theory.md)), the
ceiling is substantial.

## Next: Phase 2 — find the smallest viable annotation hook

Now we need to read the torch_spyre codegen path (Inductor lowering →
OpSpec → SDSC bundle) and identify the narrowest place to inject
`_OUT_IS_STATIC=1` on weight tensors. Two open sub-questions, restated
from the kickoff doc:

1. Does Inductor's tracing-time graph already distinguish constant /
   parameter inputs from data inputs? `nn.Parameter` should be visible
   at trace time, but we need to see whether that distinction survives
   into OpSpec.
2. Where in the OpSpec → SDSC → dxp path can an attribute travel?
   What's the existing schema for per-tensor attributes, and is
   `_OUT_IS_STATIC` already legal vocabulary on the torch_spyre side?

Phase 3 will then hack a single matmul's weight tensor to carry the
annotation manually and confirm the path lights up end-to-end.

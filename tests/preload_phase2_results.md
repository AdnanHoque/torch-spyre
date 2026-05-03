# Phase 2 results — codegen path mapped, wire format confirmed

Phase 2 was a code-reading + tracing exercise to answer two questions:

1. At the moment torch_spyre constructs an `OpSpec`, what metadata
   identifies a tensor as a weight (vs. an activation)?
2. What's the wire format DSM/DSC expects to enable cross-call
   preload, and what does torch_spyre need to emit?

Both are now answered. The path from "add an annotation in torch_spyre"
to "preload graph populated" is fully mapped end-to-end and consists
of four small touchpoints in torch_spyre with no DSC/DSM/dxp changes
required.

## What's already on the wire

DSC's SDSC-JSON parser at
[`designSpaceConfig.cpp:7427-7428`](file:///home/adnan/dt-inductor/deeptools/dsc/designSpaceConfig.cpp)
already accepts:

```json
"labeledDs_": [
  {
    "ldsIdx_": 0,
    "dsName_": "Tensor0",
    "...": "...",
    "isStatic_": 1     // <-- this field is recognized today
  }
]
```

That `isStatic_` value flows through the rest of the toolchain
unchanged:

| step | file:line | code |
|---|---|---|
| 1. DSC parses SDSC JSON | `designSpaceConfig.cpp:7427-7428` | `labeledDs.isStatic_ = (map3.second.int_value() != 0);` |
| 2. DSM reads it onto tensor property | `dsmperf.cpp:5914` | `prop.isStatic = dscLds->isStatic_;` |
| 3. DSM emits `_OUT_IS_STATIC` attribute on the dsg recv node | `graphOptimizer2.cpp:1597-1599` | `if (tprop.isStatic) { recvNode->insertAttribute("_OUT_IS_STATIC", unitVector); }` |
| 4. DSM static-ancestor pass uses it to identify preload-eligible ops | `graphOptimizer.cpp:1629-1671` | `if (parent->hasAttribute("_OUT_IS_STATIC") ...) ... tProp.isStatic = true;` |
| 5. DSM puts those ops in the preload dsg | `graphOptimizer.cpp:23752` | `gi.precomputeDsgNidx.insert(newNode->index());` |
| 6. dxp writes the dsg to disk | `dxp.cpp:354-358` | `loadmodel_to_spad_dsg.txt` |

So the deeptools side is fully ready to receive the signal. **Today
the empty `loadmodel_to_spad_dsg.txt` we measured in
[Phase 1](preload_phase1_results.md) is purely because torch_spyre never
emits `isStatic_` in its `labeledDs_`.**

## What torch_spyre would need to add — four touchpoints

Code-walking the codegen pipeline traced the path from input metadata
to SDSC JSON. The minimal change set:

### Touchpoint 1 — read parameter-ness at TensorArg creation

[`torch_spyre/_inductor/spyre_kernel.py:365-390`](../torch_spyre/_inductor/spyre_kernel.py#L365)
defines `SpyreKernel.create_tensor_arg`. At call time the kernel
already has the buffer `name`. We can ask `V.graph` whether `name`
maps to a `nn.Parameter` and stash the answer on the `TensorArg`.

The discriminator we found from a tracing probe is the
`tensor_meta.requires_grad` field on the placeholder:

```text
arg0_1 (weight):  tensor_meta.requires_grad = True   shape=(4096, 4096)
arg1_1 (input):   tensor_meta.requires_grad = False  shape=(128, 4096)
```

(Per [`diag_preload_phase2_explore.py`](diag_preload_phase2_explore.py).)
For inference graphs this cleanly separates parameters from
activations. Outputs (`buf*`) are not graph inputs at all, so they
never get flagged.

For a stricter signal — e.g., to also treat parameters with
`requires_grad=False` (frozen weights) as static — the same hook can
read `V.graph.named_parameters` directly, or cross-reference the
placeholder's `desc.idx` against the AOT module's parameter index set.

### Touchpoint 2 — add `is_static` to `TensorArg`

[`torch_spyre/_inductor/op_spec.py:24-43`](../torch_spyre/_inductor/op_spec.py#L24).
Add a single field with a default that preserves current behavior:

```python
@dataclasses.dataclass
class TensorArg:
    is_input: bool
    arg_index: int
    device_dtype: DataFormats
    device_size: list[int]
    device_coordinates: list[Expr]
    allocation: Any
    is_static: bool = False    # <-- new
```

### Touchpoint 3 — propagate through `SDSCArgs`

[`torch_spyre/_inductor/codegen/superdsc.py:41-71`](../torch_spyre/_inductor/codegen/superdsc.py#L41).
Add the same field to `SDSCArgs` and copy it across in
`_create_sdsc_tensors` ([superdsc.py:325-337](../torch_spyre/_inductor/codegen/superdsc.py#L325)):

```python
sdsc_args.append(SDSCArgs(
    layout=label,
    ...,
    is_static=arg.is_static,    # <-- new
))
```

### Touchpoint 4 — emit in SDSC JSON

[`torch_spyre/_inductor/codegen/compute_ops.py:373-394`](../torch_spyre/_inductor/codegen/compute_ops.py#L373) builds
the `labeledDs_` list. Add one key:

```python
"labeledDs_": [
    {
        "ldsIdx_": i,
        ...,
        "isStatic_": int(tensor.is_static),    # <-- new
    }
    for i, tensor in enumerate(sdsc_spec.args)
],
```

That's it. The rest of the path — DSC JSON parsing, DSM tensor
property propagation, dsengraph node generation, preload graph
construction, dxp file write — already exists and runs.

## What the discriminator misses, and what to do about it

`tensor_meta.requires_grad` is a useful default but not bulletproof:

- **Frozen weights** (`requires_grad=False` on a `Parameter`) would not
  be flagged. Models using `param.requires_grad_(False)` for
  fine-tuning / LoRA-style frozen layers would miss out.
- **Non-parameter inputs with grad** (e.g., user passes a
  `requires_grad=True` activation) would be falsely flagged. Unusual
  in inference but possible.

For Phase 3 (proof-of-concept), `requires_grad=True` is fine — it
lights up the typical inference matmul case. For a production API,
the more robust signal is to check whether the placeholder's
`desc.idx` is in the set of parameter input indices from AOTAutograd's
module signature. That set is preserved on the AOT-compiled
GraphModule even when individual placeholders have been "lifted" to
`PlainAOTInput`.

## Practical caveat — single-bundle bundles

The torch_spyre stack today produces **one matmul per dxp bundle**
(via `SpyreAsyncCompile.sdsc()` in
[`async_compile.py:41`](../torch_spyre/execution/async_compile.py#L41)).
Each call to `dxp_standalone --bundle -d <dir>` compiles one matmul.

Preload only pays off when `loadmodel_to_spad` runs **once** and many
subsequent `execute` calls reuse the LX-resident weight. So Phase 3
needs to verify two things, not one:

1. **Codegen wires up correctly**: setting `isStatic_=1` in the SDSC
   JSON for the weight tensor causes a non-empty
   `loadmodel_to_spad_dsg.txt` to appear.
2. **Runtime separates load from execute**: the runtime path (which
   `SpyreSDSCKernelRunner` triggers) actually executes the
   `loadmodel_to_spad_dsg` once and skips it on subsequent calls.

If (1) succeeds but (2) doesn't, we've confirmed the preload graph is
generated but the runtime still re-runs everything every call —
discovering this would push the project from "torch_spyre-only" to
"torch_spyre + runtime work."

## Why the granite reference fixture matters

The
[`granite_fp8-dump00001.pbdi`](file:///home/adnan/dt-inductor/deeptools/dsc/test/sengraph2/fp8/granite_33_8b_fp8/granite_fp8_75ff0b81-1283-4e11-8600-c8b06d57933b-dump00001.pbdi)
test fixture shows the format we'd expect to see emitted on a working
preload graph. `ModelInput` nodes (one per parameter) carry
`A _OUT_IS_STATIC 2 { 1 }`; `PrimaryInput` nodes (user activations) do
not. After Phase 3 lights up the wire, we should see this same shape
appear in `loadmodel_to_spad_dsg.txt`.

## Predicted Phase 3 outcome

If we add the four touchpoints above, recompile torch_spyre, and rerun
[`diag_preload_phase1.py`](diag_preload_phase1.py):

- `loadmodel_to_spad_dsg.txt` should grow from 18 bytes to a non-empty
  graph containing at least one `ModelInput`/`SenPreload`-style node
  per static tensor.
- `execute_dsg.txt` should *shrink* by the corresponding load-from-DDR
  ops that have moved to the preload graph.
- First-iter wall time should be slower than median (one-time preload
  cost), and median wall time should drop relative to today's.

If runtime separation works, the per-call DRAM-elimination win will
dominate any other lever we've explored. If it doesn't, Phase 3 still
produces a clean diagnostic: codegen-side fix is done; runtime work
becomes the next bottleneck.

## What to read for context

- [Phase 1 results](preload_phase1_results.md) — empty preload graph
  confirmed
- [Investigation kickoff](preload_investigation_plan.md) — original
  hypothesis & code-reading findings
- [`diag_preload_phase2_explore.py`](diag_preload_phase2_explore.py) —
  the exploration probe (instruments `create_tensor_arg`, dumps every
  signal V.graph offers about each tensor)

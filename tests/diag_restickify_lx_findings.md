# Ring-aware restickify: findings

## TL;DR

Under `LX_PLANNING=1`, torch-spyre's `ReStickifyOpHBM` forces a round trip
through HBM for the canonical matmul → transposed-consumer pattern at
`sencores>1`. The cross-core movement is **structural** (producer and
consumer partition the buffer along different host strides) so split
alignment cannot remove it — a ring-based on-chip shuffle (`STCDPOpLx`
/ `ReStickifyOpLx`) is genuinely the right mechanism. Path B
(`datadscs_`/STCDPOpLx) is **structurally required** — the cross-core
stick movement we need cannot be expressed in a compute-op DDL.
Path B isn't "write new codegen" — `runDcg` already exists at
[dxp.cpp:202](../../../../deeptools/dxp/dxp.cpp). It's wiring existing
data-op codegen through `--bundle`'s pre-passes (relax 2-4 guards
including [dxp.cpp:456-457](../../../../deeptools/dxp/dxp.cpp) and
[dxp.cpp:274](../../../../deeptools/dxp/dxp.cpp)).

Probe sources: [diag_restickify_lx_trace.py](diag_restickify_lx_trace.py),
[diag_capture_sc32_bundle.py](diag_capture_sc32_bundle.py).

## Goal

Stop restickification from forcing an HBM round trip when the data
could stay on-chip. The win matters because `LX_PLANNING=1` is expected
to be the going-forward default mode, which makes restickifies on
LX-resident data the *common* case, not an edge case.

## The one question, decomposed

> Can a producer → restickify → consumer chain be kept entirely on-chip —
> producer writes LX, the restickify does its relayout as a ring shuffle
> between cores' LX scratchpads, consumer reads LX — instead of the
> restickify spilling to HBM and back? And if so, whose change is it?

Three dependent sub-questions:

1. **Is there waste to recover?** (premise)
2. **Is the cross-core movement fundamental or incidental?**
3. **What's the emission mechanism?**

## Sub-question 1: Is there waste? — YES at sc=1

`tests/diag_restickify_lx_trace.py` compiles a few decode-shaped
fragments with `lx_planning=True`, `allow_all_ops_in_lx_planning=True`,
and reports each restickify's input/output buffer locations from the
final operations list (post-`scratchpad_planning`).

At `sencores=1`:

| case | restickify in | restickify out |
|---|---|---|
| `linear_x_Wt_decode` (`x @ W.t()`) | HBM (graph input) | HBM |
| `transposed_computed_intermediate` (`(a+b).t() + c`) | HBM (graph input) | **LX** |
| `matmul_then_transposed_add` (`(a@b) + c.t()`) | **LX** (matmul output) | **LX** |
| `chained_matmul_transposed` (`(a@b).t() + c`) | HBM (graph input) | **LX** |
| `qkv_attn_decode_gqa` | — (no restickify inserted) | — |

3 of 4 restickify edges touch LX-resident data. `matmul_then_transposed_add`
is the smoking gun: input AND output are LX-resident, yet the op
compiles to `ReStickifyOpHBM` — a flat 2×HBM round trip on data that
never needed to leave the chip.

## sc=32 reality: core-div-mismatch erases the LX residency

At `sencores=32`, the *same* probe shows the LX residency vanishing:

| case | sc=1 outcome | sc=32 outcome |
|---|---|---|
| `linear_x_Wt_decode` | HBM → HBM | HBM, restickify split `{4096:32}` vs matmul `{1:32}` — **cross-core** |
| `transposed_computed_intermediate` | HBM → LX | HBM, `{128:2,1:2}` vs pointwise `{128:32}` — **cross-core** |
| `matmul_then_transposed_add` | LX → LX | **HBM both ends**, `{128:32}` → `{128:2,1:2}` → `{1:32}` — **cross-core both** |
| `chained_matmul_transposed` | HBM → LX | HBM, `{256:8,1:4}` vs pointwise `{512:32}` — **cross-core** |

Every restickify at sc=32 is **core-div-mismatched** with its
producer/consumer. `work_distribution` gives the restickify a
different `op_it_space_splits` than its neighbors (often a tiny
`ncores=4` split while neighbors use 32). The scratchpad planner's
`core_div_mismatch` rule in
[scratchpad.py:370-374](../torch_spyre/_inductor/scratchpad.py) then
fires — verbatim comment *"buf users have diff core-splits → cross-core
LX read/write"* — and keeps the buffer in HBM:

```python
if using_multicore and len(users_rw) > 1:
    u0_split = users_rw[0].op_it_space_splits
    same_core_div = all(u0_split == u.op_it_space_splits for u in users_rw[1:])
core_div_mismatch[buf_name] = not same_core_div
```

That rule is a **workaround for a missing mechanism.** The HBM round
trip exists *specifically because there is no on-chip cross-core
shuffle wired up* — `STCDPOpLx` would be exactly that mechanism.

## Sub-question 2: Fundamental or incidental?

### The corrected model

A restickify lowers via [lower_restickify](../torch_spyre/_inductor/lowering.py)
to a `Pointwise` whose `inner_fn = lambda index: loader(index)` — read
index == write index. The relayout lives **entirely in the device
`SpyreTensorLayout`** (stick/stride map), not in the host index. And
the restickify's input and output `FixedTiledLayout`s share one host
layout (`finalize_layouts` does `_fixed_tiled(in_layout, restick_stl)` —
input host `size`/`stride` preserved).

Therefore a restickify induces **one** host-stride partition on both
its input and its output. Both edges are alignable iff producer and
consumer partition the buffer identically. By stride:

- **Both edges aligned** ⟺ `rs_part == prod_part == cons_part`
- **Achievable** ⟺ `prod_part == cons_part` (then work_distribution can
  give the restickify that partition)

So: **fundamental ⟺ `prod_part ≠ cons_part`**.

### What the probe found at sc=32

| restickify | producer part | consumer part | verdict |
|---|---|---|---|
| `linear_x_Wt_decode` | `{}` (graph input) | `{s4096:x32}` | HBM-LOAD |
| `transposed_computed_intermediate` | `{}` (graph input) | `{s1:x32}` | HBM-LOAD |
| `chained_matmul_transposed` | `{}` (graph input) | `{s1:x32}` | HBM-LOAD |
| `matmul_then_transposed_add` | **`{s128:x32}`** (matmul) | **`{s1:x32}`** (pointwise) | **FUNDAMENTAL** |
| `matmul_transposed_matmul` | **`{s256:x32}`** (matmul) | **`{s1:x32}`** (matmul) | **FUNDAMENTAL** |

Both restickifies that sit between two real on-chip ops are
**FUNDAMENTAL** — and the structure is identical every time. A matmul
partitions its output by the generated/N dim (large stride); a
transposed consumer wants the unit-stride dim. `{s_big:x32}` vs
`{s1:x32}` is "rows across cores" vs "columns across cores" — an
all-to-all transpose. No `work_distribution` split bridges that, and
forcing the matmul onto a non-natural split just pessimizes the matmul.

The flailing 2D splits the restickify currently gets (`{s1:x2,s128:x2}`
= 4 cores, `{s1:x4,s256:x4}` = 16 cores) are `work_distribution`
finding no good answer because there isn't one — confirmation, not
coincidence.

**Split alignment cannot fix this. The ring (`STCDPOpLx`) is genuinely
the mechanism.**

## Sub-question 3: Emission mechanism — empirical

torch-spyre currently emits restickify as a `dscs_` compute op with
`computeOp_[0].opFuncName = "ReStickifyOpHBM"` from the single
`RESTICKIFY_OP` constant in
[constants.py:17](../torch_spyre/_inductor/constants.py). torch-spyre
has never emitted a `datadscs_` entry. We tested both candidate paths
against `dxp_standalone --bundle` directly, using a real torch-spyre-generated
restickify SDSC (numCoresUsed_=32, split `{mb:32, out:1}`) as the
starting point.

### Path B — `datadscs_` data-op entry: forbidden at import

Took the deeptools reference fixture
[l3_dyn_loop_sdsc.json](../../../../deeptools/dcc/unittests/PCFG/l3_dyn_loop_sdsc.json)
(empty `dscs_`, populated `datadscs_` with an `STCDPOpLx`), wrapped it
in a `bundle.mlir`, ran `dxp_standalone --bundle -d <dir>`:

```
DtException: Datadsc not allowed, use dldsc, file dxp.cpp line 456
```

[dxp.cpp:451-460](../../../../deeptools/dxp/dxp.cpp) `Dxp::importSdsc`:

```cpp
std::unique_ptr<SuperDsc> Dxp::importSdsc(...) {
  ...
  DT_CHECK_MSG(mySdsc->dataOpdscs_.empty(), "Datadsc not allowed, use dldsc");
  DT_CHECK_MSG(!mySdsc->dscs_.empty(), "No dsc in sdsc input");
  return mySdsc;
}
```

`--bundle` mode hard-rejects any SDSC with `dataOpdscs_` non-empty,
*and* requires `dscs_` to be non-empty. The `runDcg` data-op branch at
[dxp.cpp:201-206](../../../../deeptools/dxp/dxp.cpp)
(`if (dscs_.size() == 0 && dataOpdscs_.size() > 0)`) is dead code under
`--bundle` because `importSdsc` crashes first.

#### Hack experiment evidence

Commented out the [dxp.cpp:456](../../../../deeptools/dxp/dxp.cpp)
`Datadsc not allowed` check, rebuilt `dxp_standalone` (39 s), re-ran
the fixture. Result: passed the import guard, then hit a new check at
[dxp.cpp:274](../../../../deeptools/dxp/dxp.cpp) in
`Dxp::createTrackers` — asserts `dscs_.size() == 1` for HBM tracker
setup. Reverted the patch.

This proves Path B's blockers are pre-pass **integration guards**, not
missing codegen. The pattern is clear: each guard is a small adjustment
(probably 2-4 total), and the codegen itself exists — `runDcg` at
[dxp.cpp:202](../../../../deeptools/dxp/dxp.cpp), plus sibling tools
`DataOpStandalone` and `dcg_standalone`.

### Path A — flip `dscs_` `opFuncName` to a Lx variant: silent no-op stub

Copied a real `numCoresUsed_=32` restickify SDSC, flipped
`computeOp_[0].opFuncName` from `"ReStickifyOpHBM"` to
`"ReStickifyOpLx"` and then to `"STCDPOpLx"`, ran `dxp_standalone --bundle`.

| variant | exit | init.txt | md5 | DDC verbose |
|---|---|---|---|---|
| HBM @ sc=32 (baseline) | 0 | 15,420 B (60 lines, real instructions) | `b822a34f` | `[DDC] DSC2 successfully filled` (203 ms) |
| HBM @ sc=1 | 0 | 14,135 B | `e0c66f7e` | (real) |
| ReStickifyOpLx @ sc=32 | 0 | **1,028 B** (4 lines, mostly `ffffffff`) | **`202a07bc`** | **`[DDC] DDL found but not suitable for op ReStickifyOpLx`** (6 ms) |
| ReStickifyOpLx @ sc=1 | 0 | 1,028 B | **`202a07bc`** | (same stub) |
| STCDPOpLx @ sc=32 | 0 | 1,028 B | **`202a07bc`** | (same stub) |
| STCDPOpLx @ sc=1 | 0 | 1,028 B | **`202a07bc`** | (same stub) |

The Lx variants produce **identical 1,028-byte output** across sc=1,
sc=32, and both op names — md5 `202a07bc` is the same in every Lx run.
That's a deterministic no-op fallback, not a real lowering. HBM scales
with core count; Lx doesn't. Content is mostly `ffffffff` padding —
boilerplate, no real instructions.

The decisive line is from DDC verbose
([ddc/ddl/ddl_conversion.cpp:91](../../../../deeptools/ddc/ddl/ddl_conversion.cpp)):

```
[DDC] DDL found but not suitable for op ReStickifyOpLx
```

The DDC compute-op lowerer in `--bundle` has a working DDL template
for `ReStickifyOpHBM` but **no working DDL template** for the Lx
variants. The strings parse (they're in
[dscdefn.cpp:462](../../../../deeptools/dsc/dscdefn.cpp)'s
`stringToOpFuncs` table), but lowering produces a stub.

All real `ReStickifyOpLx` lowering code in deeptools lives in `dsm/`
(the sengraph pipeline) — `lxopt.cpp`, `perfDscToSdsc.cpp`,
`senToPerfTester.cpp`, `constantOpt.cpp` — **never in `dcc/` or
`ddc/`**, which are what `--bundle` invokes via `runDcgForDlOpsStandalone`.

## Why not a Path A DDL template

A "just write a `restickify_lx.ddl` template" approach for the
compute-op path is a dead end for our actual goal (cross-core stick
movement). Two structural reasons:

1. **The HBM DDL is per-core compute only.** The existing template at
   [deeptools/ddc/ddl_templates/restickify.ddl](../../../../deeptools/ddc/ddl_templates/restickify.ddl)
   is 105 lines describing a per-core `lxlu → sfp → l0su` pipeline —
   load from local scratchpad, transform through the SFP, store to L0.
   There are no inter-core data transfers in that DDL. The HBM
   cross-core shuffle works today only because HBM is the shared
   shuffle medium; the DDL itself doesn't orchestrate cross-core
   movement. A hypothetical `restickify_lx.ddl` with the same shape
   would be a per-core *LX-local* relayout — and would not solve the
   fundamental case where the producer puts sticks on core A and the
   consumer wants them on core B.

2. **`STCDPOpLx` is structurally a data op.** Its fields —
   `prodConsList`, `idealStWindowToDtKey`, `coreIDtoTrRank`,
   `segCoreGroups` — orchestrate which core sends which stick to which
   other core. That cross-core orchestration is the entire point of
   the op, and it doesn't fit a compute-op DDL template (which assumes
   per-core compute over local memory). `STCDPOpLx` is fundamentally a
   `datadscs_` construct, and a Path A DDL would not provide the
   right framing for it.

## The deeptools ask

Wire up the existing data-op codegen path (`runDcg` /
`DataOpStandalone`) through `--bundle` by relaxing the pre-pass guards
that assume every SDSC has `dscs_.size() == 1` and no `dataOpdscs_`.
Concretely: the import guard at
[dxp.cpp:456-457](../../../../deeptools/dxp/dxp.cpp) and the tracker
setup at [dxp.cpp:274](../../../../deeptools/dxp/dxp.cpp), plus likely
1-2 more sibling guards uncovered by stepping through. The codegen
itself already exists — `runDcg` at
[dxp.cpp:202](../../../../deeptools/dxp/dxp.cpp) plus
`DataOpStandalone` and `dcg_standalone` — so this is an integration
job, not new codegen.

## What's already in place on the torch-spyre side

**This is no longer "write new codegen on either side."**

- **Deeptools side: integration, not new codegen.** The `runDcg`
  data-op path and its sibling standalone drivers (`DataOpStandalone`,
  `dcg_standalone`) already exist; the work is relaxing the 2-4
  pre-pass guards in `dxp.cpp` that block `--bundle` from reaching
  them.
- **Torch-spyre side: small change.** Gating logic (cross-core +
  on-chip producer/consumer) plus a new codegen path for `datadscs_`
  entries, since today we only emit `dscs_`. The new path is
  structurally similar to the existing `generate_sdsc` in
  [compute_ops.py](../torch_spyre/_inductor/codegen/compute_ops.py).
- The compile-time signal for "this restickify is core-div-mismatched"
  is already computable from `op.op_it_space_splits` and its neighbors
  ([scratchpad.py:365-374](../torch_spyre/_inductor/scratchpad.py)).
- All probe infrastructure exists:
  [diag_restickify_lx_trace.py](diag_restickify_lx_trace.py) hooks
  after `scratchpad_planning` to extract per-restickify partitions and
  classify FUNDAMENTAL/INCIDENTAL/HBM-LOAD;
  [diag_capture_sc32_bundle.py](diag_capture_sc32_bundle.py) captures
  fresh multi-core SDSC bundle dirs.

Once deeptools support is in place, torch-spyre work is roughly:

1. Decide the gating condition (cross-core *and* producer/consumer are
   on-chip ops, not graph inputs).
2. Add a `datadscs_` codegen path that emits an `STCDPOpLx` entry when
   the gating condition fires (instead of the `dscs_` /
   `ReStickifyOpHBM` entry we emit today).
3. Relax `core_div_mismatch` in `scratchpad.py:158-163` to keep
   ring-restickified buffers in LX instead of HBM.

## Reproducing

```bash
# Premise + structural probe (sc=1 and sc=32, all FUNDAMENTAL/INCIDENTAL/HBM-LOAD verdicts)
python3 tests/diag_restickify_lx_trace.py

# Capture a fresh multi-core restickify SDSC for the bundle experiments
python3 tests/diag_capture_sc32_bundle.py

# The bundle experiments (find a real sc=32 restickify SDSC, flip opfunc, compare)
SRC=$(find /tmp/torchinductor_adnan/inductor-spyre -name "sdsc_*ReStickify*.json" -printf "%T@ %p\n" \
       | sort -rn | head -1 | awk '{print $2}' | xargs dirname)
rm -rf /tmp/bundle_test && mkdir /tmp/bundle_test
cp "$SRC"/sdsc_*ReStickify*.json /tmp/bundle_test/
echo 'module { func.func @sdsc_bundle() { sdscbundle.sdsc_execute () {sdsc_filename="<file>.json"} return } }' \
  > /tmp/bundle_test/bundle.mlir   # set <file> to the copied filename

# Flip the opFuncName (and the outer key) with the python snippet, then:
DXP_VERBOSE=1 dxp_standalone --bundle -d /tmp/bundle_test 2>&1 | grep -iE "DDL|DDC|millisec"
```

## Methodology notes

- **Verify compile success isn't a silent fallback.** `exit=0` plus
  smaller output is *not* proof of real lowering. DDC will accept any
  opfunc name in `stringToOpFuncs` and silently emit a stub if no DDL
  template matches. Always check `DXP_VERBOSE=1` output for `"DDL
  found but not suitable"` or comparable messages, and check `init.txt`
  byte counts AND md5s across runs that *should* differ (e.g. sc=1 vs
  sc=32) — identical md5s across genuinely different inputs are the
  tell for stubbed codegen.
- **The probe's `core_div_mismatch` reproduces the planner's check
  exactly.** Direct equality on `op_it_space_splits` tuples — same as
  [scratchpad.py:370-374](). Don't be tempted by host-coefficient
  per-variable analyses for restickify: read and write index are the
  same expression (`lower_restickify` uses `loader(index)`), so
  read-coeff vs write-coeff is uninformative. The relayout is in the
  STL.
- **The partition-by-stride comparison is layout-independent.** A
  buffer's host strides are layout-invariant, so `{stride: split}`
  dicts are directly comparable across the producer, restickify, and
  consumer that all touch the same buffer — even though each op uses
  its own iteration symbols.

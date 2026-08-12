# PR 3439 — reviewer pass 6

- **Head:** `79eab71e` · **Base:** `edf70f41` · 22 files, +1431 / −246
- **Previous pass:** head `495b0448`, base `2c832809`
- **Verdict:** approve with one blocker. Fix the stale `v1` test literal; everything else is small.

---

## What actually changed

Almost none of the compiler code moved. `lx_relayout.py`, `graph_editor.py`, `superdsc.py`,
`spyre_kernel.py`, `op_spec.py`, `pass_utils.py`, `views.py`, `ir.py` are **byte-identical**
to the last reviewed state (verified by hashing the hunk bodies with line numbers stripped).

The author's real edits since `495b0448`:

| File | Change |
|---|---|
| `patches.py` | new pre-scheduling idempotence guard |
| `config.py` | +1 comment line |
| `kernel_provenance.py` + `.cpp` + its test | fingerprint bumped v1 → v2, two new fields |
| `docs/…/torch_spyre.rst` | documents `SPYRE_LX_PLANNER_RELAYOUT` |
| `docs/…/scratchpad_planning.md` | the four falsified claims corrected |
| `scheduler.py`, `allocator.py` | comment-only rewrites (net-zero, which is why numstat didn't move) |

**All seven items from pass 5 are addressed.** The remaining growth is the rebase, which
pulled 7 upstream commits — including one that landed in `patches.py`, the same function
this PR modifies.

---

## Blocker

### 1. The v1 → v2 bump misses `tests/test_prepare_kernel.py` — two tests fail

`tests/test_prepare_kernel.py:46` still builds `spyre_kernel_v1_registry_test_{key}`.
The C++ parser hard-rejects any prefix that isn't exactly `spyre_kernel_v2_`
(`kernel_provenance_registry.cpp:84-87` → `nullopt`), and `registerKernelProvenance`
turns that into `false` (`:116-123`). Two tests assert registration *succeeds*:

- `test_kernel_provenance_registry_insert_and_duplicate` — line 53
- `test_kernel_provenance_registry_rejects_conflict_without_overwrite` — line 69

Both fail deterministically. No device needed; they're module-level and don't use the
`initialize_runtime` fixture. This file arrived with upstream #3354 **in this same rebase**,
which is exactly why it was easy to miss — the sibling test file `tests/inductor/
test_kernel_provenance.py` *was* updated, and updated well (it even adds a negative case
at `:539` pinning that v1 names no longer parse).

Two sites are stale but harmless: `test_prepare_kernel.py:224` (`prepare_kernel.cpp` never
parses the prefix, it concatenates the profiler name verbatim) and
`test_kernel_provenance_registry_miss_and_unparseable_name` (uses the literal `"not_an_event"`).

**Fix** — derive it, the way the sibling file already does:

```python
from torch_spyre._inductor.kernel_provenance import KERNEL_PROVENANCE_KEY_VERSION

def _registry_event_name(key):
    return f"spyre_kernel_v{KERNEL_PROVENANCE_KEY_VERSION}_registry_test_{key}"
```

That makes the site bump-proof. Same for line 224.

---

## Should fix before merge

### 2. `docs/…/profiling/pytorch_profiler.md:81` still advertises the v1 event name

Documents `spyre_kernel_v1_<summary>_<key>#<step>`. The runtime now emits `v2_`. A user
greps a captured trace for the documented string and matches zero events — the join
silently returns nothing rather than erroring. This page also came in with #3354, so the
bump is what invalidated it.

`adding_operations.md:84` already tells contributors to bump the version on payload
changes; adding this page to that checklist would close the loop.

### 3. `scratchpad_planning.md:65` contradicts the PR's own rewrite in the same file

The PR correctly rewrote the limitation section (line ~609) to say planning now
materializes identity copies and SuperDSC lowers them to `shuffle`. But the hardware
parameter table nine hundred lines earlier still reads:

```
| Inter-core data ring | yes | not yet used by compiler |
```

That row is false at this head whenever the flag is on — which is the default. The table
is the document's quick-reference, so it's where a reader checks capability. Update the
row; leave line 66 (reduce-sum ring) alone, that one is still true.

### 4. `TensorWorkDivision` is destructured into the fingerprint but not schema-guarded

`_canonical_tensor_arg` (`kernel_provenance.py:270-283`) hand-picks exactly two fields out
of `TensorWorkDivision`, but `_validate_finalized_schema` (`:198-202`) guards only
`OpSpec`, `TensorArg`, `LoopSpec`.

The gap is one level down, and it's silent by construction: add a third field to
`TensorWorkDivision` and it's simply omitted from the hash, while `TensorArg`'s annotation
string stays `"TensorWorkDivision | None"` — so no drift guard can fire.

Concretely: a later `slice_order` field expressing a non-default core→slice permutation
would make two genuinely different relayout plans hash to the same 16-char key. The second
`registerKernelProvenance` hits the conflict branch (`:127-140`), increments `conflicts`,
returns false without storing — and every profiler lookup for that key returns the *first*
kernel's debug handles. Trace-to-source attribution points at the wrong kernel and nothing
fails.

Worth noting the file already documents `DebugHandle`'s exclusion with a reason
(`:195-197`). `TensorWorkDivision` has no such note, so this reads as an omission rather
than a decision.

**Fix** — add `_EXPECTED_TENSOR_WORK_DIVISION_SCHEMA` and include the pair in the `schemas`
tuple, then extend the parametrization at `test_kernel_provenance.py:358`.

---

## Resolved — closing the default-on question

I flagged the default-`True` flag in two prior passes. It is **not a defect**, and I'm
closing it. The lineage, by author date (all three share a committer date, so committer
date orders nothing):

| | |
|---|---|
| `9316092a` Aug 5 23:17 | "Keep LX relayout opt-in" → `False` |
| `5412deb0` Aug 6 00:33 | "Enable LX relayout by default" → `True` |
| `7eb1b134` Aug 6 01:47 | "Keep LX relayout opt-in **until DeepTools support lands**" → `False` |
| `41c6493a` Aug 10 | **"Use identity copies for LX relayouts"** → `True` |

The stated blocker in `7eb1b134` was DeepTools support. `41c6493a` is the commit that
switched the mechanism to plain identity copies — which is precisely what removes the
DeepTools dependency, since the backend now sees an ordinary copy lowered to `shuffle`
rather than a novel construct it must be taught. Re-enabling the default *in that same
commit* is coherent and deliberate.

Our own device runs corroborate it: token 203, `STCDPOpLx` present, zero `STCDPOpHBM` /
`ReStickifyOpHBM` / `DmaOp`, on the shipped DeepTools. The mechanism works on real hardware.

No action. My earlier flag was based on `7eb1b134` not being an ancestor without checking
what superseded it.

---

## Not this PR — but file it

### `patches.py`: five global mutations sit outside the `try` that restores them

`enable_spyre_context` mutates five process-global things *before* the `with` header at
`:152-158`:

```
 99  Loops.has_large_inner_fn = lambda self, threshold=None: True
105  joint_graph.pass_patterns.pop()
122  GraphLowering._update_scheduler = _spyre_update_scheduler
143  SchedulerNode.has_side_effects = _spyre_scheduler_node_has_side_effects
150  _saved_copy_noop = noop_registry.pop(torch.ops.aten.copy.default, None)   ← upstream #3681
```

Every restore lives in the `finally` at `:161-167`, which is inside the suite of a
five-manager `with`. A parenthesized multi-manager `with` is nested-`with` sugar: the suite
runs only after all five `__enter__`s succeed. If any raises, the already-entered managers
unwind but line 161 is never reached, and all five mutations leak for the life of the process.

The reachable trigger is `enable_spyre_lowerings()` at `:154` — its `first_enter` block
calls `make_fallback(overload, override_decomp=True)` (`lowering.py:159`), whose own
docstring says it exists to pre-empt an upstream "both a fallback and a decomp for same op"
assertion. Any drift between the repo's `fallback_ops` and torch's decomposition table
raises there. That manager compounds it: `_lowerings_nesting += 1` (`lowering.py:181`) sits
before its own `try`, so a raise pins the counter above zero and the lowering-registry
restores are dead for the rest of the process too.

The worst leak is line 122, not the `noop_registry` pop: a leaked `_update_scheduler` makes
**every later compile in the process** — plain CPU graphs with no Spyre tensor anywhere —
run scratchpad allocation, core division, stickification and LX relayout materialization
against a graph none of those passes were written for. `joint_graph.pass_patterns` degrades
cumulatively, since the next entry snapshots the already-shortened list at `:103` and pops again.

The sole caller (`_inductor/__init__.py:156`) records FFDC and re-raises with no reset.

**This is pre-existing structure**, from #664 / #1356 / #1659, and the PR's own guard is
correctly placed — it's a closure body that runs during `codegen()`, inside the `try`, so an
exception from `_pre_scheduling_pass` still reaches the `finally`. The rebase made it one
notch worse only by adding a fifth thing to the unprotected region. Separate issue; not a
merge condition here.

---

## Checked and clean

Things I expected to be problems and aren't. Recording them so nobody re-audits.

**The new idempotence guard is correct, and the right scope.** `CustomPreSchedulingPasses`
holds ~21 passes (`passes.py:425-465`); none needs to re-run, because every one mutates
`graph.operations` and layouts in place and nothing between two calls clears them. The one
non-mutating item, `reset_provenance_warnings()`, is re-run on every Scheduler build anyway
(`passes.py:190`). Running the pipeline *twice* is the bug — a second `insert_bmm_padding`
double-pads — so suppressing the whole pipeline is right, not over-broad.

Double-wrap trace: outer sets flag → runs pipeline once → calls inner wrapper → inner sees
the flag and skips → calls the original. Pre-scheduling runs once, the real `Scheduler`
build runs once, teardown is LIFO-correct. And double-wrap is genuinely reachable —
`_inductor/__init__.py:73/167` rebinds the `cfx.compile_fx` global and
`compile_fx.py:2527` recurses through it whenever `config_patches` is truthy.

The instance-scoped flag is safe: upstream calls `_update_scheduler` twice on one
`GraphLowering` only in the cpp-wrapper two-pass path, gated at `graph.py:2313` on
`cuda`/`xpu`, which a Spyre graph never enters. `codegen_subgraph` uses the subgraph's own
`GraphLowering`.

**Fingerprint determinism is sound.** `work_slices` / `core_id_to_work_slice` are plain
`dict[Symbol, …]` at every construction site. `_canonical_value` (`:296-303`) canonicalizes
keys to `{"sympy": srepr(...)}` first, then sorts by the JSON of the canonicalized key —
which sidesteps sympy returning a `Relational` from `<`. Same fingerprint across runs and
across `PYTHONHASHSEED`. The new fields are also genuinely populated before hashing
(`superdsc.py:953` mutates `work_division` during `sdsc()`, which is upstream of
`build_kernel_provenance_descriptor`), and fresh vs. FX-cache-replay hash the same state.

**All four rebase interactions are benign:**

- **#3681 (`noop_registry` drops `aten.copy`).** Two independent reasons it can't bite. The
  PR materializes at Inductor IR level (`graph_editor.py:170-197` registers a new
  `ComputedBuffer`); the FX node it also creates is metadata written *after*
  `post_grad_passes` already ran, and nothing re-runs `remove_noop_ops`. And it's a
  different op — the PR inserts `aten.clone` (`graph_editor.py:45-51`), #3681 pops
  `aten.copy.default`. The allocator's "post-grad no-op elimination has already run"
  comment is still true and still load-bearing. Worst case if a preserved `copy_` sits
  downstream is two value-identical copies — wasted LX residency, not a wrong tensor.
- **#3673 (WSR partial-reduction detection).** Two different meanings of "partial". Upstream's
  is a hard `raise Unsupported` about *coarse-tile loop* reduction tiling, and it runs at
  `passes.py:439`, before LX planning at `:464` — a flagged graph never reaches the planner.
  The PR's `partial` is a *core-division* K-split (`pass_utils.py:1588`), rejected at both
  ends (`lx_relayout.py:222-223`, `:258-264`). The upstream change also moved toward *more*
  rejection, so the set of graphs reaching LX planning shrank.
- **#3622 (preserve dim order and padding when tiling).** There is no cross-buffer footprint
  comparison to corrupt — both sides are evaluated against the *same* buffer name and read
  one `device_layout`. The clone copies size, stride, `device_layout` and offset verbatim
  (`graph_editor.py:151-158`). And size arithmetic uses `device_size` stick counts, not host
  strides, so host padding never reaches it.
- **#3354 (provenance).** Integrated correctly — new fields added to both the schema guard
  and the canonicalizer, version bumped on both sides. Only the two stale literals above.

**The layout-unwrap concern I carried from pass 5 is disproved, not merely unproven.**
`scheduler.py:520` uses `get_layout()` without the `MutationLayoutSHOULDREMOVE →
real_layout()` unwrap that `pass_utils.py:60-84` performs. The asymmetry is real but
unreachable: relayout sources are gated on `isinstance(producer.layout, FixedTiledLayout)`
at plan time (`lx_relayout.py:223-228`) and a mutation layout is not one; copies get a
freshly built `FixedTiledLayout` (`graph_editor.py:154-163`); `lx_names` is filled only from
`_lx_resident`, which requires `layout.allocation`, defined only in
`FixedTiledLayout.__init__`. And even if one leaked through there'd be nothing to clear —
the allocator refuses LX to both sides of every mutation (`allocator.py:282-283`, `:382-383`).
Timing closes it: every pass creating a mutation layout runs strictly before
`_maybe_scratchpad_planning`, the last pass in the pipeline.

Adding the unwrap would still cost nothing and make the guard self-sufficient rather than
resting on five non-local facts — but there is no bug here today.

---

## Claims raised and refuted — do not act on these

- **"The guard comment names only LX relayout, but other passes are non-idempotent too, and
  narrowing it would double-pad matmuls."** The double-pad arithmetic is impossible: the pad
  target is absolute, not incremental (`padding.py:251` `k_padded = k_val + pad`, `:280`
  assigns `y_padded_size[y_k_dim] = k_padded`). Re-running is idempotent by construction.
- **"The C++ comment names a test that doesn't actually pin the literal."** `profiler_event.py:48-51`
  derives both `_EVENT_NAME_PREFIX` and `_SUPPORTED_KEY_WIDTHS` from
  `KERNEL_PROVENANCE_KEY_VERSION`, and the parametrized contract test does pin the pair.

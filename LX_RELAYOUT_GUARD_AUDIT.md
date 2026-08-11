# LX relayout — guard audit

Written in response to a reviewer question: is there a lot of defensive code and guarding in
this PR that is overkill?

Short answer: partly, and about 95 of the 1011 added source lines can go — but roughly half
the guard-shaped lines are load-bearing and should be defended without apology. The deletable
mass is not spread across the eligibility filters and atomicity checks. It is two clusters
and one tautology.

The largest cluster is not defensive code at all. Some 44 lines implement a **re-entrant
planner** — support for calling `plan_allocation` a second time on an already-materialized
graph. No such call exists: `materialize_lx_relayouts` runs as the last statement of
`_push_allocation`, and the only double-plan path is the `SolveError` retry, which fires
strictly before `_push_allocation`. The visible consequence is that a reset loop writes
`None` over `None` for every op in the graph on every compile, including the two allocator
paths that never relayout at all. The guards read as over-caution; they are really downstream
of a mode that was never built.

Two findings run the other way, and matter more than any deletion: a `.get(device_dim, 0)`
default that would silently claim *every core owns slot 0* and emit that straight into the
SDSC ownership map, and a matmul-operand test that infers operand identity from a dep's
position in an `OrderedSet`.

Method note: every classification here was checked in both directions — an adversary was
asked to find the reaching path for anything called unreachable, and the upstream invariant
for anything called load-bearing. Earlier passes on this branch defended guards somewhat
reflexively on the grounds that this machinery fails silently; that reasoning is right for
guards catching reachable corruption and wrong for guards on states that cannot occur, which
actively mislead the next reader about which invariant is doing the work.

---

# Guard audit: `ah/lx-relayout-from-current-main-b4ae` @ `cc90711b`

*Paths below are relative to `/Users/adnan/torch-spyre-work/torch-spyre/`.*

## Is it overkill?

Partly â but not in the way the census suggests. About **95 of the 1011 added lines can go (~9%)**, and roughly **half of the ~110 guard-shaped lines the colleague counted are load-bearing and should be defended without apology**. The deletable mass is not spread thinly across the eligibility filters and atomicity checks; it is concentrated in **two clusters and one tautology**. Cluster one (~44 lines) is not defensive code at all â it is a dead *feature*: a re-entrant planner path that supports calling `plan_allocation` twice on an already-materialized graph. That call does not exist, because `materialize_lx_relayouts` is the last statement of `_push_allocation` (`allocator.py:1033`) and the only double-plan path is the `SolveError` retry (`allocator.py:2441-2451`), which fires strictly *before* `_push_allocation` ever runs. Cluster two (~8 lines) guards a `PerCoreView` being geometrically malformed, which its single constructor makes impossible. The tautology is `supports_lx_relayout` (~6 lines), which is read only on the one path where it is always `True`. Everything else â the try/except probe, the type filters, the atomicity gate, the post-fusion demotion â earns its keep, and two of them are the only thing standing between this branch and the silent-wrong-numbers failure mode. **Worth about half a day** for the two clusters (that is 80% of the value, and both are mechanical deletions); the remaining ~25 lines are opportunistic and can ride along with whatever touches those files next.

---

## Delete these

### 1. The replay cluster â ~44 lines, spanning two files

| where | what |
|---|---|
| `lx_relayout.py:222-244` | registry-replay early return (23) |
| `lx_relayout.py:229-231` | nested `source is None or copy is None: existing.clear()` (inside the above) |
| `lx_relayout.py:345-348` | `lx_view` reset loop over every op in the graph (4) |
| `lx_relayout.py:351` | `copies.clear()` inside `if not plans:` (1) â keep the bare `return` |
| `lx_relayout.py:59`, `:65-67`, `:359-361` | the `destination_buffer_name` field, the `or` arm of `destination_name`, and the branch that consumes it (7) |
| `allocator.py:227-229` | the `allocation.pop("lx", None)` loop (3) |
| `allocator.py:874-875` | `elif plan.destination_name in by_name: continue` (2) |
| `allocator.py:953-956` | `discard_lx_relayout_group` + `clear_lx_relayout_state` block (4) |

**Why none of it can fire.** `collect_lx_relayout_plans` has exactly one caller: `ScratchpadAllocator._prepare_buffers` (`allocator.py:222-226`). The registry is written in exactly one place, `materialize_lx_relayouts`, called at `allocator.py:1033` â the *last* statement of `_push_allocation`. `plan_allocation` runs twice on one graph only through the greedy retry at `allocator.py:2441-2451`, gated on `SolveError`, which is raised only at `ilp_solver_ortools.py:473` and `:492` inside `plan_layout` â i.e. inside `_solve`, strictly before `_push_allocation`. So on the retry the registry is provably unset. `CoOptimizingAllocator` overrides `_prepare_buffers` (`allocator.py:1712`) with no `super()`; `StrategyBCoOptimizingAllocator` overrides `plan_allocation` (`:1487`) and calls `_generate_buffers` directly (`:1545`) â neither reaches `collect` at all. `_clear_lx_relayout_groups` is reachable only from `allocator.py:882` and `:252`, both pre-push, so `discard_lx_relayout_group` always returns the empty set and `:955-956` never executes. Corollary: nothing writes `lx_view` before materialize (the only writers are the replay itself, `lx_relayout.py:369-370`, and `ir.py:105` which initialises it to `None`), so the reset loop at `:345-348` writes `None` over `None` for every op in the graph, on every compile â including the StrategyB and CoOpt paths, which never relayout at all. No test reaches any of it: the only two that set `_spyre_lx_relayout_copies` are `tests/inductor/test_work_division_hint.py:739` (empty dict) and `:806` (a scheduler-demote test that never calls collect).

**Put in its place:** one comment on `collect_lx_relayout_plans` â *"Called once per compile, on a graph that has never been materialized. `_push_allocation` materializes as its last act, so no solver retry can observe a populated registry."* If you would rather keep the cluster as insurance for a future re-plan path, say that in the comment; what is wrong today is that it reads as live safety machinery.

*One residual caveat, stated honestly:* this rests on `GraphLowering._update_scheduler` being called once per `GraphLowering`. I could not read torch's copy to confirm. But if it were called twice, `CustomPreSchedulingPasses.__call__` (`passes.py:467-480`) would re-run deadcode elimination, stickification, restickify insertion and work distribution on an already-lowered graph â catastrophic far beyond relayout. The assumption is systemic, not local to this file.

### 2. The `PerCoreView` geometry cluster â ~8 lines

- `lx_relayout.py:134-135` â `if value.free_symbols: return None`
- `lx_relayout.py:136-137` â `if slot < 0 or slot >= split: return None`
- `lx_relayout.py:153-154` â `if source_map is None or destination_map is None: return False`
- `lx_relayout.py:266` â the `_core_slices(...) is None` clause (the other two clauses in that condition stay)

**Why they cannot fire.** Every non-empty `PerCoreView` in the system is built by `_per_core_view_from_prep` (`pass_utils.py:1689`); the only other producers are empty-view short circuits (`:1558`, `:1562`, `:1744`, `:1772`, `:1779`). There, `work_slice_dims` and `core_to_slot` are populated in the *same loop from the same* `sym_to_device_dim` map (`:1659-1660`, `:1683-1686`), and `core_to_slice_mapping` emits an entry for every dim handed to it (`core_mapping.py:58-67`), so the two tuples always share a key set. Every slot expression is `Integer(0)`, `Mod(core_id, s)` or `Mod(floor(core_id/t), s)` (`core_mapping.py:59-66`), so after `subs(core_id, k)` at `lx_relayout.py:133` it is always a concrete integer â `core_id` is the sole free symbol, and `Mod` with a positive modulus lands in `[0, s)`. `splits_by_stride` only admits syms with `split > 1` (`pass_utils.py:1581`), so a work-slice dim never even gets the `Integer(0)` arm.

**Put in its place:** drop the `Optional` from `_core_slices`' return type â that alone removes both call-site checks â and one comment stating the invariant: *"Slot expressions are `Mod` by the split they are paired with, so every substitution yields an integer in range."* No test touches `_core_slices` or `_compatible_partitions` at all today, which is exactly why these rot.

### 3. `supports_lx_relayout` â ~6 lines

`allocator.py:135`, `:1485`, `:1680` (declarations), `:224` (the ternary), `:1552` (`assert not self._lx_relayout_plans`).

Read at exactly one place, `allocator.py:224`, inside the base `_prepare_buffers` â which neither co-opt allocator reaches (see cluster 1). So the `False` branch is dead and the assert asserts the initializer (`__init__:171`); `_lx_relayout_plans` is written only at `:222`, `:254` and `:957`, all on the base chain StrategyB never enters.

**Put in its place:** nothing new â the enforcement already exists elsewhere and is honest. `allocator.py:244-246` (`if not self._lx_relayout_plans: ... return`) is what actually keeps `CoOptimizingAllocator` inert, since it inherits the base `plan_allocation` and reaches `:204` with `{}`. And the two `logger.warning` calls in `select_allocator` (`:2391-2396`, `:2413-2418`) are the only thing that tells a user their `config.lx_planner_relayout` is being ignored. Add one comment at `:244`: *"Both co-opt allocators bypass the hook that populates this, so they land here with `{}`."*

### 4. The unreachable `.get` arms in the allocator â ~6 lines

- `allocator.py:870` â `source is None`. `mem_usage_by_buf` emits one entry per graph operation (`scratchpad/utils.py:134-171`); `calculate_liveness` (`:107-114`) appends an index for every op naming the buffer; `_build_bound_buffers` (`allocator.py:600-603`) drops only empty-lifetime buffers. A plan source is a `ComputedBuffer` with at least one consumer, so it is always in `by_name`.
- `allocator.py:871`, `:876` â `consumer_tick is None`. `op_index` (`:866`) is keyed by `op.get_name()` over `graph.operations`; `plan.consumer_name` came from the same list, and nothing mutates it between `:223` and `:866`.
- `allocator.py:876` â `consumer_tick not in source.uses`. Both sides read the *same memoized* `op_read_writes` over the same op list. This is the archetype of the class-B harm: it tells the reader the liveness map and the plan map could diverge, when they are one source of truth.
- `allocator.py:921` (`source is None` half) and `:925` (`destination is None` half). All three solvers return `list(self.buffers)` (`greedy_solver.py:173`/`:217`, `firstfit_bestfit_solver.py:243`, `ilp_solver_ortools.py:441`), and surviving plans were pruned at `:957-961`. **Keep the `address is None` halves of both lines** â those are the atomicity gate.

### 5. `spyre_kernel.py:1519-1523` â 5 lines

`if new_dim in new_splits and (...) != (factor, new_slot): raise ValueError("conflicting normalized ownership")`. `new_dim in new_splits` is never true: `align_tensors` resets `_synthetic_var_idx = len(new_vars)` immediately before the splits loop (`views.py:614`), forcing `synthetic_var()` into its mint branch every time, and `remap[var][0]` is the old var itself, distinct per var. One fresh symbol per `(old dim, segment)`, no collisions possible. Five lines telling the reader two loop dims can share an owner. **Replace with** that one-sentence invariant as a comment.

### 6. `superdsc.py:823-829` â 7 lines

The `assert all(split == 1 or dim in dim_order ...)` (`:823-825`) is strictly weaker than the `ValueError` at `:996-999`: the assert fires only when a `split>1` sits on a dim absent from `dim_order`, and `:827` immediately drops that key, so `_finalize`'s re-projection sees the product divided by that split and raises anyway. It can never fire alone, because every construction site bounds the product *above* by `num_cores` (`lx_relayout.py:184-185` asserts `math.prod(...) == num_cores`; `work_division_from_view` can only merge dims; `_remap_work_division`'s gcd chain preserves the product or raises at `spyre_kernel.py:1510`). The `split == 1` disjunct is dead at source: `pass_utils.py:1580-1582` skips splits â¤ 1, so a `PerCoreView` never records one.

Better than deleting the assert alone: **delete the projection at `:826-829` too** and store the symbol-renamed division verbatim. `_finalize` then does the single projection onto `mapping_dims`, and the assert's condition becomes unconstructible rather than merely unfired. Nothing reads `sdsc_arg.work_division` between `:829` and `:1274`.

### 7. `lx_relayout.py:199-205` â 7 lines, `_op_short_name`

A second, independently-written spelling of `allocator.py:1068-1089`, which predates this branch (present at `b4ae70f:886`) and serves the same purpose (`allocator.py:569` identifies restickify ops with it). Behaviourally equivalent on every input I could construct â which is precisely the hazard: two spellings of one fragile origin-resolution heuristic drift silently. The duplication is forced by a real import cycle (`allocator.py:90` imports `lx_relayout`), so the fix is to **hoist the helper into `pass_utils` and import it from both**, not to import across the cycle.

### 8. Scheduler â ~5 lines

- `scheduler.py:491-492` â `_lx_view(plan.source_name) is None or _lx_view(dep.name) is None`. A registry entry exists only because `materialize_lx_relayouts` created it, and that stamps `allocation["lx"]` and `lx_view` on both source and copy in the same loop (`lx_relayout.py:367-371`). The only mutators of either are `lx_relayout.py:120`/`:345-348`/`:367-371`, `allocator.py:229` and `allocator.py:1037`, none of which runs between materialize and this pass. Delete â it also removes two `_lx_view` calls from a hot loop.
- `scheduler.py:493` â `_lx_view(source) == _lx_view(copy)`. Redundant with `if view == source_view: continue` in `collect_lx_relayout_plans` (`lx_relayout.py:299-300`). Keep the planner's; it is at the layer that decides what is worth relaying out.
- `scheduler.py:330-332` â `if buffer is None: return None`. **Convert to `assert`, don't delete.** Three of the four call sites (`:491`, `:492`, `:524`) have proven provenance; `:372` takes any `MemoryDep` write name of an lx-resident node, which I could not fully close. The assert costs the same line and removes a real semantic overload: `None` currently means both "lookup failed" and "no planned relayout view", and at `:534` the second meaning silently downgrades the strict per-view check to the weak users-agree check.
- `scheduler.py:336` â `getattr(layout, "lx_view", None)` â `layout.lx_view`. Unreachable given `:334`: only `FixedTiledLayout` carries an `allocation` dict, and `ir.py:104-106` sets `lx_view` in the same `__init__`.

### 9. The two `compute_ops` asserts â 2 lines

`compute_ops.py:1087` and `:1114`, both `assert tensor.work_division is not None`. The backfill at `:589-600` runs unconditionally at the top of the same function, and the only two assignment sites in the tree (`compute_ops.py:600`, `superdsc.py:826`/`:1000`) never write `None`. `:1114` is doubly unreachable â it is reached only when `opfunc == "shuffle"`, written at exactly one site (`superdsc.py:1285`) under `is_lx_relayout_identity`, which itself requires both divisions non-`None`. Not serving mypy either: `tensor` is an unannotated parameter (`Any`), and `pyproject.toml` runs default strictness.

**Note the correction to the third pass:** the *backfill itself* at `:589-600` is **not** dead and must stay â `generate_sdsc` is called directly 14 times from `tests/inductor/test_coarse_tiling.py` (`:2323`, `:2342`, `:2360`, `:2375`, `:2393`, `:2409`, `:2425`, `:2481`, `:2545`, `:2607`, `:2946`, `:3015`, `:3086`, `:6657`) on hand-built `SDSCSpec`s that never set `work_division`. It is a defaulting step, not a guard. Delete the asserts, keep the default. (`test_provenance.py` is unaffected â all three of its `generate_sdsc` calls go through `parse_op_spec` first.)

### 10. Smaller items â ~6 lines

- `lx_relayout.py:175-176` â `min(fanout) > 0` / `min(fanin) > 0` are implied by their siblings in the same tuple: `bool(edges)` (`:174`) plus `len(set(fanout)) == 1` (`:178`) gives every core the same fanout `v` with `num_cores*v == len(edges) > 0`, hence `v > 0`. Same for fanin with `:179`.
- `lx_relayout.py:221` â `getattr(config, "ktir_emitter", False)`. `config.py:34` declares it unconditionally at module scope and the module ends with `install_config_module`; the default can never be taken. The same line reads `config.lx_planner_relayout` directly and `spyre_kernel.py:748` reads `_spyre_config.ktir_emitter` directly.
- `op_spec.py:210-211` â the two `"lx" in ...allocation` conjuncts, implied by the `work_division is not None` conjuncts below them (`work_division` is populated at exactly one site, `spyre_kernel.py:771-775`, under that very test, and `TensorArg.allocation` is the same dict object, round-tripping verbatim through `allocation={arg.allocation!r}` at `:1455`). Low value either way â this is a six-line *definition* predicate and the allocation test is what makes its name true on its face. If you drop them, put the coupling in the docstring.
- `lx_relayout.py:232-234`, `:367-371` â eight `getattr(x, "layout")` calls with no default. These are not guards at all (no-default `getattr` *is* attribute access) and exist only to placate a type checker; three die with the replay block. One `cast(FixedTiledLayout, source.get_layout())` reads better and type-checks the same. They contribute nothing but they inflate the "20 getattr" census the colleague is reacting to.
- `lx_relayout.py:87-88` (`device_dim >= len(device_coordinates)`) and `:196` (the `len(writes) == 1` half of `_single_write`) â **convert to asserts, not deletions.** `compute_coordinates` returns exactly one expression per device dim (`views.py:175-183`) and both call sites pass the same buffer's `device_layout`; a `ComputedBuffer` performs exactly one store. On `:196` the assert must *precede* the index, since the length test also prevents an `IndexError`. The `is_indirect()` half of `:196` stays â see below.
- **Free but not on this branch's budget:** `superdsc.py:1154-1157` (`if missing_dim is not None: ...`) is provably dead at base â `missing_dim` occurs only at `:577` (`= None`) and `:832` (returned), never assigned. Four free lines in the function this branch rewrites.
- **A dead parameter with a real cost:** `collect_lx_relayout_plans(graph, cache=None)`. The sole caller (`allocator.py:223`) passes only the graph, so every `_per_core_view_on_buf` inside recomputes the sympy-heavy `_prepare_per_core_view` from scratch â once per producer and once per consumer edge â while the memo machinery it threads into (`pass_utils.py:1725-1753`) is used by `get_ncores_for_buffers` and the co-opt search, which do pass a cache. Either wire it through or drop the parameter; today it advertises an optimization that is never taken.

---

## Make these loud

These matter more than the deletions. There are fewer of them than the census implies â there is no bare `except`, no silent `continue` over a real error, and most of the `.get(x, default)` calls are legitimate total-fills. Four items:

### 1. `lx_relayout.py:93` â `dict(view.core_to_slot).get(device_dim, 0)` â the dangerous one

The key is always present (see the geometry invariant above), but the **default is the problem, not the reachability**. A miss silently yields *"every core owns slot 0"*, and this value goes straight out as the SDSC `coreIdToWkSlice_` map via `TensorArg.work_division` (`spyre_kernel.py:787`). Nothing downstream re-checks it: `_compatible_partitions`' distinctness test runs at planning time only. That is the worst failure mode this domain has. Index it and let a `KeyError` be loud.

### 2. `pass_utils.py:1684-1686` â where the assert actually belongs

```python
expr = core_to_slot_by_name.get(str(sym))
if expr is not None:
    pruned_core_to_slot.append(...)
```

This is the **only place** that could ever desynchronize `work_slice_dims` from `core_to_slot` â and it does so *silently*, by omitting the entry. The two defaults you would harden downstream (`lx_relayout.py:93` and `:133`) are consumers of that omission. Assert the invariant here, at the producer, then index (not `.get`) at both consumers. Hardening only the consumers leaves the actual failure site tolerant. `lx_relayout.py:133` has a backstop (a slot-0-everywhere row makes every core's row identical, rejected by the distinctness test at `:180-183`, so the cost is a lost optimization); `:93` does not.

### 3. Two `.get`s that should be direct indexes

- `superdsc.py:980-981` â `work_slices.get(dim, 1)` sitting next to `core_map[str(dim)]`, a direct index, on the very next line. The asymmetry is the tell. The pre-existing `mapping_splits = tuple(int(dim_splits[dim]) for dim in mapping_dims)` at `:1259` already direct-indexes over the same `mapping_dims`, and `work_slices` gains a key at every one of the six sites `dim_splits` does (`:1082/1083`, `:1107/1108`, `:1139/1140`, `:1156/1157`, `:1215/1216`, `:1245/1246`). Index both.
- `spyre_kernel.py:1496-1498` â `new_dims = iteration_remap.get(old_dim); if new_dims is None: raise ValueError(...)`. `ownership_remap` is written on both branches of the loop (`views.py:686`, `:713`) over `splits`, which is keyed by `var_ranges.keys()` == `iteration_space.keys()` (`views.py:573-575`, `:620-626`); `work_division` keys are minted from `tuple(it_space)` at `spyre_kernel.py:774`. `iteration_remap[old_dim]` already names the missing key and costs two fewer lines.

### 4. The one guard I would **add** â `superdsc.py:985-995`

`_finalize` applies a per-tensor override for *any* arg carrying one, with no tie to `opfunc`/shuffle. That is safe today only because `spyre_kernel.py:953-956` scrubs `work_division` off every non-relayout op â 700 lines away, in a different file. If that link breaks: `_finalize` takes the override, `_build_coord_info` computes `nsplits` from it (`compute_ops.py:1088`), and `_tensor_core_map` returns `{}` because `opfunc != "shuffle"` â an SDSC that slices a tensor one way while telling the backend nothing about who owns what. **The product check at `:996-999` does not catch this**: the override's product equals `_op_num_cores` of that consumer by construction (enforced at plan time, `lx_relayout.py:296`), so the product matches while the split *dims* differ. One assert here, or pass the already-computed shuffle flag in, restates the invariant at the layer that depends on it. Cheapest real safety line on the branch.

### Three that look like class D and are not

- **`lx_relayout.py:302-304`** (`is_matmul and read_index not in (0, 1)`) â I expected this to be the worst item on the list and it is not. A matmul `ComputedBuffer` cannot have more than two `MemoryDep` reads at this point: all three lowerings `realize()` both operands and issue exactly two loads (`lowering.py:339-342`, `418-421`, `481-484`); `propagate_layouts` runs at `passes.py:443`, well before `scratchpad_planning` at `:464`, and dispatches every non-mutation `BATCH_MATMUL` to `_matmul_layouts` (`propagate_layouts.py:1238-1242`) â `identify_matmul_inputs`, whose first statement is `assert len(inputs) == 2` (`pass_utils.py:644`) â a three-read matmul would have crashed an earlier pass. The one shape that reads three buffers is the tiled accumulator, a separate *Pointwise* op carrying `MutationLayoutSHOULDREMOVE` (`coarse_tile.py:2911-2944`), rejected at `lx_relayout.py:281` before this line. Also: `scaled_mm` takes `(mat1, mat2, out_dtype)` â no scale tensors in the graph. **Still worth hardening, but as an `assert len(deps) == 2` that states the invariant, not as tolerance for a state that cannot occur.**
- **`superdsc.py:816`/`:820`** (`symbol_mapping.get(dim, dim)`) â does not swallow anything. A miss keeps the raw symbol, which fails `dim in dim_order`, which trips the assert at `:823` â or, once that assert goes, drops a `split>1` and trips the `ValueError` at `:996-999`. Loud either way. `symbol_mapping[dim]` is a readability fix, not a safety fix. (The `sympify()` at `:820` and the `.xreplace(symbol_mapping)` on the slot are both genuine no-ops â slots are sympy `Expr`s in `core_id` alone â but they are the least valuable items here.)
- **`scheduler.py:512-514`** (`if buffer is None: continue`) â **do not tighten this**, despite it being the obvious inconsistency (`allocator.py:956` does the identical operation on the identical name set and fails loud). `cc90711` already deleted this block's two siblings (`layout is None`, `not hasattr(layout, "allocation")`) on the grounds that registry names are `FixedTiledLayout` by construction. True for the `:518` path. **Not established for the `:546` path**, where `source_name` is `source_by_copy.get(name, name)` and `name` came from `lx_names` â any `MemoryDep` write name of an lx-resident node. `_lx_resident` (`:323-326`) proves only that `node.node.layout` has an allocation dict; a second write dep of the same node could be a `MutationLayoutSHOULDREMOVE`, on which `clear_lx_relayout_state` would `AttributeError` at `lx_relayout.py:120`. Either narrow `lx_names` at `:472-475` to the node's own output name, or keep the tolerance. And note `allocator.py:956`, held up as the model to follow, is itself dead code (cluster 1).

### One that is loud in the wrong place

`superdsc.py:996-999`, `spyre_kernel.py:1497-1498` and `:1510-1511` are bare `ValueError`s raised from inside `simplify_op_spec` / `parse_op_spec`. Nothing in `torch_spyre` catches `ValueError` on that path â the only fallback mechanism in the tree is `except Unsupported` (7 sites, none wrapping `codegen_kernel`). So an LX plan the normalizer cannot handle becomes a **hard compile failure of the user's model**, for a feature that is purely a performance optimization â while this same branch's philosophy one layer up is to *demote and keep compiling on HBM* (`demote_incoherent_lx_buffers` â `clear_lx_relayout_state`). For `:1510-1511` at least, the right answer is probably to run `_remap_work_division` at plan time and demote, not to abort at codegen. That is a better question for the colleague than the 9-12% line count.

---

## These earn their lines

Do not re-open these. Grouped by what would go wrong.

**Silent wrong numbers if removed:**

- `allocator.py:886-891` â stripping planned sources from every buffer's `in_place_parents`. The keystone of the atomicity contract and the least guard-looking line in the file. `:902-904` rewrites `source.uses` to end at the transfer tick, which breaks `_assert_in_place_relationships`' `parent.end_time == child.start_time + 1` (`plan_solver.py:246-249`) for any child that declared the source as parent â and `_determine_in_place` (`:778-804`) genuinely creates such edges for a source's last pointwise consumer with a different work division, exactly the shape of `test_lx_relayout_two_private_consumers_are_numerically_correct`. **Fix the comment though:** the "in-place alias of the destination onto the source, copy reads its own output" story is *not* substantiated â the destination is constructed at `:905-910` with empty `in_place_parents` and nothing ever appends to it. Only the assertion-violation half is real.
- `spyre_kernel.py:953-956` â scrubbing `work_division` / `lx_consumer_is_matmul` off non-relayout ops. Looks like belt-and-braces; is the single thing preventing an ordinary op that merely *reads* a relayout source (which by construction still has its other consumers) from inheriting a per-tensor split, changing `nsplits` in `_build_coord_info`, and emitting an empty `coreIdToWkSlice_`. Pinned by `test_work_division_hint.py:673`.
- `scheduler.py:498-500` â the missing-copy â demote check. A registry entry whose copy has no `SchedulerNode` means the source still advertises `lx_view` while nothing performs the transfer: every core reads another core's slice. Test-covered (`test_work_division_hint.py:836`, `run_registered("missing")`).
- `scheduler.py:534-538` â the strict per-view check for relayout buffers. Ordinary LX buffers only need their users to agree with each other; a relayout buffer's users must match the view the planner committed and codegen was built against, because the addresses and the ownership map were fixed at plan time.
- `allocator.py:922` / `:925` (the `address is None` halves) â the all-or-nothing gate. Exercised by `test_lx_relayout_allocation_is_atomic_and_retries_stock_once`.
- `lx_relayout.py:89-91` â `len(matches) != 1`. Reachable in *both* directions, and the reaching path is the probe itself: `:318` deliberately calls `work_division_from_view(source_view, consumer_coordinates, consumer_symbols)`, and a consumer that does not index one of the source's split dims gets `coordinates[dim] == 0` (`views.py:217-218`), so `matches` is empty. The â¥2 case is the collapsed-axis TODO at `pass_utils.py:1631-1642`. The silent alternative â `next(iter(matches))` â writes an arbitrary loop symbol into the ownership map.
- `lx_relayout.py:94-95` â conflicting-ownership raise. Two device dims can project onto one loop symbol (the stick decomposition puts one host dim on both the num-stick and within-stick dims). Without it the second write silently overwrites.

**Prevents a hard compile crash:**

- `lx_relayout.py:316-328` â the `try/except ValueError` feasibility probe, 13 lines and the most valuable block in the file. Catching this masks nothing: the *identical* call is made **unguarded** at codegen (`spyre_kernel.py:771-775`), so without the probe an unprojectable view escapes as a raw `ValueError` from the middle of `create_tensor_arg`. It is a narrow catch, not a bare one â `Unsupported` is a `RuntimeError` (`errors.py:16`), so it is not swallowed. Two honest caveats: (a) it does not cover consumers that took `if view == source_view: continue` at `:299-300`, which still read a buffer now carrying `lx_view`; (b) it validates against *pre*-fusion iteration spaces (`pass_utils.py:816-836`) while the codegen call uses *post*-fusion ranges (`spyre_kernel.py:750`). Damage is bounded to a compile crash, not wrong numbers (`spyre_kernel.py:953-956` nulls `work_division` on non-identity ops). The real fix, if you want the guarantee closed, is to re-run the projection inside `demote_incoherent_lx_buffers`, which already walks exactly the right post-fusion `(node, dep)` pairs at `scheduler.py:521-537`.
- `allocator.py:828-829` â `if name not in mem_usage: continue`. Fires on **every** relayout: the synthetic `__spyre_lx_relayout__:` destination is not yet a graph operation, so `mem_usage_by_buf` has no entry and `:832` would `KeyError`.
- `allocator.py:1013` â the prefix skip. Fresh-path destinations are placed, addressed `LifetimeBoundBuffer`s with no graph buffer. (Nit: the string literal duplicates `_DESTINATION_PREFIX`, private to `lx_relayout.py:47` â use `self._planned_lx_buffers()` or export a predicate.)
- `op_spec.py:206-207` â `len(args) != 2`. Not the vacuous arity check it looks like. The indirect/gather store path builds `[idx..., value, dst]` (`spyre_kernel.py:1090-1111`) and still labels the op `IDENTITY_OP` (`:1126`/`:1130`), and `create_op_spec` calls this predicate on *every* op (`:953`). Without it, `source, destination = args` raises for every such kernel, LX or not.

**Reachable policy, not defence:**

- `lx_relayout.py:256-261` (producer type filters), `:277-283` (consumer filters â and `break`, not `continue`, is deliberate: it skips the for/else at `:339-340` so the group is vetoed atomically), `:285-289` (indirect reads), `:290-298` (core-count match), `:299-300` (the selection criterion itself), `:208-215` (`_is_activation_source`). None of these is enforced upstream â in particular `_per_core_view_on_buf` is called for the consumer with `buf_name = source_name` (`:291-293`), so it never inspects the consumer's own layout, and `GraphEditor.is_rewritable_consumer` (`graph_editor.py:245-263`) checks only `.data`, never the layout. `lx_relayout.py:281` is the *only* screen against a `MutationLayoutSHOULDREMOVE` consumer.
- `allocator.py:872` â `source.residency_reason is not None`, and it **cannot be hoisted into the planner**: `:830-835` deliberately rewrites `ncores` and clears `core_div_mismatch` for planned buffers *before* `_residency_reasons` runs at `:843`, so the verdict is plan-dependent. Once the unreachable arms above are deleted this is the only surviving feeder of `invalid` â say so, because `invalid` currently looks like it collects four failure modes.
- `scheduler.py:372` â excluding relayout buffers from producer loop-order alignment. Two reasons, and only the first is documented: reordering a relayout *source* invalidates the committed ownership map while the `lx_view` stamp stays (silent wrong numbers); and the *copy* must read under `source_view` and write under `destination_view` simultaneously, so it cannot be aligned to either side.
- `scheduler.py:528-529` â skipping the copy when checking the source's coherence. Without it every accepted relayout demotes itself on the very next loop.
- `scheduler.py:334` (`"lx" not in getattr(layout, "allocation", {})`) â the default is correct: a CPU-resident `ComputedBuffer` has a plain `FixedLayout` with no allocation dict, and "no attribute" and "not in LX" are the same answer.
- `superdsc.py:996-999` â the product check. Load-bearing on the **override** branch (`align_tensors` truncates the *op's* division per dim at `views.py:685-686` while `_remap_work_division` either absorbs the tensor's differently-shaped split exactly or raises, so the op's product can shrink while the tensor's does not); tautological on the other. Keep it â it is the one that should survive the pair with `:823-825`.
- `compute_ops.py:1112-1113` (`opfunc != "shuffle"` gate) â the sole place the per-tensor map is gated on the DeepTools contract, and what keeps `coreIdToWkSlice_` byte-identical (`{}`) for every non-relayout op.
- `lx_relayout.py:81-82`, `:104`, `:350-352` (the `return`), `:366` (the address assert â already in the form this audit prefers, documenting the allocator's atomicity contract in one line), `scheduler.py:487-490`, `:505-506`, `:546`, `allocator.py:244-246`, `:892-893`, `:967-974` (indexed, not `.get` â the one place the branch lets a bug crash instead of degrade, and exactly the right place).
- `allocator.py:253-257` (stock re-solve) â a heuristic, not a guard, and correctly kept. Dropping it cannot produce wrong numbers. But **document the asymmetry**: it fires only when *nothing* is complete, so on partial rejection a source the solver *did* place has its address discarded at `:948-950` with no second chance (`test_work_division_hint.py:731` asserts exactly this). That is the real cost of atomicity and the code says nothing about it.

**One that looks load-bearing and is not â but keep it anyway:**

`compute_ops.py:1122-1125` (`if not 0 <= slot < split: raise`). The third pass called this the last line of defence against a silent wrong ownership map. It cannot fire: `_tensor_core_map` returns `{}` unless `opfunc == "shuffle"` (`:1112`), so only per-tensor *overrides* reach it, and every override has passed through `_remap_work_division`, which writes the pair together at `spyre_kernel.py:1524-1525` â `new_splits[d] = factor` and `new_core_map[d] = Mod(floor(slot/stride), factor)`, the **same** `factor`. `Mod` by a positive modulus is in `[0, modulus)` for every `core_id`. Keep the four lines as forward insurance â the TODO at `superdsc.py:1267` ("choose the mapping before LX planning and pass it through to codegen") has an obvious implementation, a precomputed coreâslot table, that *would* need this check â but re-document it. Its real content is *"every slot expression is a `Mod` by the split it is paired with"*, an invariant spanning three files. Presenting it as a live defence sends the next reader hunting a bug class that cannot occur.

---

## The pattern

The habit worth naming is **guarding at every layer the value passes through, instead of asserting the invariant once where the value is produced.** The clearest case is `PerCoreView` well-formedness: it is established in one loop at `pass_utils.py:1659-1686`, and then re-checked four times downstream (`lx_relayout.py:93`, `:133`, `:134-137`, `:153-154`, `:266`) â none of which can fire, while the one place that *could* silently break it, the `.get(str(sym))` omission at `pass_utils.py:1684-1686`, is the only site with no check at all. The same shape repeats with the ownership map's key totality (`superdsc.py:823-825` re-checking what `:996-999` already covers), with `work_division` being populated (`compute_ops.py:1087`/`:1114` re-asserting a backfill 490 lines above), and with `"lx" in allocation` (`op_spec.py:210-211` re-testing the exact condition under which the field it sits next to was populated, at `spyre_kernel.py:771-775`). The cost is not the lines â it is that a reader cannot tell which check is doing the work, so nobody dares delete any of them, and the one layer that actually needs hardening looks like the one that already has four backstops. The second, larger lesson is that **an unstated invariant grows a feature to defend it**: because nothing said "collect runs once per compile, on a virgin graph," someone wrote a coherent 44-line re-entrant planner path â registry replay, a `destination_buffer_name` field, a graph-wide `lx_view` reset, and matching arms in the allocator at `:874-875` and `:953-956`. It is good code for a mode that does not exist. One sentence in a docstring would have been cheaper than all of it, and would have been checked by the next reader instead of quietly rotting.
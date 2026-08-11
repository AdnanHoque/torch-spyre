# LX relayout — how much leaner can this get?

Asked of `c60f5d81`: 983 source insertions plus 299 test insertions over base `b4ae70f`.

Short answer: about 35 lines, ~3.5%. A trim, not a step change, and most of it is not
worth doing on a branch two review passes deep.

The section that matters most is **Do not cut** — three attractive-looking reductions are
net-negative or fatal, including one pairing that breaks every compile in the tree
because each half was justified by pointing at the other.

---

## The short answer

**About 35 lines. Roughly 3.5%. That is a trim, and most of it is not worth doing.**

C is already close to the floor for this feature. Of the 983 source insertions, the biggest honest saving I can defend line-by-line is ~34 lines, and half of those are motivated by drift-safety rather than by leanness. There is no 300-line step change hiding anywhere: the one deferral of that size (grouped gathers, #3440) has already been taken, and every remaining axis is either under 50 lines or is the reason the feature exists.

The reason the floor is high is structural, not stylistic. Roughly 40% of this diff is validation, and this machinery fails silently â a wrong ownership map produces wrong numbers with no error. You cannot buy leanness with guards here.

Three of the most attractive-looking cuts turned out to be net-negative or fatal on inspection, and they are documented in **Do not cut** so nobody re-proposes them. One of them â deleting `_finalize_tensor_work_divisions` *and* the `compute_ops` backfill together, each justified by pointing at the other â breaks every compile in the tree.

Given that two review passes have just been through this branch, my recommendation is: take the ~20 free lines in items 1â5 below, take the two safety consolidations (items 9â10) because they remove real drift hazards, and stop. Spend the remaining effort on the one missing test named at the end instead.

---

## Cuts worth taking

Ordered by lines saved. Only cuts that survived adversarial checking against the tree at `c60f5d8`.

### 1. Inline `insert_clone_before_consumer` â 7 lines
**Where:** `torch_spyre/_inductor/scratchpad/graph_editor.py:229-236`, call site `torch_spyre/_inductor/lx_relayout.py:354`
The whole body is `return self.push_allocation_with_clone(buffer, [consumer], input=True, private=True)`. One caller in the tree; no test names it. C already collapsed alternative A's separate 60-line inserter into the `private=` flag, so this wrapper's only remaining job is to name a flag combination.
**Lost:** a self-documenting name at the call site. Replace with a two-word comment; `private=True` *is* the semantics.

### 2. Replay branch: three dead stores plus the `replace` block â 7 lines
**Where:** `lx_relayout.py:224-226` (dead stores), `:227-235` (the `dataclasses.replace` call spread over nine lines)
The three stores write `lx_view` and `lx_consumer_is_matmul` onto layouts that nothing reads before `materialize_lx_relayouts` unconditionally resets *every* `FixedTiledLayout` at `lx_relayout.py:336-339`. I checked every reader of both fields at `c60f5d8`: `scheduler._lx_view`, `spyre_kernel.py:772`, `spyre_kernel.py:788`, `superdsc.py:551` â all downstream of materialize.
**Lost:** nothing today. The stores are what would make `collect_lx_relayout_plans` self-consistent for a caller that collects without materializing; there is no such caller.

### 3. Hoist producer-invariant work out of the consumer loop â 4 lines
**Where:** `lx_relayout.py:300-302` and `:313-317`, inside `for consumer, dep, read_index in consumer_reads:` at `:263`
`producer_coordinates = try_device_coordinates(producer.layout.device_layout, write, None)` depends only on `producer` and `write`. So does the third validation call, `work_division_from_view(source_view, producer_coordinates, tuple(iteration_space_from_op(producer)))`. Both run once per consumer. Move them up beside the existing source precondition at `:254-259`.
**Lost:** nothing. A producer-side failure currently `break`s, which skips the `for/else` at `:330` and drops every accumulated plan for that source; hoisted it becomes a `continue` on the source â identical outcome. This also deletes one `iteration_space_from_op` and one sympy projection *per consumer*, so it is a compile-time win as well as a line win.

### 4. Inline `_accepted_plans` â 4 lines
**Where:** `scratchpad/allocator.py:959-970`, single call site `:253`
A 12-line method that is one comprehension. The test exercises `_finalize_lx_relayout_allocation`, never `_accepted_plans` directly.
**Lost:** nothing â **on one condition**: keep the bare `by_name[plan.source_name]` / `by_name[plan.destination_name]` subscripts. That `KeyError` is the tripwire for a plan reaching acceptance without a placed destination buffer. Do not "tidy" it into `.get`.

### 5. Two smaller dedups in the same loop and in the scheduler â 3 + 3 lines
**a.** `lx_relayout.py:241` already builds `[d for d in op_read_writes(consumer).reads if isinstance(d, MemoryDep)]` for every consumer while indexing `reads`; `:272-274` rebuilds it verbatim inside the loop just to evaluate `any(d.is_indirect() ...)`. Record the flag on the first walk. Today's recompute is O(sources Ã consumers).
**b.** `scheduler.py:458` + `:461` build `copies_by_source`, whose single reader is `names = {source_name, *copies_by_source.get(...)}` at `:504`. But `discard_lx_relayout_group` **already returns exactly that set** (`lx_relayout.py:101-112`) and its return value is discarded at `:505`. Reorder to `names = {source_name, *discard_lx_relayout_group(V.graph, source_name)}`.
**Lost:** nothing. **Trap:** `source_by_copy` (`:459`, `:462`) is a different index with its own reader at `:546` â it stays. And any inline derivation must read the pre-built `plans_by_copy` snapshot, never call `materialized_lx_relayouts` a second time, because `discard` mutates the registry.

### 6. Collapse the two `select_allocator` warnings â 3 lines
**Where:** `allocator.py:2385-2389` and `:2407-2411` â the same three-line message with one class name changed.
Both bypasses are real: `StrategyBCoOptimizingAllocator.plan_allocation` never calls `_prepare_buffers`, and `CoOptimizingAllocator._prepare_buffers` overrides without calling `super`, so `_lx_relayout_plans` stays `{}` and `_finalize_lx_relayout_allocation` early-returns at `:239-241`.
**Lost:** nothing. Pure logging, zero correctness surface.

### 7. Delete the dead reset at `allocator.py:249` â 1 line
`self._lx_relayout_plans = {}` inside `if not complete:`. When `complete` is empty and the dict is non-empty, `rejected` is every source, so `_clear_lx_relayout_groups`'s trailing comprehension at `:953-957` has already emptied it. When the dict is empty the function returned at `:239`.
**Lost:** nothing â and worth removing because leaving it there implies the retry owns a reset it does not own.

### 8. Delete the `compute_ops` backfill â 6 lines *(borderline; take only if you are already in those test files)*
**Where:** `codegen/compute_ops.py:589-594`
Confirmed production-dead: `parse_op_spec` calls `_finalize_tensor_work_divisions` at `superdsc.py:1271`, which assigns `arg.work_division` unconditionally at `:997` on the same list that becomes `SDSCSpec(args=args)`. It is also latently wrong â it pairs `sdsc_spec.work_slices` (Symbol-keyed) with `sdsc_spec.core_id_to_work_slice` (str-keyed, indexed as `core_map[str(dim)]` at `superdsc.py:978`), violating `TensorWorkDivision`'s own invariant. It only survives because `_tensor_core_map` early-returns for non-shuffle opfuncs.
**Lost:** you must update 8 hand-built `SDSCArgs(...)` fixtures in `tests/inductor/test_coarse_tiling.py` plus `tests/inductor/test_provenance.py` â two files this branch does not currently touch. **Do not** take the tempting alternative of a silent fallback to `sdsc_spec.work_slices` in `_build_coord_info`: on the production path `spyre_kernel.py:955` nulls `work_division` on every non-shuffle arg, so the only specs carrying a per-tensor override *are* the shuffles, and substituting the operation-level division there is silent wrong numbers. **Keep both asserts** (`compute_ops.py:1081`, `:1108`).
Six lines against destabilising two untouched test files is, in my view, not a trade worth making on this branch right now.

---

### Two changes that do not save lines â take them anyway

**9. One helper for the three-field LX clear.** `allocator.py:950-952` and `scheduler.py:513-515` are byte-identical: `allocation.pop("lx", None)`, `lx_view = None`, `lx_consumer_is_matmul = False`. Net ~2 lines, but the point is the drift: a demotion that pops the allocation and forgets `lx_view` leaves a stale ownership map on a buffer that is no longer LX-resident â exactly the defect class a prior pass already found here.

Make the helper take a *layout*, not a graph and a name. The two sites deliberately differ in lookup: the allocator uses `graph.get_buffer` (raises â a missing buffer there means the plan set is corrupt and must fail before an address and a view are committed to codegen), the scheduler uses `try_get_buffer` (tolerant â fusion may legitimately have removed buffers by then). Leave the lookup and its tolerance at each site.

**And do not apply the helper to `lx_relayout.py:337-339`.** That looks like the same three lines and is the opposite operation: it is a graph-wide annotation reset over *every* op, and it omits the `allocation` pop deliberately. `materialize_lx_relayouts` is the last statement of `_push_allocation` (`allocator.py:1029`), immediately after the loop that commits `layout.allocation["lx"]` for every allocated buffer. A shared helper that pops `allocation["lx"]` there erases every LX address the allocator just committed, on every compile, for every buffer in the graph. Everything silently falls back to HBM: numerically correct, catastrophic performance, no error. Give that loop a name that cannot be confused with demotion (`reset_lx_annotations`) and a one-line comment saying why the pop is absent.

**10. Import `_DESTINATION_PREFIX` at `allocator.py:1009`.** The skip predicate hard-codes `"__spyre_lx_relayout__:"` while the name is constructed from `_DESTINATION_PREFIX` at `lx_relayout.py:47, 66-67`. Costs +1 line (allocator already imports from `lx_relayout` at `:92-94`; the call site needs `f"{_DESTINATION_PREFIX}:"` because the constant carries no colon). Rename the prefix today and `_push_allocation` silently starts pushing allocations for synthetic destination names that have no graph buffer behind them.

### And one that reads as a cut but is not

**Dedupe `_op_short_name`.** `lx_relayout.py:191-197` reimplements `allocator.py:1064-1085`, and the duplicate exists only to dodge a real cycle (`allocator.py:92` imports `lx_relayout`). But allocator's copy is **pre-existing at the base**: `git grep` finds it at `b4ae70f:allocator.py:886`, untouched by this diff. Hoisting the shared version into `scratchpad/utils.py` (which both files already import, and which imports neither) *removes* 9 lines here and *adds* ~23 to a file the branch does not touch. Branch insertions go 983 â ~997. Tree LOC goes down 9.

Do it â two definitions of "what op is this" that must agree is a genuine drift hazard, since allocator's gates `OP_OUTPUT_GOOD_FOR_LX_REUSE` membership and lx_relayout's gates the `"restickify"` weight exclusion in `_is_activation_source` â but book it as a drift fix, not as leanness. Unify on `str(name)`: both consumers only do equality and membership against strings, so nothing changes for either, and the raw form can return a bound method for targets whose `.name` is callable.

---

## The one real scope decision

**There isn't one. Narrowing scope does not give a step change here â it deletes the feature.**

Two axes look like scope you could defer. Both are load-bearing, and the evidence is in the codebase's own comments.

**The matmul-consumer axis (~45 lines).** Four symbols thread one bit from plan-time to codegen: `SHUFFLE_LAYOUT_LABELS`, `shuffle_to_matmul`, `LXRelayoutPlan.consumer_is_matmul`, `TensorArg.lx_consumer_is_matmul`. Cutting them would mean "v1 relayouts only into pointwise consumers." That inverts the feature's value, because the pre-existing core-division pass already fixes pointwise mismatches for free:

- `allocator.py:1401-1406`: *"Only pointwise ops are flipped; reductions/matmuls keep work-division's split... overriding a matmul's split to chase pinning regressed kernel time ~2.5x (mlp-linear-kn.t, SENCORES=32; PT-util 66%â33%)."*
- `allocator.py:1451-1452`: *"Let this pointwise op adopt a matmul's tiling to pin its shared buffer to LX. High-value, so added regardless of DEFAULT_VARIANT_CAP."*

So a matmul's tiling is pinned **by construction**, and pointwise ops are already flipped onto it. The divergence that survives the flip machinery is precisely the matmul-adjacent one â in a transformer prefill, the normed hidden state into q/k/v, the same hidden into gate and up, the score BMM. A pointwise-only relayout fires only in the narrow residue where the flip declined. You would be spending 938 lines to solve a problem the existing pass already solves.

There is also a hardware-facing half. `shuffle_to_matmul` (`superdsc.py:739-742`) relabels the shuffle's first descriptor from `OUTPUT` to `KERNEL`, and `tests/inductor/test_work_division_hint.py:590` asserts `root["dscs_"][0]["shuffle"]["labeledDs_"][0]["dsType_"] == "KERNEL"`. Cutting the axis does not merely narrow coverage â it hands the PT a descriptor with the wrong role for a buffer it reads as its kernel operand.

**Multi-consumer fan-out (~24-38 lines).** One source feeding q, k and v â or gate and up â is by definition a source with several consumers whose tilings conflict. That is exactly the case no flip can solve, because a flip picks *one* split for the producer. The `for/else` at `lx_relayout.py:263-331` is what makes per-source atomicity real: any `break` skips the `else` and drops the whole source's plan list. That all-or-nothing guarantee is what licenses `allocator.py:830` setting `mem_usage[name]["core_div_mismatch"] = False` on a **real graph buffer** â an override that is only honest if *every* divergent consumer got a copy. Weaken the grouping and that override becomes a lie: a source pinned in LX with an un-copied divergent consumer reading it.

The single end-to-end numerical test, `test_lx_relayout_two_private_consumers_are_numerically_correct` (`tests/inductor/test_work_division_hint.py:628`), is exactly this shape and exercises both axes at once. Cutting either leaves the feature with no e2e coverage and no target workload.

**Close the question.** The scope decision that mattered â deferring grouped gathers to #3440 â has already been made. What remains is not scope; it is the feature.

---

## Do not cut

Things that look redundant, in decreasing order of how convincing the mistake is.

**`_finalize_tensor_work_divisions` (`superdsc.py:967-997`) â and never together with the `compute_ops` backfill.**
`git grep -n "work_division = "` at `c60f5d8` finds exactly three assignment sites in the source tree: `superdsc.py:823` (guarded by `if arg.work_division is not None` at `:814` â it only re-keys an existing division, never creates one), `superdsc.py:997` (unconditional, every arg), and `compute_ops.py:594` (backfill for every `None`). On the production path every non-shuffle arg arrives with `work_division` `None`, because `spyre_kernel.py:955` explicitly nulls them. Delete both unconditional assigners â each justified by pointing at the other â and `assert tensor.work_division is not None` at `compute_ops.py:1081` fires on the first tensor of the first op of every compile. Total compiler failure.

The prior recommendation to delete `_finalize` "outright" also takes its `math.prod(effective.work_slices.values()) != num_cores` raise at `:993-996` with it. That is the only ownership check that runs *after* both `align_tensors`/`_remap_work_division` and `symbol_mapping` â the last chance to catch a map where some core owns nothing or two cores own the same slice. And `tests/inductor/test_work_division_hint.py:601` asserts `all(arg.work_division is not None for arg in ordinary_sdsc.args)` after a bare `parse_op_spec`, where the backfill cannot cover for it.

Honest sizing, since it has been over-claimed: only the `else TensorWorkDivision({...for dim in mapping_dims}, ...)` branch is arguably duplicated work â about 9 lines, not 28.

**The `dim_order` projection at `superdsc.py:814-826` is not a redundant first hop.**
It projects the override onto *that tensor's* device dims, so a split on a dim the tensor does not have is dropped to 1 by `splits.get(dim, 1)` at `:824`. The `mapping_dims` projection at `:986-996` is a different key set (`mapping_dims` picks up `mb_sym`/`stick_sym`/`missing_dim`/`k_sym` added after `_create_sdsc_tensors`). Collapse the first into a pass-through and such a split *survives*, `math.prod` then equals `num_cores`, the guard passes, and `_tensor_core_map` (`compute_ops.py:1104-1122`) emits a `coreIdToWkSlice_` entry naming a dimension for which `_build_coord_info` â which iterates the tensor's `dim_order` â emits no `coordInfo` at all. A malformed shuffle descriptor handed to DeepTools. Add the comment it is missing, or a future reviewer will delete it.

**`_remap_work_division` (`spyre_kernel.py:1489-1527`) and the `ownership_remap` plumbing in `views.py`.**
The seductive proposal is "compute ownership once, after `simplify_op_spec`, by projecting the device-dim-keyed `PerCoreView` onto the already-normalized coordinates." Its precondition is false, and `views.py` proves it: `align_tensors` inserts restored size-1 dims, decomposes device dims, appends gap dims, splits the stick dim, and finally rank-extends **per tensor** â `views.py:788-790`: `gap = rank - len(t["size"]); t["size"] = [sympy.S.One] * gap + t["size"]; t["coordinates"] = [sympy.S.Zero] * gap + t["coordinates"]`. A `work_slice_dims` entry of `(0, 4)` projected after that reads `device_coordinates[0]`, which is now the constant zero: no free symbols, `ValueError`, hard compile failure. Where `gap` happens to be 0 it is worse â the stale index lands on a *different real* device dim, nothing raises, and the shuffle assigns cores to the wrong axis.

`_remap_work_division` is not a re-derivation of the device projection anyway; it is a loop-symbol-space remap that redistributes a split across the sub-symbols `align_tensors` minted. Delete it and `symbol_mapping.get(dim, dim)` leaves old vars unmapped, `splits.get(dim, 1)` turns every split into 1, and the shuffle tells all 32 cores they own the whole tensor. Note also that the little-endian slot ladder â the exact site of the sub-dimension endianness inversion a prior pass found â is pinned by an explicit regression test at `tests/inductor/test_work_division_hint.py:605-613`. The delicate code is delicate because the problem is.

**The registry (`_REGISTRY`, `LXRelayoutPlan.edge`, `discard_lx_relayout_group`) â do not replace it with a back-pointer on the copy's layout.**
The scheduler enumerates *expected* copies from the registry and then checks `if copy_name not in seen_copies: invalid_sources[plan.source_name] = f"missing relayout copy {copy_name}"` (`scheduler.py:494-496`). A field stored on the copy's own layout can only be found by walking buffers that still exist â when the copy is eliminated, the field is eliminated with it, and the check degrades into a no-op that can never fire.

Copy elimination is not hypothetical. The copy is `aten.clone`, and the immediate parent of this branch is `b4ae70f` â *"fix: ensure copies do not get deleted so they remain in WSR groups (#3615)"* â whose fix is a monkeypatch in `patches.py:140-142` stopping `remove_noop_ops` from eliminating `aten.copy`, `aten.clone`, and `aten.alias`. This feature depends on exactly the op class the codebase already had to protect.

The failure mode if the check stops working is silent: the source still carries `allocation["lx"]` and `lx_view` from `lx_relayout.py:358-360`, `demote_incoherent_lx_buffers` finds no mismatch because the only remaining user is the producer's own write, and the consumer reads a copy buffer with a valid LX address and no writer. Uninitialized LX read, no error. (A *source*-side forward pointer would preserve absence detection and is a legitimate alternative â but carry `num_cores` as a second field rather than recomputing it, because `_op_num_cores` on a clone whose `op_it_space_splits` came from `_clone_output_splits` can yield 1.)

**The three `work_division_from_view` pre-flights (`lx_relayout.py:307-319`).**
Their return values are discarded; only the `ValueError` matters, which makes them look like ten wasted lines. The same function is called for real at `spyre_kernel.py:771` with no `try/except` anywhere up the stack. These pre-flights are what turn "cannot project this ownership map" into "skip the plan, round-trip HBM" instead of a hard compile abort. They are an approximation, not a proof â they run against op-level iteration spaces while codegen uses the kernel's â so they shrink the crash surface without closing it. Keep them regardless.

**The source/destination address-overlap check (`allocator.py:917-923`).**
This is the **only** address check in the entire feature. `demote_incoherent_lx_buffers` compares per-core *views*, never addresses. Overlap does not fault: the SHUFFLE identity copy reads bytes it already overwrote and produces garbage, silently.

**`_compatible_partitions` and `_overlap` (`lx_relayout.py:136-181`).**
The proposed algebraic collapse â replace the O(coresÂ²) overlap graph with per-dimension `S_d % D_d == 0 or D_d % S_d == 0` â is mathematically *correct*, given the other guards in the same `all(...)`: both coreâslot maps are bijections and both split products equal `num_cores`, so both views are regular product grids, the overlap count factorizes per dimension, and uniform fan-in/fan-out is exactly per-dim divisibility. That is precisely why it should not be taken. The identity holds only because of its neighbours; any future weakening of those neighbours silently invalidates it, and this function is the proof that the shuffle is a permutation with exactly one writer per byte. Trading a check a reader can eyeball for one a reader must prove is the wrong direction on the one predicate that decides whether a relayout is materializable at all.

**`spyre_kernel.py:953-956` â the loop that nulls `work_division` and `lx_consumer_is_matmul` on every non-relayout op.**
It reads as plumbing. Without it, an ordinary op reading or writing a relayout source emits `coordInfo` nsplits taken from the relayout's view instead of its own operation division. Six lines that never fire in a passing test.

**`scheduler.py:489` â the `_lx_view(source) == _lx_view(copy)` clause.**
One line, and `op_spec.is_lx_relayout_identity` looks like it makes it redundant. It does not: that predicate runs later, on post-normalization `TensorWorkDivision`s, inside the regenerated wrapper's process; this clause guards a mutable layout annotation at planning time. More decisively, `allocator.py:821-830` deliberately switches off the normal `core_div_mismatch` safety net for exactly these buffers, which makes `demote_incoherent_lx_buffers` the only remaining check that disagreeing users are mediated by a shuffle. One line is not worth thinning a tripwire on the one path where the usual guard was turned off.

**The all-fail re-solve (`allocator.py:248-252`).**
It is genuinely perf-only â by the time `if not complete:` is reached, `_clear_lx_relayout_groups` has nulled every source's address, so the truncated source lifetime can never ship alongside a consumer still reading from LX. But it is not free to remove: `test_lx_relayout_allocation_is_atomic_and_retries_stock_once` asserts `Solver.calls == 2` at `tests/inductor/test_work_division_hint.py:735`. Five lines, in exchange for deleting a deliberately-written test and losing LX residency for every planned source on the fallback path. Leave it.

**Per-source atomicity in the allocator (`allocator.py:905-957`).**
It is arguable that the scheduler's demote pass would catch a dropped edge â `view != expected` plus `demote(source_by_copy.get(name, name), culprit)` does tear the group down. But that makes correctness depend on a downstream pass in a different file, behind `if not _spyre_config.lx_planning: return nodes`, in a feature where a wrong ownership map produces wrong numbers with no error. The allocator enforces it locally for ~15 lines.

---

### Where the effort actually belongs

The `missing relayout copy` branch (`scheduler.py:494-496`) â the guard that catches an eliminated copy, on an op class this repo already had to patch to keep alive â **has no test**. `test_lx_relayout_scheduler_demotes_groups_but_not_ordinary_unary` hard-codes `_spyre_lx_relayout_copies={plan.edge: ("destination", plan)}` and keeps the copy node present in both of its cases; it only varies which node's per-core view drifts. A third case that drops the destination node from `nodes` and asserts the source is demoted is worth more than every line saved above, because it is the difference between "the guard works" and "the guard has never once been observed to fire."

Second, one asymmetry nobody has costed, offered as an observation rather than a cut: on the replay path `plan.destination_buffer_name` names a real graph buffer, so `_append_lx_relayout_destinations` skips it, `entries` ends up empty, and the `2*use+1` tick rescale plus the synthetic `LifetimeBoundBuffer` never happen â replay places source and copy under a completely different lifetime model than pass 1, and `_accepted_plans` then indexes `by_name[plan.destination_name]` unguarded. Nothing in the suite exercises a second `scratchpad_planning` pass over the same `GraphLowering`, and the only second `plan_allocation` call (`allocator.py:2432-2442`, on `SolveError`) fires strictly *before* `materialize_lx_relayouts`, so the registry is empty on that retry. That makes the replay branch (`lx_relayout.py:216-236` plus `destination_buffer_name` and the branch at `:350-355`, ~28 lines) the largest single unclaimed candidate in the diff â and also one whose failure mode, if it turns out to be reachable, is a second clone chain inserted silently. Do not remove it on the strength of a grep. Either write the test that proves it unreachable, or give it a comment naming what it defends.
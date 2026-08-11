# LX relayout — final review

Reviews `cd6b0005` ("Simplify LX relayout lifecycle"), the fourth and last commit in this
review cycle. Net −25 lines while *adding* invariant checks: the dead re-entrant-planner
path is gone, `_op_short_name` is hoisted to `pass_utils.py:113` rather than imported across
a dependency that would have cycled, and both `.get(dim, 0)` owner-slot defaults are replaced
by direct indexing.

**Verdict: ship after two one-line edits.** Neither is a correctness fix — nothing is broken
today. Both are altitude fixes, where a *skip* became a *compile abort* inside a pass whose
written contract is never to fail a compile.

Two concerns raised before this pass were checked and dismissed, and the reasoning is
recorded here so they are not raised again:

- The `assert len(deps) == 2` FP8 worry is unfounded. `lower_scaled_mm`
  (`lowering.py:333-412`) builds the reduction from exactly two inputs; scales come from
  separate ops. The `padding.py:194` cross-reference does not transfer either — padding
  counts any dep with a `name`, including StarDep and WeakDep, while the planner counts only
  `MemoryDep`. The one genuine 3-`MemoryDep` matmul, the tiled accumulator, always carries
  `MutationLayoutSHOULDREMOVE` and is rejected 32 lines earlier.
- `assert` is a safe carrier in this repo. No `PYTHONOPTIMIZE`, no `-O`/`-OO` in any script,
  config or workflow, no Dockerfile; the tree already relies on 153 assertions including
  pre-existing correctness gates. Two new checks would nonetheless be silent-wrong-numbers if
  ever stripped and are called out below as candidates for explicit raises.

One change went unremarked through three passes and is the most substantive correctness
improvement in the commit: the reordering at `lx_relayout.py:279-292` moves the source-view
projection check above `if view == source_view: continue`, closing the probe blind spot where
`materialize_lx_relayouts` stamps `lx_view` on a shared source and `create_tensor_arg`
projects it for every reader with no try/except on that path.

---

## Merge verdict

**Ship after two named one-line edits.**

Four passes and a guard audit have now walked every deletion, every new assertion, and every test change in this commit, twice over from independent starting points. The result is convergent and clean: **no assertion in `cd6b0005` can fire on a valid graph**, no deleted branch was load-bearing, no test assertion was weakened, and the numeric device tests (`tests/inductor/test_work_division_hint.py:640-707`) are byte-identical to their pre-commit form. Both auditors independently re-derived the unreachability argument for the two asserts flagged as high-risk (`lx_relayout.py:295`, `allocator.py:864`) and both settle. The commit is also *better* than the earlier passes credited it: the reordering at `lx_relayout.py:279-292` moves the source-view projection check *above* the `if view == source_view: continue`, which closes a real path to an uncaught `ValueError` out of codegen â `materialize_lx_relayouts` stamps `lx_view` on the shared source buffer (`lx_relayout.py:343`) and `SpyreKernel.create_tensor_arg` projects it for *every* reader (`spyre_kernel.py:771-775`) with no try/except on that path. That is the most substantive correctness change here and no pass flagged it until the challenge round.

The two edits below are not correctness fixes â nothing is broken today. They are altitude fixes: two places where a *skip* was converted into a *compile abort* in passes whose written contract is to never fail a compile. In a codebase where a wrong ownership map is silent wrong numbers and a dropped edge is merely slower, the asymmetry matters enough to spend two lines on.

## Blockers

**1. `torch_spyre/_inductor/lx_relayout.py:295` â an assert that makes the next line dead code.**

```
293  is_matmul = _is_matmul_op(consumer)
294  if is_matmul:
295      assert len(deps) == 2, f"matmul {consumer_name} has {len(deps)} reads"
296  if (not is_matmul and not isinstance(consumer.data, Pointwise)) or (
297      is_matmul and read_index not in (0, 1)
298  ):
299      break
```

`read_index` is the `enumerate` index into that same `deps` list (built at `lx_relayout.py:215-217`, rebuilt identically at `:265-267` from the memoized `op_read_writes`). So `len(deps) == 2` makes the `read_index not in (0, 1)` arm at `:297` provably unreachable. The file now states two contradictory beliefs about matmul arity, and the assert silently swallowed a live guard.

Failing scenario: not reachable today â I closed both candidate paths. FP8 carries no third read (`lower_scaled_mm`, `lowering.py:333-412`, builds `Reduction.create(reduction_type=BATCH_MATMUL_FP8_OP, input_node=[mat1, mat2])`; scales are separate ops from `scaled_mm_decomp`, `decompositions.py:789-824`). The genuine 3-MemoryDep matmul â the tiled accumulator at `propagate_layouts.py:1694-1698` â always carries `MutationLayoutSHOULDREMOVE` and `break`s 32 lines earlier at `lx_relayout.py:261`. The scenario is *future*: any lowering that emits a 3-MemoryDep matmul without a mutation layout aborts the entire compile instead of skipping one relayout candidate. torch-spyre's own matmul pass treats this as skippable (`padding.py:194` is a `continue`).

Fix: `if is_matmul and len(deps) != 2: break`, and delete the now-dead arm at `:297`.

**2. `torch_spyre/_inductor/scheduler.py:490-491` â two rejection arms of the demote backstop became crashes.**

```
488  source_view = _lx_view(plan.source_name)
489  destination_view = _lx_view(dep.name)
490  assert source_view is not None and destination_view is not None
491  assert source_view != destination_view
```

`demote_incoherent_lx_buffers` exists precisely to absorb state planning could not foresee â its own docstring (`scheduler.py:433-447`) says "the only safe answer is to give up LX residency" and "Deliberately verification-only". Both lines previously fed `invalid_sources[...] â demote() â HBM fallback`. Line `491` is strictly redundant with `lx_relayout.py:338` (`assert plan.source_view != plan.destination_view`), asserted at the point the plan is committed on a frozen dataclass nothing mutates in between.

Failing scenario: also not reachable today (`allocation["lx"]` and `lx_view` are written together only at `lx_relayout.py:341-344`, cleared together only by `clear_lx_state` at `:120-122`; no `demote()` runs before the loop at `:472-495` completes). The scenario is a future pass that pops `"lx"` from one side of a relayout pair â which converts a correct HBM demotion into a compile crash, in the one function whose entire job is to not do that.

Fix: delete `:491`; fold `source_view is None or destination_view is None` back into the `if` at `:492` so it lands in `invalid_sources`.

## The assert question

Asserts are a safe carrier **in this repo**, verified rather than assumed: zero hits for `PYTHONOPTIMIZE`, zero for `python -O`/`-OO`/`compileall`/`optimize=` across `setup.py`, `Makefile`, `pyproject.toml`, `tox.ini`, and `.github/workflows`; no Dockerfile in the tree at `cd6b0005`; `pyproject` uses plain `setuptools.build_meta` and a wheel ships `.py` sources, so optimization level is a property of the interpreter at run time, not of the artifact. The codebase already carries ~153 asserts under `torch_spyre/`, including pre-existing correctness gates like `lx_relayout.py:337`. So this commit adds no new exposure. Most of the new asserts are self-enforcing even if stripped â `int(value)` raises `TypeError`, missing dict keys raise `KeyError`, `superdsc.py:995` raises `ValueError`. **No load-bearing check needs to become an explicit raise.** The one candidate proposed in an earlier pass â `superdsc.py:979`, where a stripped assert lets a stale `work_division` reach `arg.work_division = effective` at `:999` â does not survive scrutiny: the *same* recomputed `is_lx_relayout_identity` predicate also selects `opfunc = "shuffle"` at `superdsc.py:1288-1291`, so a predicate flip is wrong numbers *before and independently of* the assert. Converting it to a raise fixes nothing. The real fix there, if anyone wants it, is to thread the decision already made at `spyre_kernel.py:953` onto the `OpSpec` so `parse_op_spec` reads it instead of recomputing â and that fixes `opfunc` too. Follow-up, not merge-blocking.

## Follow-ups

- **Write the ordering fact down.** At `allocator.py:1014`: `materialize_lx_relayouts` must remain the last statement of `_push_allocation`. That single fact is what makes the `SolveError` retry (`allocator.py:2394-2404`), the three `assert not materialized_lx_relayouts(graph)` (`lx_relayout.py:207, 323, 328`), and the newly address-only `_clear_lx_relayout_groups` (`allocator.py:924`) all correct at once â and it is currently written nowhere.
- **Comment the flag at `patches.py:114`.** Name what depends on `_spyre_pre_scheduling_complete`: the three collect-time asserts, and the fact that `_append_lx_relayout_destinations` is non-idempotent (`allocator.py:883-884` rewrites every use to `2 * use + 1`, so a raw `op_index` can never re-appear in `source.uses`).
- **Confirm the pre-scheduling semantics with whoever found the double-`_update_scheduler`.** The flag silences all ~23 passes in `CustomPreSchedulingPasses` (`passes.py:406-465`) on a repeat call, not just LX planning. Running once is almost certainly right; it should be an explicit decision, not a side effect of an LX cleanup. Note also that `__call__` early-returns on a graph with no spyre-device ops, yet the flag is set unconditionally.
- **Assert the source-side half-tick surgery.** `allocator.py:891-893` does `source.uses = sorted({use for use in source.uses if use != consumer_tick} | {transfer_tick})`. The `| {transfer_tick}` term is what keeps the source alive across the shuffle; drop it and the solver may hand the source's address away before the shuffle reads it â silent wrong numbers, no test in the tree would notice. One line: `assert by_name["complete"].uses == [0, 2]`.
- Untested branches, each ~4-6 lines in a test that already has the scaffolding: the source/destination overlap rejection (`allocator.py:913-919`); the barred-source arm (`allocator.py:865-871`, needs `residency_reason="barred"`); the "invalid relayout copy" demotion (`scheduler.py:492-493`, needs a copy node with a second read).
- `_single_write` (`lx_relayout.py:184-191`) is half assert, half graceful `None`. Collapse to `if len(writes) != 1 or writes[0].is_indirect(): return None` â the signature already advertises `| None` and the caller at `:222-227` already branches on it.
- `_lx_view` (`scheduler.py:330-337`) â restore `if buffer is None: return None` and make the `isinstance` a `return None` guard. Same reasoning as blocker 2; this pass should not be able to fail a compile. Its test currently mocks `FixedTiledLayout` to `SimpleNamespace` (`test_work_division_hint.py:865`), so the isinstance assert is a tautology under test.
- Merge-note accuracy: the `_op_short_name` hoist is *not* an identical-duplicate collapse. The allocator's deleted copy used `or`-chaining and returned the raw object; the survivor (`pass_utils.py:113`) uses truthiness. They diverge on a falsy-but-present attribute, and `_get_op_name` feeds LX-eligibility decisions at `allocator.py:289, 561, 2036`.

## What this review found overall

Across four passes this review found three real silent-wrong-numbers defects in this code â a sub-dimension endianness inversion, a stale `work_division` override, and a slot-0 ownership default â and this final commit correctly retires the last of those at both `lx_relayout` sites by replacing `.get(dim, 0)` with a hard index, backed by a genuine total-ness argument (`pass_utils.py:1670-1671` writes `work_slice_dims[dev_dim]` and `sym_to_device_dim[sym]` in the same loop iteration). The review was also wrong in specific, correctable ways: `superdsc.py:979` was misdiagnosed (the recompute is pre-existing and its real consequence is a wrong `opfunc`, not the assert), `superdsc.py:990`'s causality was inverted (the deleted `dim_order` assert guarded a projection this commit *removed*; net, the silent-default surface shrank), the "walked every new assertion" claim missed `assert device_dim < len(device_coordinates)` at `lx_relayout.py:86` â the one assert that replaced a `ValueError` three callers actively catch â and two correct conclusions were reached through wrong evidence chains (a `coarse_tile.py` citation for matmul arity that names the wrong population of ops, and a "solvers return `list(buffers)` in full" argument applied to a function that never sees a solver). What survives all of it is a commit that is a genuine net reduction, is safe, and whose only remaining defects are two places where a `break` was written as an `assert`.
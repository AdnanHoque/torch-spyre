# Z-order tile traversal — Phase 0 Step 1 verification

## Question

Can torch_spyre's iteration_space lever express Z-order (or other
non-row-major) tile traversal for matmul, without modifying deeptools
kernel templates?

## TL;DR

**No.** The iteration_space lever can express block STRUCTURE but
not block TRAVERSAL ORDER. The kernel template hardcodes loop nest
order to specific dim names; reordering dims renames them and
breaks the template's semantic. The genuinely valuable Z-order
optimization (within-core tile reuse for scratchpad locality)
requires deeptools-side kernel template work.

A tractable secondary question for torch_spyre — block-to-core
assignment in Z-order — is mechanically possible but its value
question was already answered by our earlier placement probes:
**core_id placement on the Data Ring doesn't affect wall time**.
So even the secondary angle delivers nothing.

## Verification

### What the SDSC IR exposes

For a standard matmul, the SDSC JSON has:

```json
{
  "matmul": {
    "numWkSlicesPerDim_": {"mb": 32, "out": 1, "in": 1},
    "coreIdToWkSlice_": { ... },
    "dscs_": [{
      "matmul": {
        "N_": {"mb_": 128, "out_": 1024, "in_": 8192},
        "primaryDsInfo_": {
          "INPUT":  {"layoutDimOrder_": ["mb", "in"]},
          "KERNEL": {"layoutDimOrder_": ["in", "out"]},
          "OUTPUT": {"layoutDimOrder_": ["mb", "out"]}
        }
      }
    }]
  }
}
```

What torch_spyre controls:
- `N_` per-dim sizes (already done via `_force_split` patcher)
- `numWkSlicesPerDim_` (planner heuristics, e.g., k_fast)
- `coreIdToWkSlice_` (k_fast permutation)
- `layoutDimOrder_` (per-tensor data layout)

What torch_spyre does NOT control:
- The kernel template's loop nest order
- The order in which a single core walks its assigned tiles

### The matmul dim labels are positional and template-bound

In `torch_spyre/_inductor/constants.py`:
```python
MATMUL_DIM_LABELS = ["y", "x", "mb", "out", "in"]
```

And in `superdsc.py:parse_op_spec`:
```python
symbol_mapping = {
    sym: Symbol(dim_labels[i]) for i, sym in enumerate(op_spec.iteration_space)
}
```

For a 3D matmul, position 0 → `mb`, position 1 → `out`, position 2
→ `in`. The kernel template (.smc files in deeptools) hardcodes
loop directives to these names: `LX_MVLOOPCNT imm=Dmb_Cmb`, etc.

**Reordering iteration_space dict keys** → renames dims (position 0
becomes `mb`) → kernel template uses WRONG semantic for the
reordered dims → produces wrong output.

So we can't manipulate iteration order via dict reordering.

### Iteration space CAN gain extra dims via tensor reshape

By expressing a matmul as a batched matmul, we get extra outer dims:

```
2D matmul (M=128, N=1024, K=8192):
  iteration_space = {mb: 128, out: 1024, in: 8192}

3D bmm (M_blocks=8, M_per=16, N=1024, K=8192) with B replicated:
  iteration_space = {x: 8, mb: 16, out: 1024, in: 8192}

5D matmul (with two batch dims):
  iteration_space = {y: 2, x: 4, mb: 32, out: 256, in: 8192}
```

Verified all three compile and run correctly on AIU with 4D and 5D
producing batched-matmul SDSC.

**But there's a catch**: in higher-dim cases, the K dim's matrix B
gets REPLICATED across the batch dims (one copy per batch). For a
true matmul with shared B, attempting `4D @ 2D = (4D-broadcast)`
fails: `InductorError: batch1 must be a 3D tensor`. Inductor's
matmul implementation only handles 2D × 2D and 3D × 3D, not higher
dim broadcasting.

So we can express block structure but only by paying replication
cost on the shared operand.

### What this means for traversal order

Even with block structure expressed, the kernel template iterates
the resulting iteration space in its hardcoded order (probably
`x → mb → out → in` row-major or some fixed pattern). To get Z-order
within the iteration space, the template's loop nest itself would
need to be Z-order.

**Z-order traversal is in deeptools, not torch_spyre.**

## Two layers, two levers

The Z-order optimization can be applied at two layers:

### Layer A: block-to-core assignment (across cores)

Which core processes which (M_block, N_block) tile. This IS
torch_spyre-accessible via `coreIdToWkSlice_` patching (the same
machinery k_fast uses).

**But our prior placement probes already answer this**: core_id
placement on the Data Ring (which serves block fetches) is
placement-invariant. The 96-shape multicast permutation probe
showed 0.6% median spread across permutations. So Z-order at this
layer **delivers nothing on AIU**.

### Layer B: tile traversal within a core (across iterations)

Which tile a core walks first, second, etc. for scratchpad locality.
This is in the kernel template's loop nest. NOT torch_spyre.

**This is where the value would be**: Z-order traversal keeps
spatially-nearby tiles in cache, reducing scratchpad eviction. The
HipKittens GPU work and the NeuronMM Trainium work both demonstrate
this is the valuable layer. AIU's two-level (LX + L0) scratchpad
hierarchy with no hardware cache makes it especially relevant.

## Implications for the project

The proposal's pitch was a torch_spyre-side project. **It isn't.**
Layer A is in torch_spyre but doesn't deliver value (placement
proven irrelevant). Layer B is in deeptools and would require
kernel template modifications.

If the project is pursued, it must be a **deeptools-side project**,
either:
1. Modify kernel template loop nest order (.smc file changes) to
   emit Z-order traversal under a config flag.
2. Add a new kernel template variant `batchmatmul_fp16_zorder_fwd.smc`
   that does Z-order traversal, with torch_spyre dispatching to it
   when appropriate.

Both require deeptools partnership and 4-6 month timelines as the
proposal estimated. **Solo torch_spyre work is not a viable path.**

## Recommendation

The Phase 0 verification confirms the proposal's "4-6 months" timeline
estimate is right but ALSO confirms it's deeptools-side, not torch_spyre.

Three options:

1. **Pursue with deeptools partnership.** Coordinate with the
   scratchpad-optimization team owning the "1H 2026 priority #3"
   roadmap item. Find out if traversal-aware allocation is in scope.
   If yes, propose a torch_spyre-side dispatch hint (e.g., "shape (M,
   N) blocked into Z-order tiles" → triggers a new kernel template).
2. **Defer.** The CGO 2026 paper #2 ("Loop Absorption / Loop Index
   Sequencing") may address this independently. Wait for that paper
   to land and decide whether to extend or compete.
3. **Pivot to a different project.** Several solo torch_spyre
   candidates remain unblocked: 6-tensor fusion cap (issue #827),
   LX residency planner, SDPA-to-bmm regression fix. Any of these
   delivers solo work without needing deeptools partnership.

Honest read: the value of Z-order on AIU is real but the layer
where it lives makes it organizationally larger than the proposal
implied. Worth pursuing if the deeptools team is open to it; not
worth pursuing solo.

## Files

- This doc — Phase 0 step 1 findings

# Independent reproduction, and the fix that unblocked P07

Date: 2026-07-27

All work below ran in fresh workspaces (`/home/adnan/claude-isolated/` and
`/home/adnan-cdx/claude-isolated/`). The original run directories were read but
never written, so their provenance is intact.

## 1. The best result reproduces

Rebuilt from scratch — clone at `59545440`, apply
`patches/torch-spyre/p06_completion_p08_bridge_BEST.patch`, run on a different
pod (`adnan-clc-spyre-dev-pf`) than the original.

| Gate | Original | Reproduction |
| --- | ---: | ---: |
| Median device time | 247.407 ms | **246.322 ms** |
| Correctness | token 203 | token 203, 6/6 requests |
| Degenerate kernels | 0 | 0 |

Per-request device sums: `246.322, 246.118, 246.944, 245.916, 246.497` ms.
Kernel structure verified as 42 kernels/request (1 input + 40 blocks + 1 final
stage) on all five.

The 1.1 ms difference is pod-to-pod variation, not an improvement. **247 ms
remains the honest headline number**; this run only shows it is stable and
rebuildable from the archived patch alone.

The same nine relayout shuffles were emitted (`7, 9, 18, 38, 42, 45, 53, 56,
59`), and the P08 edge matched its documented topology exactly: `45_shuffle` =
16 source cores → 32 destination cores, 1.00 MiB local / 3.00 MiB remote.

### A locality lead, visible in the same data

Per-shuffle traffic for the accepted stack:

| sdsc | transfers | local | remote | local MiB | remote MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42_shuffle | 128 | 32 | 96 | 4.00 | 12.00 |
| 53_shuffle | 128 | 32 | 96 | 4.00 | 12.00 |
| 56_shuffle | 128 | 32 | 96 | 4.00 | 12.00 |
| **59_shuffle** | **64** | **2** | **62** | **0.78** | **24.22** |
| 45_shuffle (P08) | 64 | 16 | 48 | 1.00 | 3.00 |
| 18_shuffle | 128 | 32 | 96 | 1.00 | 3.00 |
| 38_shuffle | 128 | 32 | 96 | 1.00 | 3.00 |
| 9_shuffle | 896 | 28 | 868 | 0.01 | 0.21 |
| 7_shuffle | 32 | 1 | 31 | 0.00 | 0.01 |

`59_shuffle` is the outlier worth attention: it moves 24.22 MiB remote against
0.78 MiB local — a 31:1 ratio, the worst of the nine, and unlike `7_shuffle`
(the known 1-local/31-remote pattern) it carries real volume. Every other
large shuffle sits at 3:1. Nothing here proves it is fixable, but it is the
largest unexplained locality gap in the accepted stack.

## 2. P07 was not merely unfinished — it did not compile

The last status on P07 was that a matcher fix had landed, a targeted unit suite
was green, and a one-layer device compile was being rerun. That rerun fails.

Reproduced byte-identically in a clean workspace:

```
File ".../torch_spyre/_inductor/codegen/superdsc.py", line 864, in parse_op_spec
    sym: Symbol(dim_labels[i]) for i, sym in enumerate(op_spec.iteration_space)
IndexError: list index out of range
```

### Root cause

`parse_op_spec` builds one label per iteration-space dimension:

```python
dim_labels = (
    _get_op_dim_labels(ndim, is_matmul)
    if op_spec.dim_labels_override is None
    else op_spec.dim_labels_override      # <- frozen, never re-checked
)
symbol_mapping = {
    sym: Symbol(dim_labels[i]) for i, sym in enumerate(op_spec.iteration_space)
}
```

When no override is set, labels are generated from `ndim`, so they always match.
`_split_p07_rope_shuffle` sets an override — and it sets it *correctly*, dropping
the rotary-row symbol from both the iteration space and the label list together,
3 and 3.

The problem is what happens afterwards. Instrumenting the failure gives:

```
op='shuffle' ndim=4 n_labels=3
labels=['mb','i','out']  iter_space=['d0','d3','d4','z0']
```

A later pass adds a band symbol `z0` to the iteration space — the split's own
8-way token split being re-materialised. The override was captured before that
pass ran and nobody extends it, so the fourth dimension has no label and the
list index walks off the end.

This is a latent sharp edge in `parse_op_spec`, not really a P07 bug: *any*
override that predates work division has it. P07 is simply the first caller to
set one.

### The fix

Extend a short override with labels it does not already use, instead of
indexing past the end (`patches/torch-spyre/p07_label_extension_fix.patch`,
also applied to the branch source). It is inert unless the override is short,
which today only happens on the P07 path.

### Result

P07 now compiles and runs, and produces the topology it was built for — four
independent row shuffles rather than two combined ones:

| sdsc | transfers | local | remote | src cores | dst cores |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12_shuffle | 64 | 16 | 48 | 16 | 32 |
| 13_shuffle | 64 | 16 | 48 | 16 | 32 |
| 18_shuffle | 64 | 16 | 48 | 16 | 32 |
| 19_shuffle | 64 | 16 | 48 | 16 | 32 |

Four gathers = two rotary output rows × two consumers (Q and K), which is the
SenDNN topology this edge was replaying.

Correctness: the run generates token `44`. That is the **correct** value here —
`p09_control_a`, the unmodified one-layer control, also generates `44`. Token
`203` is the full-40 gate and does not apply to a one-layer run.

**What this does not yet show:** one layer, in isolation, with every other
oracle off. It is not integrated into the accepted stack and has no full-40
run and no timing. Treat it as "the blocker is removed", not "P07 is done".

## 3. P09's full-model run had already passed, unreported

`p09_full40_a` in the cdx lane generates token **203** on a full 40-layer run,
with 5 relayout plans. The delegated rerun succeeded; its result was simply
never read back.

One caveat, stated because it matters for the acceptance contract: that run
captured **no** `STCDP_FINAL_TRANSFER` output, so the LX-only transport gate is
**unverified** for it — not failed, unverified. Closing it needs one rerun with
`STCDP_DUMP_TRANSFERS=1`. Correctness and transport are separate gates and only
correctness is currently evidenced at full scale.

## Reproducing any of this

`scripts/run_repro.sh` rebuilds the 247 ms measurement;
`scripts/run_p07p09.sh` drives the P07/P09 arms. `scripts/analyze_repro.py`
computes the device median from a Kineto trace and `scripts/stcdp_check.py`
prints the per-shuffle local/remote traffic table.

The P07/P09 lane needs the cdx pod specifically: its Python lives in that pod's
own `/tmp`, and its home is not on the shared PVC.

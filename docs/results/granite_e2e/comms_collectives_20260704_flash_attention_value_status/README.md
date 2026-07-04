# Flash Attention Collective Relayout Value Status

Date: 2026-07-04

This artifact records the current status of the DLDSC grouped all-gather / matmul operand broadcast exploration for `test_flash.py`.

## Bottom line

We have unblocked DXP/runtime smoke for the full flash attention script, but the collective relayout path is not value-correct yet.

Three backend paths were checked:

| path | run | result | interpretation |
| --- | --- | --- | --- |
| Manual diagnostic per-stick ring lowering | `value_correct_after_relayout_on_20260704_044752` | Fails CPU comparison: 31.5% mismatched, max abs diff 3.841796875 | Physical movement is incomplete/wrong even though DXP/runtime can execute. |
| Built-in backend path with guard left on | `flash_builtin_ifn_no_manual_20260704_091946` | Fails in DXP with intentional guard | Deeptools currently blocks this case because staged RHS chunks are not safely bound into the matmul transfer loop. |
| Built-in backend path with unsafe guard override | `flash_builtin_ifn_unsafe_value_20260704_092645` | Fails CPU comparison: 96.9% mismatched, max abs diff `inf` | The guard is justified; simply forcing the path through is not correct. |

The compile-only smoke case for the unsafe path also passed:

| path | run | result |
| --- | --- | --- |
| Built-in backend path with unsafe guard override and CPU check skipped | `flash_builtin_ifn_unsafe_20260704_092108` | `SUCCESS`, but this proves only compile/runtime smoke. |

## Working diagnosis

The Torch-side DLDSC contract is reaching Deeptools: the matmul RHS operand is classified as a grouped all-gather / replicate pattern. The remaining gap is physical realization.

The current manual lowering creates synthetic per-stick ring transfers with scalar LX addresses. That bypasses the normal transfer-loop address math, so it likely moves repeated flat chunks rather than the coordinate-specific RHS sticks required by each consumer matmul slice.

The guarded Deeptools path says the same thing more directly:

```text
matmul_operand_broadcast metadata was classified, but physical lowering is blocked:
current DL matmul lowering still consumes one resident LDS operand pointer and
cannot bind staged RHS chunks from IFN/STCDP.
```

So the next backend task is not more metadata; it is to bind the staged collective operand into the matmul transfer loop with correct coordinate-derived source/destination addressing.

## Related smaller probe

`flash_no_scalar_value_backend1_20260704_091125` failed earlier in DXP with:

```text
out_reuse_dim.size() == 1
```

Inspection showed this assertion is triggered by non-relayout batchmatmul SDSCs in the smaller `test_flash_no_scalar.py` graph, not by the relayout-classified matmuls. It appears to be a separate L3 scheduler/layout assumption exposed by that smaller probe shape.

## Follow-up checks

After the initial artifact snapshot, we gated the synthetic per-stick ring metadata so it only runs when `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1`.

Deeptools diagnostic commit:

```text
352919bf3f9c0efb2430568c667111aeb0a99e95 [diagnostic] gate synthetic matmul broadcast ring lowering
```

Then we reran the built-in unsafe path without the manual ring metadata:

| path | run | result | interpretation |
| --- | --- | --- | --- |
| Built-in unsafe IFN/STCDP path, manual ring disabled | `flash_builtin_ifn_unsafe_value_noring_20260704_093946` | Fails CPU comparison: 99.2% mismatched, max abs diff `inf` | The built-in path still does not bind the staged RHS operand correctly. |
| Baseline `test_flash.py`, relayout disabled | `flash_baseline_relayout_off_value_20260704_094500` | Fails CPU comparison: 75.1% mismatched, max abs diff `inf` | The full `test_flash.py` CPU comparison is not a reliable correctness oracle in this environment. |

We also built several smaller value probes. A simple rank-4 batched matmul passes, and a direct RHS-broadcast matmul with `work_div={"H":4,"M":8}` passes, but that direct case does not create an LX-resident producer edge and emits no `matmul_operand_broadcast` plan. Probes that force a separate RHS producer with `exp` or `neg` fail even with relayout disabled, so they are not clean correctness harnesses.

Current reliable statement:

```text
The DLDSC metadata and DXP/runtime smoke path for flash attention are unblocked,
but value correctness for grouped all-gather / matmul operand broadcast remains
unproven. The next requirement is a clean value harness with a known-good
LX-resident producer edge, or a backend implementation that binds staged
collective RHS chunks into the matmul transfer loop and is then checked against
that harness.
```

## Files

- `manual_ring_value_stderr_tail.txt`: tail of the value failure for the manual diagnostic ring lowering.
- `builtin_guard_stderr_tail.txt`: guard failure for the built-in backend path.
- `builtin_unsafe_smoke_stdout_tail.txt`: compile/runtime smoke success for the unsafe built-in path.
- `builtin_unsafe_value_stderr_tail.txt`: value failure for the unsafe built-in path.
- `manual_ring_value_stderr.log`, `builtin_guard_stderr.log`, `builtin_unsafe_value_stderr.log`: full logs for the main failure cases.

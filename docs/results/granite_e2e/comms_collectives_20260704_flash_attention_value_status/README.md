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

## Files

- `manual_ring_value_stderr_tail.txt`: tail of the value failure for the manual diagnostic ring lowering.
- `builtin_guard_stderr_tail.txt`: guard failure for the built-in backend path.
- `builtin_unsafe_smoke_stdout_tail.txt`: compile/runtime smoke success for the unsafe built-in path.
- `builtin_unsafe_value_stderr_tail.txt`: value failure for the unsafe built-in path.
- `manual_ring_value_stderr.log`, `builtin_guard_stderr.log`, `builtin_unsafe_value_stderr.log`: full logs for the main failure cases.

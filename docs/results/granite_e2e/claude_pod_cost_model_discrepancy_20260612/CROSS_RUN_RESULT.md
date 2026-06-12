# Cross-run result: the discrepancy is DeepTools codegen

Ran Codex's generated programs on the Claude device via `replay_foreign.py`
(launch_kernel replay, no recompile, 20 reps device self-time).

| program (same SDSC) | on Codex device | on **Claude device** | verdict |
|---|--:|--:|---|
| Claude's QK^T 1_4_8_1 | — | 1635 us | (baseline) |
| **Codex's QK^T 1_4_8_1** | 735 us | **740.6 us** | runs FAST on Claude HW |
| Codex's control 2_8_2_1 | 60 us | 59.8 us | sanity ok |

SDSC byte-identical (`9e8385cfec7f8707`), but `init.txt` differs
(Codex `e809da81…` vs Claude `28f897a2…`).

## Conclusion
The **same** Claude device/runtime/firmware runs Claude's program at 1635 us and
Codex's program at **740 us**, from the **identical SDSC**. So the 2.2x is entirely
in the **generated program**, i.e. **DeepTools post-SDSC codegen** — not
runtime/firmware/device (exonerated), not torch-spyre lowering (SDSC identical),
not DT_OPT (ablated to zero both pods), not flex (ablated on Claude side). The
Claude-pod harvest DeepTools (+932) emits a ~2.2x slower program for the
inefficient split than Codex's `/opt/ibm/spyre/deeptools`; efficient splits (the
control) compile to equally-fast programs on both (so the codegen gap is
split-specific, which is why it masquerades as a cost-model difference).

## Implication
The Claude-pod 240-split oracle (and the +13%/+20% cost-model gaps derived from it)
is **distorted by DeepTools codegen quality**: "bad" splits look 2-3x worse than
they fundamentally are. On a DeepTools build that schedules every split near-ideal
(Codex's), the work-division split choice is a small lever — consistent with the
earlier attribution that the Granite gap is backend/residency, not work-division.
The actionable fix for the Claude pod is the DeepTools build, not the cost model.

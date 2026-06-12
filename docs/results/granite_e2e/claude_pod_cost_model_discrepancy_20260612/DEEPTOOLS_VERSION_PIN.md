# DeepTools version pin: the codegen delta is +549 vs +932

Cross-run proved the cross-pod gap is DeepTools post-SDSC codegen. Exact builds:

| build (ibm-deeptools, 2.0.0-0.main.1+...) | source | init.txt (QK^T 1_4_8_1) | my-HW timing |
|---|---|---|--:|
| **+489.251492d** | my pod `/opt/ibm/spyre` | `c39168ed` | 1620 us (slow) |
| **+549.7325fd8** | Codex pod `/opt/ibm/spyre` | `e809da81` | **740 us (FAST)** |
| **+932.a1ec02a** | my harvest (the stack I sweep on) | `28f897a2` | 1635 us (slow) |

Same `main` branch. **My stack is +932 — NEWER than Codex's fast +549 — yet ~2.2x
slower codegen.** So this is a **DeepTools codegen REGRESSION in the +549 -> +932
window** on main, on the inefficient matmul splits, not me being behind.

- My +932 is fully-consistent (harvest stack, dxp_standalone via PATH) -> solid.
- My +489 test paired +489 deeptools with harvest senlib +148 (version mismatch),
  so "+489 also slow" is suggestive, not clean. The fast program is reproducible
  ONLY from Codex's +549.7325fd8 (verified: his program runs 740us on my HW).

## Consequences
- The Claude-pod 240-split oracle runs on +932 -> the "bad split" penalties are
  inflated by the codegen regression. Re-running it on +932 keeps mixing
  cost-model behavior with the codegen regression.
- To re-measure the work-division lever cleanly, the oracle must run on
  +549.7325fd8 (obtain that RPM and LD/PATH it, or have the Codex pod run the
  full 240-sweep). torch-spyre invokes codegen via `subprocess dxp_standalone`
  (async_compile.py:63), so swapping the dxp_standalone on PATH is sufficient.
- Report to deeptools: ~2.2x codegen regression on small/inefficient matmul
  splits between +549.7325fd8 and +932.a1ec02a (same SDSC input).

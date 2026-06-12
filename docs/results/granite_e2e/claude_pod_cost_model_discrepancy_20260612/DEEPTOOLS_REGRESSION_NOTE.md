# NOTE: DeepTools codegen regression (+549.7325fd8 → +932.a1ec02a)

**~2.2x device-time regression in DeepTools post-SDSC codegen for small/inefficient
matmul splits, on `main`, between build +549 and +932.**

## Exact builds
- GOOD: `ibm-deeptools-2.0.0-0.main.1+549.7325fd8_0.el10` (commit 7325fd8)
- BAD:  `ibm-deeptools-2.0.0-0.main.1+932.a1ec02a_0.el10` (commit a1ec02a) — newer

## Repro (one split, fully isolated)
- Op: bmm `[512,32,128] x [512,128,512]` (Granite QK^T prefill), forced split `1_4_8_1`.
- The torch-inductor SDSC input is **byte-identical** across both builds
  (`sdsc_0.json` sha256 `9e8385cfec7f8707…`).
- `dxp_standalone --bundle` on that SDSC emits a **different program** per build
  (`init.txt`: +549 `e809da81…`, +932 `28f897a2…`).
- Replaying each generated program on the **same device/runtime/firmware** (via
  `launch_kernel`, 20-rep device self-time):
  - +549 program → **740 µs**
  - +932 program → **1635 µs**  (2.2x)
  - control split `2_8_2_1` → 60 µs on both (so it is split-specific, not global).

## Why it matters
The regression inflates the apparent cost of "inefficient" work-division splits,
which masquerades as a work-division/cost-model problem. On +549 the split choice
barely matters (every split schedules near-ideal); on +932 bad splits pay ~2-3x.

## Evidence in this tree
- `programs/` (Claude +932) and `../codex_pod_program_export_20260612/` (Codex +549):
  identical SDSC, different `init.txt`.
- `replay_foreign.py` + `CROSS_RUN_RESULT.md`: timing follows the *program*, not the
  device. `DEEPTOOLS_VERSION_PIN.md`: the +489/+549/+932 three-build comparison.

## Bisect hint
+489.251492d (older than 549) is ALSO slow on this HW, so the good codegen is
specific to the +549 window — the optimization landed in (489,549] and regressed
in (549,932].  (The +489 datapoint used a mismatched senlib, so treat as suggestive.)

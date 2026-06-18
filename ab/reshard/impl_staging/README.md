# Substrate implementation — workflow-authored artifacts (STAGED, not yet wired)

Output of the `c2c-substrate-impl` workflow (2026-06-18). These are
**CPU-authored, not yet wired into the build and not yet device-validated**.
They build the genuine core-to-core ring-reshard + warp-spec substrate for the
SwiGLU `mul -> down-proj` reduction-input edge (co-assignment is dropped — the
prize is the ring/MPMD/warp-spec substrate). See `INTEGRATION.md` + `RUNBOOK.txt`.

## Decisive findings
- **EBR carrier (opcode-settled):** the genuine ring legs (`L3_STU`=248/`L3_LDU`=248,
  symmetric, EAR-routed) are CORRECT. The broken `3200*core` EBR is on 40 minority
  `L3_STMU` (RINGDTHBMU = ring-store-WITH-HBM-writeback) instructions — an HBM-mirror
  leg, not the ring landing.
- **flash-ws does NOT solve it:** its cross-core handoff is planner-only/fail-closed;
  every device-realized flash-ws bridge is 1-D pure-LX (`hbmSize_:0`), where
  `core==column-band` makes `3200*core` correct by construction. The gather doesn't
  dodge it (same `computeMulticastOptMetadata` chokepoint).
- **flash-ws warp-spec IS real and reusable:** `flash_pipeline_schedule` +
  `InputFetchNeighbor` overlap, `L3DlOpsScheduler` places the gather inside the DL
  op inner loop with soft inter-engine sync (PT array || L3 ring || LX). But it's a
  producer-side KV PREFETCH, not a reduction-input gather, and concurrency
  simultaneity is unverified (rests on median_ms + static engine tokens).

## Two open decisions (need the user)
1. **Foundation/branch:** the planner/realize grafts target flash-ws files
   (`onchip_handoff.py`, `onchip_realize.py`) that are NOT on `core-to-core` (off
   main). Land on flash-ws, merge flash-ws into core-to-core, or port a
   flash-ws-independent realizer?
2. **Fix carrier (gated by a cheap CPU probe):** does DCG honor a stamped `ebrInit_`
   (-> inductor-only fix) or recompute from `coreId` (-> deeptools fix at
   `perfDscToSdsc.cpp ~2099`, a multi-site change + dxp rebuild — PAUSE-for-approval)?
   The scoped `dsm.cpp:6764` one-liner is BLOCKED (the SDSC flattens
   `{mb:4,out:8}->{mb:32,out:1}`, losing the band at the carrier).

## Files
- `INTEGRATION.md` — the substrate integration doc.
- `RUNBOOK.txt` — the 8-step serial gate runbook (CPU offline -> probes -> [pause] deeptools build -> solo device).
- `authored_planner-realize.txt` — production reshard package + planner/realize grafts (target flash-ws spine).
- `authored_ebr-patch.txt` — EBR patch (status: BLOCKED at scoped site; real fix upstream).
- `authored_probes.txt` — sketched CPU probes (attribution + ebrInit stamp).
- `authored_warpspec.txt` — warp-spec pipelined-schedule builders.

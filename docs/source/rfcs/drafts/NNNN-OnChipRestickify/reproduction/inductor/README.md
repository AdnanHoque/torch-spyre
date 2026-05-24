# Inductor compiler-pass reproduction scripts

Scripts that exercise the **compiler-driven** on-chip handoff (the realize pass),
as opposed to the manual splice harness in `../splice/` + `../devval/`. See recipe
§6b ("The inductor compiler-pass realization") and `onchip_realization_design.md`.

| Script | What it does |
|---|---|
| `e2e_onchip.py` / `e2e_onchip.sh` | The compiler-driven E2E: `SPYRE_ONCHIP_HANDOFF_REALIZE=1` makes `torch.compile` itself emit the mixed bundle; the patched dxp (first on PATH) accepts it; device runs value-correct. Flag OFF = baseline negative control. Proven result: OFF -> no `datadscs_`; ON -> `opFuncsUsed_=['STCDPOpLx']`, `max_err 0.013672` (= baseline). |
| `run_block_baselines.sh` | Device baselines for the full-block workloads in `../workloads/` (transformer block, MoE FFN, MoE full block). Records compile failures honestly (transformer + MoE block do not compile on the current stack; MoE FFN ~125.85 ms). |

All paths are parameterized via `../env.sh` (`PYTHON`, `PATCHED_DXP`, `VAL_BOOT`,
`WORK_DIR`). Run device scripts **solo** (single shared accelerator).

## Where the inductor SOURCE lives

These scripts import `torch_spyre` from the **`tier0-tier1-onchip`** branch via the
`VAL_BOOT` shim (`../val-boot/sitecustomize.py` prepends `ONCHIP_SRC`). The realize
pass implementation is on that branch, not copied here (it is source, not a
reproduction script):

- `torch_spyre/_inductor/onchip_realize.py` — `detect_onchip_edge`, `apply_lx_flip`,
  `fold_onchip_handoff`, `realize_onchip_handoff`, `realize_streamed_handoff`
- `torch_spyre/_inductor/onchip_handoff.py` — the fail-closed planner
- `torch_spyre/_inductor/codegen/onchip_bridge.py` — the synthesizer (+ streaming
  `build_streamed_bridge`)
- `torch_spyre/_inductor/codegen/bundle.py` — `generate_bundle` realize integration
- `torch_spyre/_inductor/config.py` — `onchip_handoff_realize` flag
- `tests/_inductor/test_onchip_{realize,handoff,streaming,emit_matches_splice}_logic.py`

Point `ONCHIP_SRC` (in `../env.sh`) at a `tier0-tier1-onchip` checkout to run these.

# Codex Pod Granite Measurement Artifacts

These artifacts were generated on the Codex pod `adnan-cdx-spyre-dev-pf` on 2026-06-12.

Files:

- `codex_pod_device_timing_sweep_repro_summary_20260612.md`: forced-split sweep summary over the 12 Granite golden matmul shapes.
- `codex_pod_device_best_vs_picks_repro_20260612.csv`: reproduced device-best splits and reference picks.
- `codex_pod_current_devtree_picks_before_cost_tune_20260612.csv`: live emitted SDSC splits from the current development tree before the new cost-model tune.
- `codex_pod_current_devtree_picks_after_cost_tune_20260612.csv`: live emitted SDSC splits after the new cost-model tune.
- `codex_pod_cost_model_tune_summary_20260612.md`: before/after selected split timing summary and validation.
- `codex_pod_upstream_main_measurement_summary_20260612.md`: isolated upstream/main split measurement notes.

Important caveat: pristine upstream/main hit the PyTorch 2.12 fake-tensor / joint-graph setup failure before timing. The upstream-main summary therefore uses emitted SDSC splits from a probe-only no-joint workaround and imputes device time from the completed forced-split timing table.

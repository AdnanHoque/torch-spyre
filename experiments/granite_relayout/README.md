# Granite relayout parity experiment record

This directory is the self-contained archival checkpoint for the Granite 3.3
8B B1/S512 SenDNN parity investigation. Start with
`../../GRANITE_RELAYOUT_HANDOFF.md`, then use this directory for executable
helpers, exact source overlays, and evidence.

The checkpoint records work in progress; it is not a parity claim. No pull
request is associated with this branch.

## Directory map

- `scripts/`: pod/environment validation and run wrappers, including the exact
  isolated P02 command.
- `tools/`: every analysis/probe helper used during the study, plus logit and
  emitted-transport checkers.
- `runbooks/`: original SenDNN/Torch-Spyre reproduction and SMC runbooks.
- `patches/`: earlier Torch-Spyre prototype patches and the complete validated
  DeepTools all-to-all common-refinement handoff.
- `overlays/`: exact dirty DeepTools and FMS files paired with this checkpoint.
  Their base commits are in `provenance/source_state.json`.
- `artifacts/`: compact raw and derived evidence: full-model traces/logs,
  SenDNN post-LXOpt SDSCs, SMC summaries, relayout catalogs, P02 failed gates,
  correct reference logits, and P06 emitted-transport examples.
- `reporting/`: the two PDF builders and all generated chart inputs.
- `provenance/`: exact repository heads, worktree state, run index, and
  checksums.

## Reproduce the current P02 gate

On the recorded pod:

```bash
cd experiments/granite_relayout
scripts/run_p02_one_layer.sh my_unique_run_name
tools/inspect_relayout_run.py \
  /home/adnan/codex-isolated/device_parity_pr2939_20260725/runs/my_unique_run_name \
  --source buf29 --consumer buf31
tools/compare_logits.py \
  artifacts/p02_failed_gates/reference_logits \
  /home/adnan/codex-isolated/device_parity_pr2939_20260725/runs/my_unique_run_name/logits
```

Never reuse a run directory. DXP processing and compiler exports are not
idempotent.

## Current experiment state at archival time

- PR 2939 shuffle infrastructure and the paired DeepTools overlay are present.
- SenDNN P02 is cataloged as V -> AV, 8x4 source ownership to full 32-core
  replication, logical all-gather, 31 MiB remote traffic per layer.
- Torch discovers the same `buf29 -> buf31` plan.
- The allocator back-gap filter was implemented so only consumers rewritten to
  S2 are excluded; producer and unrelated S1 uses remain checked.
- The next graph-output clone attempt reached ReinterpretView reconstruction
  and stopped before device execution because this PyTorch's `ReinterpretView`
  dataclass requires keyword arguments. The archived failing `_d` and `_e`
  logs preserve both failure steps.
- Full-model parity remains unproven. Use the acceptance gates in the root
  handoff document.

## Large artifact policy

Useful compressed raw data is checked in (including the full Torch trace and
SenDNN post-LXOpt SDSC archive). Uncompressed copies, model weights, compiler
caches, and generated export trees are intentionally not duplicated. Their
original paths are recorded in `ARTIFACTS.md`; `tools/index_artifacts.py` can
refresh a complete machine-readable inventory.


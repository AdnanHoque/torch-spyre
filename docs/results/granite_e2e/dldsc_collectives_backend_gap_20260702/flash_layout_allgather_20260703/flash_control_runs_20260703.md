# Flash Control Runs - 2026-07-03

These controls were run after the one-edge `105_batchmatmul` isolation. They show that `test_flash.py` is not value-correct in this experimental checkout even before materializing the new layout-allgather relayout.

## collectives_disabled

- Run directory: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_collectives_disabled_control_20260703_030748`
- Backend plan count: `0`

```text
Mismatched elements: 12602433 / 16777216 (75.1%)
Greatest absolute difference: inf at index (0, 0, 1, 0) (up to 0.1 allowed)
Greatest relative difference: nan at index (0, 0, 1, 0) (up to 0.1 allowed)
```

## all_relayout_off

- Run directory: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_all_relayout_off_control_20260703_031210`
- Backend plan count: `0`

```text
Mismatched elements: 12602622 / 16777216 (75.1%)
Greatest absolute difference: inf at index (0, 0, 1, 1) (up to 0.1 allowed)
Greatest relative difference: nan at index (0, 0, 1, 1) (up to 0.1 allowed)
```

## Takeaway

The one-edge layout-allgather run is still useful because it confirms that a single active relayout worsens the already-bad mismatch (`90.7%` mismatch vs `75.1%` with all relayout flags off). However, this flash script/run configuration is not a clean pass/fail oracle for the all-gather primitive until we establish a value-correct no-relayout baseline on current main or a smaller validated attention probe.

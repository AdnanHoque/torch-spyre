# Codex Pod Program Export For Claude Replay

This directory contains the Codex-pod generated post-SDSC program artifacts for the same four splits used in the Claude/Codex discrepancy investigation.

Source run root: `/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/device_timing_repro_20260612_040740/all_splits`

Each `programs/<case>/` directory contains the emitted Deeptools/cache artifacts for that split:

- `bundle.mlir`
- `sdsc_0.json`
- `execute*/pagi.json`
- `*_dsg.txt`
- `loadprogram_to_device/*/init.txt`
- `segment_size.json`
- generated Torch-Inductor Python under `generated_python/`
- `SHA256SUMS` for that case

Codex-pod observed timing rows are in `codex_pod_timing_rows.tsv`.

The main replay target is `prefill_QKT_512x32x512x128_1_4_8_1`: same SDSC/bundle as Claude slow case, but this pod measured about 735 us. The control is `decode_attnatV_32x64x128x576_2_8_2_1`, which measured about 60 us on both pods despite post-SDSC program-byte differences.

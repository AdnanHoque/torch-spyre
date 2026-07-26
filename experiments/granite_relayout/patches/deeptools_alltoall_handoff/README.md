# DeepTools all-to-all common-refinement handoff

Validated 2026-07-25 against:

- Torch-Spyre PR #2939 head `31c703c70c03a71c57ea6d6a72b035803e850f96`
- DeepTools master `e3944781cb25b76abeb9b3e87c1f5c5879e84229`
- `DXP_LX_FRAC_AVAIL=0.2`

## Contents

- `deeptools_alltoall_common_refinement.patch`: minimal DeepTools source fix.
- `example_bundle/sdsc_0.json`: pre-DXP standard `SHUFFLE` SDSC.
- `example_bundle/bundle.mlir`: bundle driver required by `dxp_standalone`.
- `generate_example.py`: regenerates the example from PR #2939 source.
- `base_failure.txt`: failure from unpatched DeepTools master.
- `patched_success.txt`: successful patched DXP result.
- `broadcast_status.md`: separate broadcast investigation verdict.

## Example geometry

The example models the Granite MLP activation shape `[512, 12800]`:

- producer work division: 4x8;
- consumer work division: 32x1;
- 32 source and 32 destination cores;
- fan-in = fan-out = 8;
- 256 common-refinement transfers;
- equal source and destination per-core sizes.

Torch-Spyre emits this as the ordinary `op="shuffle"`; all-to-all is the
ownership geometry, not a separate op name.

## Fix size

The DeepTools fix is one C++ file and a 29-line diff: 28 insertions and one
deletion. It makes three focused corrections:

1. Preserve and reuse the destination allocation's coordinate folds for a
   direct two-LX-argument SHUFFLE.
2. Lower the residual compute row to `IDENTITY` after `STCDP-LX` performs the
   physical movement.
3. Clear the full stale producer coordinate description for non-direct true
   repartitions so DDC rebuilds it from the consumer schedule.

For comparison, the Torch-Spyre PR update is 38 inserted and eight replaced
production-code lines across two Python files, plus 153 lines of tests.

## Reproduction

From an unmodified DeepTools checkout at the recorded base:

```bash
export DEEPTOOLS_PATH=/path/to/deeptools
/path/to/dxp_standalone -d example_bundle
```

This reproduces `base_failure.txt`.

Apply and rebuild:

```bash
git apply deeptools_alltoall_common_refinement.patch
# Rebuild dxp and its dependent libraries using the normal DeepTools build.
export DEEPTOOLS_PATH=/path/to/deeptools
/path/to/rebuilt/dxp_standalone -d fresh-copy-of-example_bundle
```

Use a fresh bundle for every DXP run; DXP processing is not idempotent.

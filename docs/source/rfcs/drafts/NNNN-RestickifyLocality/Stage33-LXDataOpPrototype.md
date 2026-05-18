# Stage 33: LX Data-Op Restickify Prototype

## Summary

This stage adds a default-off diagnostic prototype for representing a
restickify-like movement as a Deeptools data-op DSC under `datadscs_`, rather
than as the production Torch-Spyre `ReStickifyOpHBM` compute-op SDSC.

The goal is deliberately narrow: prove the backend contract for
`STCDPOpLx`, `ReStickifyOpLx`, and `ReStickifyOpHBM` on small standalone SDSC
artifacts before changing normal Torch-Inductor lowering.

## What Changed

- Added `SPYRE_RESTICKIFY_LX_DATAOP=1`.
- Added `torch_spyre._inductor.codegen.restickify_lx_dataop`, which can emit a
  standalone SuperDsc containing one data-op DSC.
- Added `tools/restickify_lx_dataop_probe.py` to generate baseline and Stage
  3B-shaped data-op artifacts.
- Added focused unit coverage for the emitted JSON shape.

The normal compile path is unchanged. Production restickifies still lower as
`ReStickifyOpHBM` compute-op SDSCs unless a future stage explicitly wires this
data-op path into bundle generation.

## Prototype Modes

The probe synthesizes a square two-dimensional restickify-like movement. For
the backend-facing JSON it uses canonical Deeptools dimensions (`mb_` and
`out_`) because data-op import rejects arbitrary symbolic names such as
`d0`/`d1` in top-level `N_`.

- `baseline`: producer-owned pieces are split along `d1`, while the restickify
  output pieces are split along `d0`. This mirrors the high-byte-hop case we
  measured for `adds_then_matmul 2048`.
- `stage3b`: both producer and restickify output pieces are split along `d1`.
  This mirrors the certified zero-byte-hop Stage 3B mapping.

For each mode the probe can emit:

- `STCDPOpLx`: the main LX-to-LX data movement candidate.
- `ReStickifyOpLx`: restickify-specific LX candidate.
- `ReStickifyOpHBM`: HBM control path using the same data-op container.

## Commands

Generate standalone JSON:

```sh
SPYRE_RESTICKIFY_LX_DATAOP=1 \
python tools/restickify_lx_dataop_probe.py \
  --size 2048 \
  --num-cores 32 \
  --output-dir /tmp/restickify-lx-dataop-probe
```

Compile generated artifacts through DXP standalone:

```sh
SPYRE_RESTICKIFY_LX_DATAOP=1 \
python tools/restickify_lx_dataop_probe.py \
  --size 2048 \
  --num-cores 32 \
  --run-dcg \
  --output-dir /tmp/restickify-lx-dataop-probe
```

For data-op-only artifacts, `dcg_standalone -initSdsc` is the useful contract
check. `L3DlOpsScheduler_standalone` is aimed at ordinary `dscs_` compute-op
SDSCs and rejects data-op-only inputs with "No dsc in sdsc input".

## Interpretation

Passing JSON/unit tests only means the torch-spyre-side artifact shape is
plausible. Passing `dxp_standalone` means Deeptools accepts the standalone
bundle. Neither result proves physical RIU/LX traffic by itself.

The hardware evidence still needs the timing/counter path:

- kernel timing to compare `ReStickifyOpHBM` versus the LX data-op candidates
- `aiu-smi` or AIUPTI-derived traffic to see memory movement while the kernel
  runs
- generated-code inspection to confirm whether the backend selected L3SU/L3LU
  RIU movement or an HBM path

## Acceptance For This Stage

- Syntax check for the new emitter and probe.
- Unit test for the emitted JSON structure.
- Standalone generation of baseline and Stage 3B artifacts.
- `dcg_standalone` acceptance results from the Spyre pod:
  - `ReStickifyOpLx`: accepted for both baseline and Stage 3B-shaped artifacts.
  - `STCDPOpLx`: rejected when input/output stick dimensions differ; the
    backend asserts equal input/output stick dimensions, so this is a same-stick
    copy/movement candidate rather than a restickify candidate.
  - `ReStickifyOpHBM`: rejected in this minimal data-op form because the HBM op
    expects `coreIDtoANInfo` analytical-address metadata.

# Stage 34: LX Movement Inspection

## Summary

This stage checked whether the standalone data-op artifacts can show movement
that stays out of HBM.

The result is useful but nuanced:

- We can prove a Deeptools data-op can lower cross-core LX-resident movement
  without HBM placement by using a same-stick `STCDPOpLx` control.
- We can prove minimal `ReStickifyOpLx` artifacts lower as LX-resident
  restickifies, but in this minimal form they generate local LX/PE movement and
  do not yet demonstrate cross-core restickify all-to-all.
- We still have not proven physical hardware traffic with counters. This is
  compiler/backend artifact evidence, not RIU counter evidence.

## Commands

Generate and DCG-compile the restickify-shaped LX artifacts:

```sh
SPYRE_RESTICKIFY_LX_DATAOP=1 \
python tools/restickify_lx_dataop_probe.py \
  --size 2048 \
  --num-cores 32 \
  --output-dir /tmp/restickify-lx-dataop-dcg-probe \
  --run-dcg
```

Generate the same-stick STCDP LX-to-LX control:

```sh
SPYRE_RESTICKIFY_LX_DATAOP=1 \
python tools/restickify_lx_dataop_probe.py \
  --size 2048 \
  --num-cores 32 \
  --op STCDPOpLx \
  --stcdp-same-stick \
  --run-dcg \
  --output-dir /tmp/restickify-lx-stcdp-control
```

Lower the generated STCDP control to Dataflow IR:

```sh
DataOpStandalone \
  --ddsc-init-sdsc=/tmp/restickify-lx-stcdp-control/sdsc_baseline_STCDPOpLx_2048.json \
  --ddsc-out-dir=/tmp/restickify-lx-stcdp-standalone/baseline
```

## Results

### ReStickifyOpLx

`dcg_standalone` accepts both baseline-shaped and Stage3B-shaped
`ReStickifyOpLx` artifacts:

| Mode | DCG result | Log signal |
|---|---:|---|
| baseline | `0` | `Creating pcfg for coreID:* : LX : PE0` |
| Stage3B | `0` | `Creating pcfg for coreID:* : LX : PE0` |

The emitted data-op SDSC has LX-only placements for both input and output:

| Mode | LDS | Placement types | Pieces | HBM size |
|---|---|---|---:|---:|
| baseline | `dataIN_L0` | `lx` | 32 | 0 |
| baseline | `dataOUT_L0` | `lx` | 32 | 0 |
| Stage3B | `dataIN_L0` | `lx` | 32 | 0 |
| Stage3B | `dataOUT_L0` | `lx` | 32 | 0 |

`DataOpStandalone` output for these minimal `ReStickifyOpLx` cases contains
`lxlu`, `lxsu`, `pe0`, and local FIFO-style sends, with no `hbm`, `l3lu`, or
`l3su` data path. That means the minimal LX restickify artifact is resident in
LX, but it is not the cross-core all-to-all pattern we need for Stage3B proof.

### STCDPOpLx Same-Stick Control

The same-stick STCDP control gives the clean cross-core evidence:

| Mode | DCG result | Log signal |
|---|---:|---|
| baseline | `0` | `L3SU : L3LU : LX : PE0` |
| Stage3B | `0` | `LX : PE0` |

Dataflow IR summary:

| Mode | Lines | `l3lu` hits | `l3su` hits | `lxlu` hits | `lxsu` hits | `send` hits | `storage` hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 41307 | 1604 | 1856 | 224 | 192 | 320 | 0 |
| Stage3B | 1447 | 0 | 0 | 224 | 192 | 64 | 0 |

The baseline Dataflow IR contains debug names such as:

```text
c0-l3lu-ringDT-ring-lx-OL-0-0
c0-l3su-ringDT-lx-ring-OL-0-0
```

That is the compiler/backend evidence we wanted: the cross-core control uses
L3 load/store units and ring-facing `ring-lx` / `lx-ring` paths, while the
Stage3B-shaped local ownership case removes those L3 ring paths. The source
SDSC placements are LX-only and have `hbmSize_ = 0`.

## Interpretation

This proves the backend has an LX-resident cross-core movement path for
same-stick movement. It does not yet prove cross-core restickification, because
`STCDPOpLx` requires equal input/output stick definitions, while restickify
changes stick definitions.

For the restickify project, the current state is:

1. `ReStickifyOpLx` exists and compiles for LX-resident restickification.
2. `STCDPOpLx` proves Deeptools can lower LX-to-LX cross-core movement without
   an HBM placement.
3. We still need a backend contract for "change stick layout and move across
   cores" in one data-op, or we need to compose local `ReStickifyOpLx` with
   cross-core `STCDPOpLx`.

## Next Step

The next prototype should try a two-step data-op sequence:

1. `ReStickifyOpLx` local per-core stick-layout conversion.
2. `STCDPOpLx` same-stick cross-core movement.

If Deeptools can schedule both data-ops in one standalone artifact, then we can
compare it against the current `ReStickifyOpHBM` path and finally run a
hardware counter/timing test for HBM avoidance.

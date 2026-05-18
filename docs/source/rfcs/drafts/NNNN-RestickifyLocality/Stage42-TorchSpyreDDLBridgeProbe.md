# Stage 42: Torch-Spyre to DDL Restickify Bridge Probe

## Summary

Stage 41 proved that the Deeptools restickify DDL fixture can compile to an
LX/SFP/PT-only senprog when DXP's generic pre-DDC passes are bypassed. Stage 42
asked the next question: can a real Torch-Spyre-generated `ReStickifyOpHBM`
SDSC be reshaped into that DDL input form?

The answer is yes for a real `adds_then_matmul` restickify, with caveats. This
does not require a Deeptools source change. It is a Torch-Spyre-side diagnostic
bridge that rewrites the SDSC contract and then uses the Stage 41 preload shim
as the validation harness.

## Code Change

Added:

```text
tools/restickify_torch_spyre_ddl_bridge_probe.py
```

The probe:

1. reads a Torch-Spyre `ReStickifyOpHBM` SDSC,
2. synthesizes a compact DDL-style restickify input:
   - LX-only input/output allocations,
   - `INPUT`/`OUTPUT` primary data-space roles,
   - two `dataStageParam_` entries,
   - an LX-local transfer/loop skeleton,
   - preserved `coreIdToWkSlice_`, `numWkSlicesPerDim_`, and fold cardinality,
3. runs `ddc_standalone` and `dcc_standalone`,
4. optionally runs `dxp_standalone` with the Stage 41 preload shim, and
5. reports the DDC/DCC/DXP return codes plus senprog token counts.

This is a probe, not a production lowering path.

## Commands

Regenerate the known high-signal case:

```sh
SPYRE_RESTICKIFY_RING_TELEMETRY=1 \
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=0 \
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=0 \
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --output-dir /tmp/stage42-regen-probe \
  --fail-on-error
```

Bridge the first generated restickify SDSC:

```sh
python3 /tmp/restickify_torch_spyre_ddl_bridge_probe.py \
  --sdsc /tmp/torchinductor_1000800000/tmpc91aha5k/inductor-spyre/sdsc_fused_add_t_0_vs3e120g/sdsc_0_ReStickifyOpHBM.json \
  --output-dir /tmp/stage42-bridge-adds2048-add \
  --deeptools-bin /opt/ibm/spyre/deeptools/bin \
  --senarch rcudd1a \
  --run-deeptools \
  --run-dxp-preload
```

Bridge the second generated restickify SDSC:

```sh
python3 /tmp/restickify_torch_spyre_ddl_bridge_probe.py \
  --sdsc /tmp/torchinductor_1000800000/tmpc91aha5k/inductor-spyre/sdsc_fused_mm_1_lkf0ek44/sdsc_0_ReStickifyOpHBM.json \
  --output-dir /tmp/stage42-bridge-adds2048-mm \
  --deeptools-bin /opt/ibm/spyre/deeptools/bin \
  --senarch rcudd1a \
  --run-deeptools \
  --run-dxp-preload
```

## Results

The regenerated `adds_then_matmul` case still produced the same compiler
telemetry guardrail:

```text
restickifies=2
bytes=16777216
byte_hops=67108864
```

### First 2048 Restickify: Success

The first generated restickify was an HBM/HBM Torch-Spyre SDSC:

```text
source schedule nodes: 2
source allocations: hbm, hbm
source slices: mb:32, out:1
```

The bridge rewrote it into a DDL-style LX-local input:

```text
synthesized schedule nodes: 7
synthesized allocations: lx, lx
synthesized dataStageParam count: 2
```

The full DDC/DCC/DXP-with-shim path succeeded:

```text
ddc_rc=0
dcc_rc=0
dxp_rc=0
```

The generated senprog has LX/SFP/PT work and no visible L3/HBM tokens:

```text
bytes: 958793
LXLU: 32
LXSU: 32
SFP: 896
PT: 8928
L3LU: 0
L3SU: 0
HBM: 0
```

This is the strongest evidence so far that a real Torch-Spyre-generated
restickify from the high-signal 2048 case can be represented as an LX-local DDL
restickify program.

### Second 2048 Restickify: DDC Success, DCC/DXP Failure

The second generated restickify also reached DDC successfully:

```text
ddc_rc=0
```

But DCC/DXP rejected the DDC-expanded program with an LRF boundary error:

```text
dcc_rc=1
dxp_rc=-6
Register initialization out of boundary:
lxlu0 : LRF0 : 8126336
```

So the bridge is not yet a universal replacement for current HBM restickify
lowering. Some restickify layouts/shapes satisfy the DDL template contract and
some still need deeper schedule/register constraints.

### Fold Cardinality Check

A smaller 8-core Torch-Spyre restickify also succeeded when the bridge
preserved Torch-Spyre's emitted fold cardinality:

```text
ddc_rc=0
dcc_rc=0
dxp_rc=0
senprog: LXLU=8, LXSU=8, SFP=224, PT=872, HBM=0
```

Forcing the Deeptools fixture's `coreletFoldProp_.factor_ = 2` on that same
Torch-Spyre SDSC failed immediately:

```text
DtException: Different cardinality between json and caller
```

That means a real bridge must preserve the fold structure emitted for the
Torch-Spyre graph. Copying the fixture's corelet shape blindly is invalid.

## Interpretation

We have moved from "Deeptools has a standalone LX-local fixture" to "a real
Torch-Spyre restickify can be reshaped into that LX-local DDL contract." This
is a meaningful step toward proving that restickification does not inherently
require an HBM round trip.

The result also narrows the remaining blockers:

- Torch-Spyre currently emits regular HBM/HBM restickify SDSCs.
- The DDL path expects a different compact input contract.
- DXP must route that compact input around generic pre-DDC corelet splitting
  and L3 scheduling, as shown in Stage 41.
- Larger or differently oriented restickifies can still hit DCC/DXP register
  bounds after DDC expansion.

## Next Step

The next production-shaped prototype should be narrow:

1. Add a Torch-Spyre experimental flag that emits DDL-style LX-local restickify
   SDSCs only for cases that match the successful contract.
2. Require a post-emit compile check in prototype mode: DDC + DCC + DXP must
   produce a senprog with `HBM=0`, `L3LU=0`, and `L3SU=0`.
3. Leave all unsupported restickifies on the existing HBM path.

This keeps the project honest: we only claim LX-local restickify for the subset
that the toolchain actually compiles today.

## Validation

Local:

```text
python3 -m py_compile tools/restickify_torch_spyre_ddl_bridge_probe.py
```

Pod:

```text
python3 /tmp/restickify_torch_spyre_ddl_bridge_probe.py ...
```

Artifacts were copied locally under:

```text
artifacts/stage42_torch_spyre_ddl_bridge_probe/
```

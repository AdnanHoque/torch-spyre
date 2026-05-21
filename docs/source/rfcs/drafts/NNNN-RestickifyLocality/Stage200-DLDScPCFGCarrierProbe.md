# Stage 200: DLDSc/PCFG Carrier Probe

## Goal

Retry the Torch-Spyre-only "legal DLDSc" route for LX-to-LX restickify:
package the already-proven PT/LX bridge as something the normal DXP bundle path
can accept, without requiring Deeptools to import mixed `datadscs_` inside a
bundle SDSC.

This stage does not change hardware behavior. It is an artifact-shape probe.

## Setup

Input mixed SDSC:

```text
/tmp/torchinductor_1000800000/tmp7snvp9fy/inductor-spyre/sdsc_fused_add_t_0_hi0dsfdn/sdsc_1_MixedReStickifyOpWithPTLxConsumer.json
```

Probe:

```bash
python3 tools/restickify_dldsc_pcfg_probe.py \
  --mixed-sdsc /tmp/torchinductor_1000800000/tmp7snvp9fy/inductor-spyre/sdsc_fused_add_t_0_hi0dsfdn/sdsc_1_MixedReStickifyOpWithPTLxConsumer.json \
  --output-dir /tmp/stage200-dldsc-pcfg-v3 \
  --run-deeptools
```

Artifacts copied locally:

```text
artifacts/stage200_dldsc_pcfg/summary.json
```

## Variants

The probe first asks `dcg_standalone` to export the mixed SDSC's generated
PCFG. It then creates DLDSc-only carriers:

- remove `datadscs_` but keep the mixed schedule;
- remove `datadscs_` and clear the schedule;
- remove `datadscs_` and emit a normal DL-only schedule using
  `datadsc_idx=-1`;
- attach expanded or compressed PCFG to those DL-only carriers.

It also runs a control:

- `dcg_standalone -skip_pcfggen -initSdsc <mixed> -initPcfg <pcfg> -s`

The control verifies that the exported PCFG itself is usable when a standalone
tool is explicitly told not to regenerate PCFG.

## Results

| Variant | DCC | DXP | Units | Meaning |
|---|---:|---:|---|---|
| `dldsc_only_no_pcfg_keep_schedule` | -6 | -6 | none | Invalid: schedule still references missing data-op indices. |
| `dldsc_only_no_pcfg_clear_schedule` | -6 | -6 | none | Invalid: DCC's PCFG-to-DF path expects a data-op module when no schedule exists. |
| `dldsc_only_no_pcfg_dl_schedule` | 0 | 0 | `l3lu:32,l3su:32` | Legal DL-only carrier, but it contains no LX bridge. |
| `dldsc_only_full_pcfg_dl_schedule` | -6 | -6 | none | Embedded full PCFG is not a usable bundle carrier here. |
| `dldsc_only_compressed_pcfg_dl_schedule` | -6 | -6 | none | Embedded compressed PCFG is not a usable bundle carrier here. |
| `dcg_exported_strip_dataops` | 0 | 0 | `l3lu:32,l3su:32` | Exported mixed SDSC stripped to DL-only is legal but loses the bridge. |
| `dcg_exported_full_pcfg` | -6 | -6 | none | Full PCFG still does not turn the carrier into a bridge. |
| `dcg_skip_pcfg_control` | 0 | n/a | `l3/lx/pt/sfp/pe` | The PCFG is valid when consumed by the standalone skip-PCFG path. |

Control unit summary:

```text
l0lu0:32, l0su0:32, l3lu:32, l3su:32,
lxlu0:32, lxlu1:32, lxsu0:32, lxsu1:32,
pe0:32, ptrow0_0:32, sfp0:32
```

The control generated 13,760 counted program instructions and no HBM marker in
the generated senprog text.

## Interpretation

The result is negative for the clean DLDSc/PCFG carrier idea.

The exported PCFG is real: `dcg_standalone -skip_pcfggen` can consume it and
produce the expected LX/PT/SFP/L3 program. However, the normal DCC/DXP SDSC
path does not provide an equivalent "trust this embedded PCFG" mode. It
regenerates or stitches from the SDSC structure. Once `datadscs_` are removed,
the legal DLDSc carrier either:

- compiles as a consumer-only DL artifact, losing the LX bridge; or
- crashes because embedded PCFG still references metadata that the DL-only
  carrier no longer owns.

So this route does not yet give us a production-ready Torch-Spyre-only
replacement for `ReStickifyOpHBM`.

## Consequence

The strongest working non-HBM artifact remains the mixed schedule:

- one consumer DL DSC in `dscs_`;
- the PT/LX bridge in `datadscs_`;
- a per-core schedule that runs bridge data ops before the consumer DL op.

To productionize without a broad Deeptools change, the next likely route is not
"embed PCFG in DLDSc". It is either:

- make Torch-Spyre emit an accepted mixed DL+data-op schedule through an
  officially supported Deeptools/DXP contract; or
- generate a true DL op/function for the LX endpoint contract so the bridge is
  not represented as a data op at all.

## Files

- `tools/restickify_dldsc_pcfg_probe.py`
- `artifacts/stage200_dldsc_pcfg/summary.json`

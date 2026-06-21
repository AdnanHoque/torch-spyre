# Deeptools STCDPOpLx Range Patch

This directory records the Deeptools side of the LX relayout STCDPOpLx prototype.
It is stored in the Torch fork as a third-party patch because the Deeptools repo is owned/gated separately.

- Deeptools base: `621bb9fad8`
- Deeptools prototype head: `29254c37d3f2ee5c96a7323fdfd701026b63546c`
- Deeptools prototype branch: `lx-relayout-stcdp-range-proto`
- Patch file: `lx-relayout-stcdp-range-deeptools.patch`

Apply from a Deeptools checkout at or near the base commit:

```bash
git checkout 621bb9fad8
git apply /path/to/lx-relayout-stcdp-range-deeptools.patch
```

The patch contains the narrow DXP/DCG/DCC changes needed for scheduled mixed SDSCs with ranged `STCDPOpLx` LX-to-LX remap payloads.
The Torch-side branch emits `SPYRE_ONCHIP_MOVE_CARRIER=stcdp_range` payloads that this patch lowers through the existing ring-transfer path.

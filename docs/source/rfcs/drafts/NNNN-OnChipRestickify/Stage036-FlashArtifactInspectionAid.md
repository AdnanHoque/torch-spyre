# Stage 036: Flash Artifact Inspection Aid

Date: 2026-05-26

## Purpose

Stage 035 proved that the patched Deeptools path can route overlap-prefix
`STCDPOpLx` prefetch PCFGs through `LXLU1/LXSU1/PE1`, but the proof was based
on manual cache inspection.  This stage adds a focused post-run inspector so
future `warp_overlap_probe` attempts can quickly compare:

- source mixed-SDSC `coreletId` requests;
- source LX start addresses;
- mixed-SDSC schedule rows and overlap rows;
- DXP debug PCFG component routing; and
- transfer-node corelet IDs and LX address endpoints.

## Tool

Added:

```text
tools/onchip_flash_artifact_inspect.py
```

The tool reads either a `TORCHINDUCTOR_CACHE_DIR` path or an
`tools/onchip_sdpa_sweep.py --output-json` file.  It is intentionally separate
from the sweep harness: the sweep stays responsible for compile/execute/result
collection, while this helper handles deeper artifact inspection after a run.

Strict mode for the current probe is:

```sh
python3 tools/onchip_flash_artifact_inspect.py \
  --sweep-json /tmp/sdpa-stage035-prefetch-corelet.json \
  --expect-prefetch-corelet 1 \
  --require-overlap-prefix
```

## Stage035 Cache Check

The command above was run against the saved Stage 035 pod cache by copying the
tool to `/tmp` on the pod and inspecting:

```text
/tmp/sdpa-stage035-prefetch-corelet.json
```

Result:

```text
exit code 0
```

The report confirms the expected route for the executed overlap-prefix sidecar:

```text
overlap_prefix=True
prefetch_corelet=1
overlap_rows: c0..c31#2=[2, 0, 1, 1] (32 cores)
source_op_corelets: ['1']
source_dst_start_addrs: ['[17408]', '[18432]', '[17920]', '[18944]']
debug_components: {'lxlu1': 128, 'lxsu1': 128, 'pe1': 128}
debug_transfer_corelet_ids: {'-1': 256, '1': 128}
```

The same cache also contains source-only tile sidecars without DXP debug output;
that is expected because only the executed sidecar is lowered through DXP in
the failing run.  The inspector validates source routing for all overlap-prefix
tiles and requires at least one routed DXP debug artifact for strict mode.

## Tests

Added:

```text
tests/_inductor/test_onchip_flash_artifact_inspect.py
```

Verification run:

```sh
python3 tests/_inductor/test_onchip_flash_artifact_inspect.py
python3 -m py_compile \
  tools/onchip_flash_artifact_inspect.py \
  tests/_inductor/test_onchip_flash_artifact_inspect.py
```

Result:

```text
2/2 passed
py_compile passed
```

## Changed Files

```text
tools/onchip_flash_artifact_inspect.py
tests/_inductor/test_onchip_flash_artifact_inspect.py
docs/source/rfcs/drafts/NNNN-OnChipRestickify/Stage036-FlashArtifactInspectionAid.md
```

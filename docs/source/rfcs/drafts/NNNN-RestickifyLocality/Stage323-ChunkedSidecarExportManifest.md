# Stage 323: Chunked Sidecar Export Manifest

## Summary

Stage 323 packages the Stage 321 row-chunk PT-LX sidecars into deterministic
export manifests. This does not launch hardware and does not replace the stock
`ReStickifyOpHBM` path. It only validates that each generated bridge chunk can
be exported independently into usable DeeRT artifacts with no `HBM` token in
the generated `senprog.txt`.

The new helper is:

```sh
python tools/restickify_chunked_sidecar_export.py \
  --sidecar-dir <kernel-code-dir> \
  --output-dir <export-manifest-dir> \
  --retries 5 \
  --timeout-seconds 60 \
  --require-no-hbm \
  --fail-on-error
```

The helper writes `manifest.json` with one record per sidecar chunk, including:

- sidecar path and chunk index;
- number of dataops in the sidecar;
- every export attempt, return code, stdout/stderr log path, `init.txt`, and
  `senprog.txt`;
- selected successful attempt;
- `senprog.txt` token counts for `HBM`, `L3LU`, `L3SU`, `LXLU`, `LXSU`, `SFP`,
  and `PT`.

These token counts are program-text counts, not hardware traffic counters.
They are useful as a lowering sanity check: `HBM=0` means this bridge program
does not contain an explicit HBM load/store route in the generated senprog.

## Commands

Common pod environment:

```sh
export HOME=/home/adnan-cdx
export DTI_PROJECT_ROOT=$HOME/dt-inductor-mixed
export PATH=$HOME/.local/bin:$DTI_PROJECT_ROOT/sentient/deeptools/bin:/opt/ibm/spyre/deeptools/bin:$PATH
source "$DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh"
cd /tmp/torch-spyre-goal
```

For each size:

```sh
base=/tmp/stage322-native-validgap-auto-rowchunk-files-real-sidecar-${size}/kernel_code/matmul_then_add_${size}/0001_sdsc_fused_addmm_t_0
out=/tmp/stage323-chunked-sidecar-manifest-${size}-r5-timeout
rm -rf "$out"
python tools/restickify_chunked_sidecar_export.py \
  --sidecar-dir "$base" \
  --output-dir "$out" \
  --retries 5 \
  --timeout-seconds 60 \
  --require-no-hbm \
  --fail-on-error
```

For the 512 first pass, `--timeout-seconds 30` was sufficient. The 2048 run
used `--timeout-seconds 60`.

## Results

| Size | Chunks | Successful | Failed | Total Attempts | Retried Chunks | Max Attempts | Timeouts | Selected `HBM` Tokens | Selected `LXLU` Tokens | Selected `LXSU` Tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 8 | 0 | 12 | 4 | 2 | 0 | 0 | 128 | 128 |
| 1024 | 16 | 16 | 0 | 22 | 3 | 4 | 0 | 0 | 512 | 512 |
| 2048 | 32 | 32 | 0 | 39 | 6 | 3 | 1 | 0 | 2048 | 2048 |

Full selected token totals:

| Size | `HBM` | `L3LU` | `L3SU` | `LXLU` | `LXSU` | `SFP` | `PT` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0 | 768 | 264 | 128 | 128 | 1856 | 8704 |
| 1024 | 0 | 1536 | 816 | 512 | 512 | 7424 | 34816 |
| 2048 | 0 | 2976 | 96 | 2048 | 2048 | 29696 | 139264 |

The failed attempts were still mostly the known post-export instability:
`returncode=-11` after artifact generation. The manifest helper retries those
attempts in isolated directories and selects only a clean `returncode=0`
attempt. In the 2048 run, one attempt hit the new per-attempt timeout and the
next retry succeeded.

## Interpretation

This moves the prototype one step forward:

- Stage 321 showed row chunks can avoid DCC IBUFF failures.
- Stage 323 shows all row chunks for 512, 1024, and 2048 can be exported into
  clean no-HBM DeeRT artifacts with bounded retry.

This is still not a runnable Torch-Spyre replacement path. The stock HBM
fallback remains the only normal runtime path. The remaining production-shaped
work is to consume the manifest-selected `init.txt` artifacts as an ordered
bridge sequence and bind them into the normal producer/restickify/consumer
bundle contract without reintroducing `ReStickifyOpHBM`.

## Artifacts

The pod manifests were generated under:

```text
/tmp/stage323-chunked-sidecar-manifest-512-r5-timeout/manifest.json
/tmp/stage323-chunked-sidecar-manifest-1024-r5-timeout/manifest.json
/tmp/stage323-chunked-sidecar-manifest-2048-r5-timeout/manifest.json
```

Local copies for analysis:

```text
artifacts/stage323_chunked_sidecar_manifest/manifest_512.json
artifacts/stage323_chunked_sidecar_manifest/manifest_1024.json
artifacts/stage323_chunked_sidecar_manifest/manifest_2048.json
```

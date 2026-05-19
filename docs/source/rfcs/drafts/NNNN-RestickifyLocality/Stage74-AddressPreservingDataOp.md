# Stage 74: Address-Preserving Data-Op Restickify Prototype

## Goal

Stage 73 proved that Deeptools can generate HBM-free core-to-core RIU transfer
programs. Stage 74 moved back toward Torch-Spyre restickify and tested the next
missing contract:

```text
Use real Torch-Spyre scheduled producer/consumer LX addresses
  while
lowering a restickify-like two-step data-op artifact.
```

This is still not production lowering. It is a standalone `DataOpStandalone`
prototype to check whether the data-op path can preserve endpoint address
identity.

## Tool

Added:

```text
tools/restickify_address_preserving_dataop_probe.py
```

The tool:

1. Finds a producer/restickify/consumer SDSC triplet in a generated Torch-Spyre
   code directory.
2. Matches producer output to restickify input by HBM base address.
3. Matches restickify output to consumer input by HBM base address.
4. Runs `L3DlOpsScheduler_standalone` on producer and consumer SDSCs.
5. Extracts the matched producer output and consumer input LX allocation starts.
6. Generates a two-step data-op seed:

   ```text
   ReStickifyOpLx -> STCDPOpLx
   ```

7. Patches endpoint `PieceInfo.PlacementInfo[*].startAddr` values:

   ```text
   data-op input endpoint  <- producer scheduled LX address
   data-op output endpoint <- consumer scheduled LX address
   ```

8. Runs `DataOpStandalone` and summarizes traffic terms in the stitched
   Dataflow MLIR.

## Fixture

Captured Torch-Spyre code directory:

```text
/tmp/restickify-input-fetch-capture/kernel_code/computed_transpose_adds_then_matmul_2048/0001_sdsc_fused_add_t_0
```

Triplet:

```text
sdsc_0_add.json
sdsc_1_ReStickifyOpHBM.json
sdsc_2_add.json
```

Matched flow:

```text
producer sdsc_0_add Tensor2
  -> restickify input Tensor0
  -> restickify output Tensor1
  -> consumer sdsc_2_add Tensor1
```

The matching was not hard-coded by tensor index. The probe matched the endpoint
by HBM base address:

```text
restickify input HBM base  = 51539607552
restickify output HBM base = 68719476736
```

## Commands

Stage3B-shaped address-preserving artifact:

```sh
cd /tmp/torch-spyre-lx-dataop
export PYTHONPATH=/tmp/torch-spyre-lx-dataop:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export SENTIENT_BASE_INSTALL_DIR=/opt/ibm/spyre
export DEEPTOOLS_INSTALL_DIR=/opt/ibm/spyre/deeptools
export PATH=/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/runtime/bin:$PATH
export LD_LIBRARY_PATH=/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}

python3 tools/restickify_address_preserving_dataop_probe.py \
  --code-dir /tmp/restickify-input-fetch-capture/kernel_code/computed_transpose_adds_then_matmul_2048/0001_sdsc_fused_add_t_0 \
  --output-dir /tmp/stage74-address-preserving \
  --mode stage3b
```

Baseline-shaped contrast:

```sh
python3 tools/restickify_address_preserving_dataop_probe.py \
  --code-dir /tmp/restickify-input-fetch-capture/kernel_code/computed_transpose_adds_then_matmul_2048/0001_sdsc_fused_add_t_0 \
  --output-dir /tmp/stage74-address-preserving-baseline \
  --mode baseline
```

## Address Evidence

The scheduler materialized these matched LX endpoints:

| Endpoint | Matched LDS | Scheduled LX base |
|---|---:|---:|
| producer output | `sdsc_0_add` `Tensor2` / `ldsIdx=2` | `16384` |
| consumer input | `sdsc_2_add` `Tensor1` / `ldsIdx=1` | `8192` |

The Stage3B artifact patched all 32 endpoint pieces:

| Endpoint | Pieces patched | Example before | Example after |
|---|---:|---:|---:|
| producer input endpoint | 32 | `0` | `16384` |
| consumer output endpoint | 32 | `1572864` | `8192` |

That is the core Stage 74 improvement over the earlier compact-address DDL
probe: the source and destination endpoint addresses now come from the
scheduled Torch-Spyre producer/consumer SDSCs.

## Lowering Result

Both address-preserving artifacts passed `DataOpStandalone`.

Traffic-term summary from `dataOp_out.mlir`:

| Mode | `DataOpStandalone` | `HBM` | `L3` | `L3LU` | `L3SU` | `LXLU` | `LXSU` | `LX` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-shaped | pass | 128 | 3075 | 1347 | 1600 | 512 | 448 | 3713 |
| Stage3B-shaped | pass | 0 | 0 | 0 | 0 | 512 | 448 | 1792 |

The Stage3B-shaped artifact also reported one-to-one data-op transfer metadata:

```text
0 --> [ 0 ]
1 --> [ 1 ]
...
31 --> [ 31 ]
```

## Interpretation

This is the strongest compiler-side restickify locality result so far:

- endpoint matching comes from a real Torch-Spyre generated bundle;
- endpoint LX addresses come from the Deeptools scheduler, not from a compact
  fake address map;
- the Stage3B-shaped two-step data-op lowers through `DataOpStandalone`;
- `LXLU/LXSU` activity remains present;
- `L3/HBM` text terms disappear in the Stage3B-shaped artifact;
- the baseline-shaped contrast keeps substantial `L3LU/L3SU` traffic.

This supports the core Stage3B hypothesis:

```text
if producer and restickify/consumer ownership are aligned,
the data-op path can preserve LX-resident movement and avoid ring-facing L3
traffic for the local case.
```

It also separates two ideas that were previously tangled:

1. Stage 73 proved no-HBM core-to-core RIU movement exists in Deeptools.
2. Stage 74 proves the restickify-like data-op path can preserve real
   Torch-Spyre scheduled endpoint LX addresses and, when Stage3B-shaped, lower
   without L3/HBM traffic terms.

## Remaining Blocker

This is not yet a production Torch-Spyre restickify replacement.

The artifact is still a standalone data-op SDSC lowered through
`DataOpStandalone`. Direct `senprog` generation for folded scheduled
multi-data-op SDSCs remains blocked by Deeptools:

```text
Codegen for Folded Super-DSC is not supported
```

And generic DXP bundle import still rejects raw `datadscs_`:

```text
Datadsc not allowed, use dldsc
```

So the next integration problem is packaging, not the address-matching proof.

## Next Step

The next step should be a runtime-packaging investigation:

1. Take the `DataOpStandalone` output for the Stage3B address-preserving
   artifact.
2. Identify the runtime-facing path that can execute that stitched dataflow
   output, or the current DLDSc equivalent expected by DXP.
3. Avoid returning to the compact DDL source-address trick unless we can prove
   it aliases the real producer allocation correctly.

The target success condition is now sharper:

```text
same endpoint address-preserving data-op artifact
  plus
runtime bundle packaging
  plus
correct tensor output
```

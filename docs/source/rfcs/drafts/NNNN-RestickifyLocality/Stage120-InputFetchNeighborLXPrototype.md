# Stage 120: InputFetchNeighbor LX-to-LX Prototype

## Summary

This stage produced the first working Deeptools compiler prototype for the
restickify movement we have been chasing:

```text
producer-owned LX buffer
  -> RIU-facing L3LU/L3SU transfer
  -> consumer-owned LX buffer
```

The generated program contains no HBM traffic and no HBM text at all. It uses
Deeptools' existing `InputFetchNeighbor` path, which preserves real scheduled
producer/consumer LX base addresses instead of inventing compact local
addresses.

This is not yet a Torch-Spyre end-to-end tensor-correct replacement pass. It is
a working compiler/codegen prototype that proves Deeptools can generate and
verify HBM-free core-to-core LX movement for the restickify edge shape.

## What Changed

Two probe-only changes were added:

- `tools/restickify_input_fetch_neighbor_probe.py`
  - adapts a Torch-Spyre producer/consumer pair into the installed
    `InputFetchNeighbor` contract;
  - retags the consumer restickify input as `INPUT`;
  - marks the producer output and consumer input as LX-pinned;
  - populates `coreStateInit_` from the scheduled LX allocation nodes;
  - aliases Torch-Spyre `mb/out` metadata to Deeptools' current `ij/in`
    `InputFetchNeighbor` assumptions;
  - can optionally reverse consumer core ownership to force cross-core traffic.

- `tools/dcg_inpfetch_senprog_probe.cpp`
  - imports the adapted producer/consumer SDSCs;
  - clears the in-memory trivial factor-1 fold that blocks the stock
    `dcg_inpfetch_standalone -s` path;
  - calls the normal Deeptools `runDcgForInputFetchNeighbor` path;
  - emits `senprog.txt`.

No Deeptools source was modified or pushed.

## Commands

Generate the adapted reverse-core fixture:

```sh
cd /tmp/torch-spyre-stage76-current
python3 tools/restickify_input_fetch_neighbor_probe.py \
  --code-dir /tmp/restickify-input-fetch-capture/kernel_code/computed_transpose_adds_then_matmul_2048/0001_sdsc_fused_add_t_0 \
  --output-dir /tmp/stage116-inpfetch-reverse \
  --adapt-scheduled-lx-neighbor \
  --alias-mb-out-to-ij-in \
  --consumer-core-map reverse \
  --run
```

Compile and run the senprog wrapper:

```sh
g++ -std=c++17 \
  -I/tmp/deeptools-headers-stage80 \
  -I/opt/ibm/spyre/deeptools/include \
  /tmp/torch-spyre-stage76-current/tools/dcg_inpfetch_senprog_probe.cpp \
  -L/opt/ibm/spyre/deeptools/lib \
  -ldcg -ldcg_fe -ldcg_be -ldsc -ldpc -lsharedtools -lutil -lcommon -ljson11 \
  -Wl,-rpath,/opt/ibm/spyre/deeptools/lib \
  -o /tmp/stage118-dcg-inpfetch-senprog-probe

/tmp/stage118-dcg-inpfetch-senprog-probe \
  /tmp/stage116-inpfetch-reverse/direct_0001_sdsc_fused_add_t_0_1/adapted_scheduled_lx_neighbor/consumer_main.scheduled.input_lx_neighbor.json \
  /tmp/stage116-inpfetch-reverse/direct_0001_sdsc_fused_add_t_0_1/adapted_scheduled_lx_neighbor/producer_pre.scheduled.lx_neighbor.json \
  /tmp/stage119-inpfetch-senprog-reverse
```

## Results

### Identity Ownership

The identity control keeps each producer slice on the same consumer core:

```text
0 --> [ 0 ]
1 --> [ 1 ]
...
31 --> [ 31 ]
```

The generated program verifies, but it has no load/store movement:

| Token | Count |
|---|---:|
| `HBM` / `hbm` | 0 |
| `Program for unit l3lu` | 32 |
| `Program for unit l3su` | 0 |
| `L3_LDU` | 0 |
| `L3_STU` | 0 |
| `L3_SYNC` | 32 |
| `EAR` | 0 |

This is the expected "already local" control case.

### Reverse Ownership

The reverse fixture forces cross-core movement:

```text
0 --> [ 31 ]
1 --> [ 30 ]
...
30 --> [ 1 ]
31 --> [ 0 ]
```

The generated programs verified successfully for all 32 cores:

```text
Verifying generated progs: Passed
```

The generated `senprog.txt` has the desired HBM-free RIU-facing signature:

| Token | Count |
|---|---:|
| `HBM` / `hbm` | 0 |
| `Program for unit l3lu` | 32 |
| `Program for unit l3su` | 32 |
| `L3_LDU` | 64 |
| `L3_STU` | 64 |
| `L3_SYNC` | 64 |
| `EAR` | 64 |

Representative instructions:

```text
Program for unit l3lu:
L3_MVLOOPCNT :: be:0 dyn_loop:0 imm:32 src0:0
L3_LDU :: be:0 burst:30 group:0 node:128 src0:0 src1:0
L3_ADDLARIMM :: be:1 imm:34 src0:0
L3_SYNC :: be:0 soft:1 synctag:81

Program for unit l3su:
L3_MVLOOPCNT :: be:0 dyn_loop:0 imm:32 src0:0
L3_STU :: be:0 burst:30 group:0 node:128 src0:0 src1:0
L3_ADDLARIMM :: be:1 imm:34 src0:0
```

The important observation is that cross-core movement appears as L3LU/L3SU
programs, not `ReStickifyOpHBM`, and no HBM program text is emitted.

## Interpretation

This answers the key hardware/compiler-path question:

```text
Does Deeptools have an LX-to-LX/core-to-core data movement path that can avoid
an HBM round trip?
```

Yes. The path is `InputFetchNeighbor`/`STCDPOpLx`, and the generated program
uses RIU-facing L3 load/store units with no HBM traffic.

It also clarifies why our earlier Torch-Spyre bundle splicing failed:

- replacing only a standalone restickify SDSC is not enough;
- the consumer boundary must also agree that its input is an internal LX
  neighbor input;
- separate runtime launches cannot safely hand LX-resident intermediates across
  an external op boundary;
- the correct production abstraction is an internal scheduled edge, not a
  post-hoc replacement of `ReStickifyOpHBM`.

## Remaining Work

The next prototype should move this from "standalone compiler proof" toward
Torch-Spyre integration:

1. Generate an internal-edge descriptor from Torch-Spyre when a producer output
   feeds a restickify input and the consumer can read an LX-neighbor input.
2. Avoid the `mb/out` to `ij/in` probe alias by either generalizing
   `InputFetchNeighbor` dimension ordering or emitting a Deeptools-native shape
   that satisfies the existing contract.
3. Package the generated InputFetchNeighbor program into the Torch-Spyre/Flex
   runtime artifact, rather than only printing `senprog.txt`.
4. Validate with a small value-correct graph:

```text
producer add -> LX-to-LX restickify/input-fetch -> consumer add
```

Success for the next stage means the first fused bundle retires on hardware
without `ReStickifyOpHBM` and without stream hardware error.

## Artifacts

Local copies:

```text
artifacts/stage120_inpfetch_lx_to_lx/identity_senprog.txt
artifacts/stage120_inpfetch_lx_to_lx/reverse_senprog.txt
artifacts/stage120_inpfetch_lx_to_lx/input_fetch_neighbor_reverse_summary.json
```

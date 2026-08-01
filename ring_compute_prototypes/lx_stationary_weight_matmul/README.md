# LX-fed stationary-weight matmul

This prototype is the fixed-grid causal proof for the ring-aware matmul
substrate. It does not change work division: both arms use the same `M4 N8 K1`
BMM, the same pointwise producer, and the same HBM-resident weight.

## Dataflow

1. Four producer cores (`0,8,16,24`) compute one M-cohort activation shard and
   leave it in LX.
2. The BMM keeps W N-sharded in HBM across eight owners per cohort.
3. The LX planner recognizes that each activation shard is needed by all eight
   N owners in its cohort and emits a `matmul_operand_broadcast` contract.
4. DeepTools lowers that contract to grouped `STCDPOpLx` fan-out into the
   BMM's already-reserved LX `INPUT` allocation immediately before the DL
   schedule step.
5. The producer, communication, and BMM execute as one device bundle. There is
   no standalone shuffle root, restickify root, or HBM round trip on the
   producer-to-BMM edge.

The matched control disables this realization, so the identical producer edge
spills to HBM and the BMM reloads it. This isolates core-to-core communication
from tiling and work-division effects.

The route uses the existing LX `STCDPOpLx`/input-fetch machinery. It proves
useful core-to-core fan-out and fused consumption; it does **not** yet prove
dual-fabric overlap, direct PT consumption of subtiles as they arrive, or a
hardware ring-utilization percentage.

## Compiler substrate

Torch-Spyre contributes only the missing binding layer:

- `lx_matmul_operand_broadcast` is an opt-in compiler configuration;
- a planned BMM operand retains the producer's sparse LX ownership and address;
- the grouped fan-out contract is serialized on the consumer SDSC;
- normal explicit-shuffle materialization remains unchanged for other edges.

The matching DeepTools delta is tracked in
`deeptools_input_operand.patch`. It:

- accepts BMM `INPUT` as well as `KERNEL` for this existing contract;
- exposes the already-existing resident input-fetch materialization under
  `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_RESIDENT_IFN=1`;
- records the physical lowering as `lowered_resident_input_fetch` rather than
  claiming that the diagnostic kernel-neighbor route was realized.

## Device results

All rows passed CPU-reference correctness, exact structure, backend-plan, and
Kineto device-event gates on `a6-quantization/adnan-cdx-spyre-dev-pf`.

| Shape | HBM control | LX-fed | Speedup |
|---|---:|---:|---:|
| M64 K2048 N4096, repeated | 235.220 us | 150.729 us | **1.5605x** |
| M128 K2048 N4096 | 353.5735 us | 178.5265 us | **1.9805x** |
| M256 K2048 N4096 | 414.896 us | 237.132 us | **1.7496x** |
| M512 K2048 N4096 | 804.3255 us | 436.912 us | **1.8409x** |
| M64 K2048 N8192 | 448.2005 us | 255.426 us | **1.7547x** |
| M64 K2048 N12288 | 661.917 us | 383.5345 us | **1.7258x** |

The repeated M64 row is an HBM-LX-LX-HBM bracket with 10 warmups and 30
events per run. The other rows use 3 warmups and 10 events as regime probes.
Full hashes, paths, protocol, and nonclaims are in `device_results.json`.

Concatenating two or three N4096 projections into one wider stationary-W BMM
is about `1.18x` faster than summing two or three accepted single-projection LX
bundle medians. That is an amortization signal, not yet a directly timed
multi-root QKV/MLP integration.

### Direct fused-projection result

The probe now emits production-like logical output tuples. The fused arm
concatenates W, executes one BMM, and returns Q/K/V or gate/up as zero-copy
views; the separate arm emits one BMM per projection. Every row below uses 10
warmups and 30 matched device events and passes correctness, structure, backend
plan, and trace gates.

| Shape | Fused LX | Best fused HBM | Separate HBM | LX / best fused | LX / separate |
|---|---:|---:|---:|---:|---:|
| M64 K2048 P2 4096/4096 | 255.9275 us | 429.4255 us | 437.9515 us | **1.6779x** | **1.7112x** |
| M64 K4096 QKV 4096/1024/1024 | 398.6985 us | 652.978 us | 836.789 us | **1.6378x** | **2.0988x** |
| M64 K4096 gate/up 12800/12800 | 1609.6935 us | 2549.586 us | 3797.929 us | **1.5839x** | **2.3594x** |
| M512 K4096 QKV 4096/1024/1024 | 729.745 us | 1959.398 us | 2738.2085 us | **2.6850x** | **3.7523x** |
| M512 K4096 gate/up 12800/12800 | 2145.0875 us | 7274.988 us | 12978.1095 us | **3.3915x** | **6.0502x** |

`Best fused HBM` is the automatic-grid incumbent dataflow. The same-grid HBM
controls are also retained in `multi_projection_results.json`; their LX
speedups range from `1.6449x` to `5.7624x`, proving that the gain is not from a
different work division.

M512 initially fell back because the allocator reserved disjoint 1 MiB S1 and
S2 buffers against a 1.984 MiB usable LX. Resident matmul fan-out does not need
that copy: every source owner already contains its destination bytes, and
remote replicas use the same address. The allocator now aliases S2 to S1 only
for size-preserving matmul broadcasts whose source-owner and destination-owner
coordinates are identical. Generic shuffles and expanding all-gathers remain
disjoint. This unlocks M512 without changing M64 timing.

The final stop decision is evidence-based:

- auto-grid LX is rejected because its strided physical consumer order is
  numerically incorrect on the current resident backend route;
- a temporary multi-consumer implementation was correct but slower: 1009.1675
  us for three Q/K/V BMMs and 991.9125 us for Q plus fused-KV, versus 729.745 us
  for one fused BMM, so it was removed;
- M512 gate/up is only 3.3898% above the sum of prior conversion-free
  per-projection compute-oracle medians, leaving too little credible benefit
  for a chunked-IFN schedule path.

The remaining QKV gap is primarily heterogeneous BMM schedule efficiency, not
ring transfer. It should be revisited only with a single-fan-out, mixed-grid
consumer contract or stronger PT scheduling evidence—not by adding another
copy or restickify.

## Reproduction

The accepted device stack used:

```text
Torch-Spyre head: 2c1ab140d23f7e200ef45ef26b057229cb393727
DeepTools head:   a4930be14b6e7d01f7447b7692a79a20487c09c3
DeepTools diff:   12d37dfbe5c115092cb46ab9a9eb132245436d3d0284abbd4d18bf6e2fbebd65
DXP binary:       fb15c38f2207449eaa725aed62d8e365954f1aa0f4a95286d5dd90b9cdad28be
```

Apply and build the backend delta, then run the probe with the source checkout
on `PYTHONPATH`:

```bash
git -C "$DEEPTOOLS_SRC" apply \
  ring_compute_prototypes/lx_stationary_weight_matmul/deeptools_input_operand.patch
cmake --build "$DEEPTOOLS_BUILD" -j8 --target dxp_standalone

export DEEPTOOLS_PATH="$DEEPTOOLS_SRC"
export DXP_LX_FRAC_AVAIL=0.2
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_RESIDENT_IFN=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR="$RUN_DIR/backend_plans"

python ring_compute_prototypes/lx_stationary_weight_matmul/probe.py \
  --run-dir "$RUN_DIR" --route lx --m 64 --k 4096 \
  --projection-widths 4096,1024,1024 --projection-schedule fused --grid fixed \
  --warmups 10 --runs 30 --expected-torch-head "$TORCH_SPYRE_HEAD"
```

Use `--route hbm` without the two `MATMUL_OPERAND_BROADCAST` variables for the
matched control.

## Decision

The fused LX projection algorithm and destination-alias substrate are worth
keeping. They deliver large same-grid and best-control wins across decode and
M512. Do not add multi-consumer or chunk-streaming machinery on the current
evidence. The next gate is model integration with producer-native activation
layout and consumer-native Q/K/V or gate/up views, followed by E2E timing.

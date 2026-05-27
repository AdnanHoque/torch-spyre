# Stage 058: K/V Repack Executable Pair Probe

Date: 2026-05-27

## Purpose

Stage057 emitted a descriptor-only K/V repack plan for the real block64 prefill
boundary:

```text
low-core ReStickifyOpHBM output
  -> future 32-core batchmatmul input1
```

Stage058 promotes that plan to a default-off executable-facing probe.  It still
does not complete warp-specialized prefill attention: the executable path reaches
device runtime, but it is not value-correct.

## Change

New gates:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_TILE=-1
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_IFN_TRANSFER=1
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_SUBPIECE_REUSE=1
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_GROUP_SIZE=0
```

`PAIR_TILE=-2` scans for the first eligible K/V edge.  The pair builder emits:

- a producer sidecar replacing the low-core `ReStickifyOpHBM`, with its output
  LX-pinned;
- a consumer sidecar replacing the future `batchmatmul`, with an `STCDPOpLx`
  copy into the consumer input LX before compute;
- executable PieceInfo without the plan-only `broadcastSourcePieceKey_` and
  `broadcastConsumerCore_` keys, because DXP rejects those custom keys.

Sweep variants:

```text
kv_repack_pair_auto
kv_repack_pair_no_ifn_auto
kv_repack_pair_no_reuse_auto
kv_repack_pair_group16_auto
```

The sweep child also now explicitly calls `torch_spyre._autoload()` before
importing fallback modules or constructing `spyre` tensors, and the sweep script
bootstraps its repository root on `sys.path` so pod runs use the synced checkout
instead of an older installed Torch-Spyre.

## Local Validation

```text
python3 -m py_compile torch_spyre/_inductor/config.py \
  torch_spyre/_inductor/codegen/bundle.py \
  torch_spyre/_inductor/onchip_realize.py \
  tools/onchip_sdpa_sweep.py \
  tests/_inductor/test_config_logic.py \
  tests/_inductor/test_onchip_realize_logic.py \
  tests/_inductor/test_onchip_sdpa_sweep_logic.py \
  tests/_inductor/test_onchip_flash_pipeline_logic.py
python3 tests/_inductor/test_config_logic.py
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py
python3 tests/_inductor/test_onchip_realize_logic.py
python3 tests/_inductor/test_onchip_flash_pipeline_logic.py
git diff --check
```

Results:

```text
test_config_logic.py: 17/17 passed
test_onchip_sdpa_sweep_logic.py: 23/23 passed
test_onchip_realize_logic.py: 71/71 passed
test_onchip_flash_pipeline_logic.py: 11/11 passed
git diff --check: clean
```

The same config, sweep, and realize tests passed in the pod worktree:

```text
/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
```

## Device Results

All device probes used:

```text
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
TORCH_DEVICE_BACKEND_AUTOLOAD=0
PATH=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp:$PATH
LD_LIBRARY_PATH=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/lib:...
```

### Parser cleanup

The first executable sidecar included the plan-only PieceInfo keys and aborted in
DXP's PieceInfo parser.  The executable path now strips those keys while the
plan artifact keeps them.

### Default pair

Cache:

```text
/tmp/sdpa-stage058-kv-repack-pair-default-kv_repack_pair_auto-B1-H2-L128-D64-C0-643250-745880
```

The intended sidecars were selected:

```text
sdsc_mixed_flash_kv_repack_broadcast_pair_1_input1_producer.json
sdsc_mixed_flash_kv_repack_broadcast_pair_1_input1_consumer.json
```

Consumer metadata:

```text
source=3_ReStickifyOpHBM
consumer=4_batchmatmul
input=1
source pieces=2
destination pieces=64
source_lx_base=16384
consumer_lx_base=278528
schedule0=[[0,-1,0,1],[-1,0,1,0]]
```

The program compiled and ran, but failed value correctness:

```text
Mismatched elements: 16309 / 16384 (99.5%)
Greatest absolute difference: 22272.0
```

### IFN-transfer A/B

Cache:

```text
/tmp/sdpa-stage058-kv-repack-pair-noifn-kv_repack_pair_no_ifn_auto-B1-H2-L128-D64-C0-642897-15114
```

This disabled only the input-fetch transfer marker:

```text
kv_repack_input_fetch_transfer=False
transfer_nodes=[]
source pieces=2
destination pieces=64
```

It produced the same value failure:

```text
Mismatched elements: 16309 / 16384 (99.5%)
Greatest absolute difference: 22272.0
```

So the transfer marker is not the primary corruption source for this run.

### Subpiece-reuse A/B

Cache:

```text
/tmp/sdpa-stage058-kv-repack-pair-noreuse-kv_repack_pair_no_reuse_auto-B1-H2-L128-D64-C0-643625-882822
```

This emitted:

```text
op={"name": "STCDPOpLx", "enSubPieceReuse": 0}
source pieces=2
destination pieces=64
```

It did not produce a value result.  Runtime timed out after 280 seconds with a
compute response-block timeout:

```text
RAS::RUNTIMESCHEDULER::ComputeHardwareError
RAS::RESPONSEWORKER::RbTimeOut
```

This makes disabling subpiece reuse an unsafe descriptor shape for this graph.

### Group-16 A/B

Cache:

```text
/tmp/sdpa-stage058-kv-repack-pair-group16-kv_repack_pair_group16_auto-B1-H2-L128-D64-C0-644044-583999
```

This split the broadcast into two data-ops:

```text
group0 cores 0..15, source pieces=2, destination pieces=32
group1 cores 16..31, source pieces=2, destination pieces=32
```

DXP rejected the two-dataop mixed SDSC during PCFG-to-dataflow conversion:

```text
DtException: senpcfgs_.count(pair)
PCFGToDataflowIR.cpp line 3346
```

That means the desired <=16-consumer grouping cannot currently be expressed as
two disjoint `STCDPOpLx` dataops in the same consumer mixed SDSC row.

## Current Status

Stage058 advances the warp-specialized prefill path from a descriptor-only K/V
plan to selected, executable producer/consumer sidecars.  It also narrows the
next blocker:

- the one-dataop 2-to-64 fanout compiles and runs but produces wrong values;
- removing the IFN marker does not change the failure;
- disabling subpiece reuse hangs the runtime;
- splitting the fanout into two 16-core groups is rejected by DXP lowering.

The next useful step is not more causal plumbing.  It is a K/V fanout contract
that DXP can lower and runtime can execute value-correctly, likely either as a
Foundation-supported grouped broadcast, a sequence of legal sidecars, or a
different producer/consumer schedule that avoids one producer subpiece feeding
all 32 consumer cores in one data-op.

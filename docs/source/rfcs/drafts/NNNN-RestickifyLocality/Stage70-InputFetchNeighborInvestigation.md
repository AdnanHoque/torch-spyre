# Stage 70: Input-Fetch Neighbor Investigation

## Summary

Stage 69 showed that the DDL bridge can generate a no-HBM/no-L3 restickify-like
program:

```text
HBM=0, L3LU=0, L3SU=0, LXLU=32, LXSU=32, SFP=896, PT=8928
```

but it is not correct because the compact source-address mode erases the
producer tensor's real LX allocation identity.

This investigation looked for an existing Deeptools contract that keeps both:

```text
producer LX allocation identity
  plus
cross-core/LX-neighbor transfer lowering
```

The best lead is Deeptools' existing **InputFetchNeighbor** path. This looks
closer to the production mechanism we need than the DDL compact-address bridge.

## Key Finding

Deeptools already has a path named `InputFetchNeighbor` that builds an
`STCDPOpLx` data operation from a producer `SuperDsc` and a consumer/main
`SuperDsc`.

The important function is:

```text
deeptools/dcg/dcg_fe/pcfg_gen/inputNeighFetchOp.cpp
  DcgFE::fillDataDSCForInputFetchNeighbor(...)
```

That function constructs a data op whose input pieces come from the producer
output DSC and whose output pieces feed the consumer input DSC. Crucially, it
copies the actual LX base addresses from each side:

```text
ldsDLDSC->coreStateInit_[cidx].lbrInit_[0]
```

This is exactly the contract the compact DDL bridge was missing. The bridge made
every source core read local address `0`; `InputFetchNeighbor` instead derives
the source/destination piece addresses from the real producer/consumer
`coreStateInit_` records.

## Existing Entry Points

There are two relevant entry points.

Standalone two-SDSC flow:

```text
deeptools/dcg/tools/dcg_inpfetch_standalone.cpp
  -initSdscMain <consumer>
  -initSdscPre  <producer>
  runDcgForInputFetchNeighbor(mySDscMain, &mySDscPre)
```

Scheduled in-bundle flow:

```text
deeptools/dcg/dcg_manager/dcg_manager.cpp
  DcgManager::runDcgForDataOpsDlOps(...)
```

When `coreIdToDscSchedule` contains a schedule step with both a data-op index
and a DL-op index:

```text
[datadsc_idx, dldsc_idx, after_sync, before_sync]
```

the non-`SENPCFG` path treats this as requiring input-neighbor fetch and calls:

```text
generatePcfgIRForDataOpInpFetch(mySDsc, nullptr, datadsc_idx)
```

The scheduler also recognizes LX-neighbor input tensors:

```text
deeptools/dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp
  isLabeledDsLXNeighbor(...)
```

It identifies a consumer input as LX-neighbor when:

1. the labeled data structure is LX pinned,
2. the labeled data structure is an input,
3. the consumer DSC appears in a schedule step paired with a data-op index.

## Built-In Ring Traffic Model

`InputFetchNeighbor` also has internal ring-traffic accounting:

```text
deeptools/dcg/dcg_fe/pcfg_gen/inputNeighFetchOp.cpp
  DcgFE::computeTrafficPerChunk(...)
```

It walks producer-core data-table entries and uses:

```text
dtTable_[dtIdx].CCWHopCWHop
dtTable_[dtIdx].selectedMCMode
```

to accumulate per-chunk traffic in sticks:

```text
total_ccw
total_cw
core_ccw_pass
core_cw_pass
datatr_prechunk_both
```

This is useful because it is a Deeptools-native version of the ring-hop model we
have been approximating in Torch-Spyre telemetry.

## KB Cross-Check

The Spyre knowledgebase supports this direction:

- The AIU ring is the sole path for off-chip-memory-to-LX and cross-core
  LX-to-LX movement.
- L3LU/L3SU are the ring-facing interfaces.
- The current DSC2 schedule tree lowers transfer nodes into L3LU and
  LXLU/LXSU `dataflow.send`/`dataflow.receive` pairs.
- The proposed Schedule IR explicitly owns transfer plans, LX allocations, sync
  boundaries, and core ownership.

So the missing abstraction sits at the schedule/allocation boundary, not inside
the late DDL bridge spelling.

## Why This Is Better Than The DDL Bridge

The DDL bridge gave us one useful diagnostic: it proved that a generated program
can contain LXLU reads and LXSU writes without HBM/L3 tokens.

But the bridge cannot currently express:

```text
read the producer's actual per-core LX allocation
  while
presenting a DDC/DCC-friendly LXLU source transfer
```

`InputFetchNeighbor` appears designed for that exact problem. It builds data
pieces from producer and consumer DSC ownership, uses actual LX addresses, and
has ring route accounting.

## Current Concerns

There is still one important integration question.

In `runDcgForDataOpsDlOps`, a schedule step with both `datadsc_idx` and
`dldsc_idx` enters the input-fetch path for non-`SENPCFG` targets. The DCC
`PCFGToDFManager` maps scheduled steps by `datadsc_idx` when
`coreIdToDscSchedule` exists. We need to verify with a tiny artifact whether the
resulting DCC/DDF path emits both:

1. the input-neighbor data movement program, and
2. the consumer DL program,

in the expected order.

This is a verification task, not a reason to abandon the path.

There is also an existing Torch-Spyre-side data-op prototype:

```text
torch_spyre/_inductor/codegen/restickify_lx_dataop.py
```

That prototype is still useful, but it builds standalone data-op payloads from a
single `SDSCSpec`. The new hypothesis is slightly different: generate or reuse a
producer `SuperDsc` and a consumer `SuperDsc`, then let Deeptools'
`InputFetchNeighbor` construct the identity-preserving `STCDPOpLx` movement
between them. In other words, do not merely replace `ReStickifyOpHBM` with a
single standalone data op; connect the producer and consumer ownership records
through the input-neighbor contract.

## Next Experiment

Use the Stage 62/69 clean fixture:

```python
def computed_transpose_adds_then_matmul(a, b, c, d):
    return (a + (b + c).t()) @ d
```

Build a tiny standalone `InputFetchNeighbor` probe:

1. Capture/export the producer add `SuperDsc` and consumer add/restickify-side
   `SuperDsc` from the generated bundle.
2. Run `dcg_inpfetch_standalone` with:

   ```text
   -initSdscPre  <producer add output owner>
   -initSdscMain <consumer/restickify input owner>
   ```

3. Inspect the generated data op for:

   ```text
   HBM=0
   L3LU/L3SU>0
   ringDT comments with lx/ring endpoints
   ```

4. Confirm source piece addresses match producer `coreStateInit_.lbrInit_`
   rather than compact local address `0`.
5. If standalone works, build the same shape as a scheduled SDSC with
   `coreIdToDscSchedule` step `[0, 0, false, false]`.
6. Only then wire Torch-Spyre restickify lowering to emit this schedule/data-op
   shape behind a default-off flag.

Stage 73 corrected the expected token signature: cross-core LX-to-LX traffic over
RIU can legitimately appear as ring-facing `L3LU`/`L3SU` instructions. The key
negative proof is absence of `HBM`, not absence of `L3`.

## Working Conclusion

The next serious path should be:

```text
stop broadening the DDL compact-address bridge
  and
prototype restickify as Deeptools InputFetchNeighbor / STCDPOpLx
```

This is the first path we have found that appears to preserve the producer's LX
allocation identity and has first-class Deeptools ring traffic accounting.

The compact DDL bridge should remain a diagnostic artifact only.

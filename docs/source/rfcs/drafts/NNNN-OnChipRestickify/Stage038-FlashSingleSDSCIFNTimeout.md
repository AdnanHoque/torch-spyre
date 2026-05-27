# Stage 038 - Single-SDSC IFN overlap compiles but times out

## Summary

The Stage038 probe replaced the unsafe independent overlap sidecar with an
InputFetchNeighbor-shaped single-SDSC diagnostic tile.  The diagnostic can now
be generated with a DXP-importable transfer marker, but executing it as a bundle
replacement times out on device.  The current Torch path therefore keeps the
diagnostic artifact for inspection and marks it `ifn_runtime_safe: false`, so
bundle replacement fails closed.

## Torch artifact shape

The generated `generated-flash-prefill-overlap-prefix-ifn-tile` artifact:

- attaches the first batchmatmul input (`ldsIdx_ == 0`);
- rewrites that input LX-resident at `CONSUMER_LX_BASE`;
- adds a full `NO_COMPONENT -> LX` transfer marker with canonical `prev_`,
  transfer-offset envelope fields, and `allocUsers_` linkage from the LX
  allocate node;
- hydrates legacy `CoreD_`, `CoreletD_`, and `B_` from DSC2
  `dataStageParam_` / shard metadata so Deeptools IFN code does not see `-1`
  dimensions;
- emits one same-LX `STCDPOpLx` dataop in row `[0, 0, 0, 0]`.

This is intentionally diagnostic-only:

```json
"ifn_runtime_safe": false,
"ifn_runtime_rejection_reason": "single_sdsc_ifn_no_real_predecessor"
```

`generate_bundle()` now refuses to replace an executable SDSC with an
`overlap_prefix` artifact unless `ifn_runtime_safe` is explicitly true.

## Deeptools findings

On the pod, the following experimental Deeptools patches were needed to push
the diagnostic through DXP codegen:

- allow mixed `datadscs_` + `dscs_` SDSCs with `coreIdToDscSchedule` in
  `dxp/SdscTree.cpp`;
- use the scheduled `datadscIdx` in the single-SDSC IFN PCFG branch;
- allow HBM-pinned tensors and one LX-neighbor tensor to coexist in scheduler
  analysis, while still running HBM sync insertion independently;
- allow double-buffering and input-neighbor fetch to coexist for the experiment;
- allow HBM-pinned non-neighbor tensors in the IFN DLDSC verifier;
- remove the hard `i/j` wrapper check so flash layouts can use the generic
  non-IJ comparator path.

After those patches, `dxp_standalone --bundle` succeeded for:

```text
/tmp/sdpa-stage038-ifn-legacydims-warp_overlap_probe-B1-H2-L128-D64-557610-204466
```

## Execution results

Failing diagnostic replacement:

```text
/tmp/sdpa-stage038-ifn-dxp-bundles-warp_overlap_probe-B1-H2-L128-D64-559324-848742
L=128 warp_overlap_probe status=timeout timeout_s=240
```

The generated debug schedule contained the expected IFN lowering:

```text
transfer_lds0_src:no_component_dst:no_component_lx_neighbor
sync_soft_send_l3lu_to_lxlu
sync_soft_receive_lxlu_from_l3lu
```

A no-soft-sync scheduler diagnostic also timed out under `timeout 180`; the
backtrace stopped in runtime artifact load / launch scheduling:

```text
flex::PfRuntimeScheduler::issueBarrier
spyre::getOrLoadArtifacts
spyre::launchKernel
```

Controls after the same Deeptools patch stack:

```text
vanilla
  status=ok max_err=0.00537109

manual non-overlap mixed execute tile 0
  status=ok max_err=0.0078125

fail-closed warp_overlap_probe
  status=ok max_err=0.00341796875
  cache=/tmp/sdpa-stage038-ifn-failclosed-warp_overlap_probe-B1-H2-L128-D64-562683-554746
```

## Conclusion

The single-SDSC IFN diagnostic is structurally close enough for DXP codegen, but
it is not a valid runtime replacement.  The likely root cause is semantic:
single-SDSC IFN creates input-neighbor fetch synchronization without a real
predecessor SDSC / producer relationship.  The same-LX dataop is not a real
producer, so runtime can wait on input-neighbor/barrier state that is never
satisfied.

The next value-correct route is the two-SDSC `mySDscPre` IFN path, where
Deeptools constructs the data movement from an actual predecessor output to the
consumer input.  Keep single-SDSC IFN fail-closed until that predecessor-backed
path executes standalone.

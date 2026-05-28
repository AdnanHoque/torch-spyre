# Stage073 - K/V HBM Prefetch Hoist Probe

## Goal

Move from the passing HBM-staged K/V hoist toward a warp-specialized-style
attention path: hoist the future low-core K/V producer, then prefetch that
future K/V HBM payload into the future batchmatmul's LX input while other work
is executing.

## Torch-side shape

- `SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_HOIST_TILE=-2` enables the
  prefetch-hoist candidate.
- The builder emits a hoisted future producer before the current tile.
- The overlap form replaces the current batchmatmul with a mixed SDSC that
  runs `STCDPOpHBM` prefetch dataops alongside the current compute.
- The future consumer is replaced by a compute-only SDSC whose K/V input is
  marked as prefilled LX.
- `SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_SERIAL=1` is a diagnostic mode:
  it fuses the HBM prefetch dataops into the future consumer and schedules them
  before the future batchmatmul. This isolates HBM dataop layout from
  current-tile overlap and cross-program LX lifetime.
- `SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_LX_BASE` is a diagnostic LX
  base override used to test address aliasing.

## Lower-stack fixes required to reach execution

- DcgManager now treats non-`STCDPOpLx` dataop plus DL-op schedule rows as
  independent overlap, not input-neighbor fetch.
- Stitcher permits duplicate units in a schedule row for this independent
  non-LX overlap case.
- Input-neighbor PCFG generation is guarded for non-input DSCs and invalid dims.
- PCFG generation uses the requested `datadscIdx` instead of assuming dataop 0.
- L3 scheduling can emit the required prefilled-LX marker for external LX
  inputs and relocate the existing LX allocate node near that marker.

## Evidence

The staged-hoist baseline remains value-correct:

- Variant: `onchip_hbm_kv_layout_xform_kv_hbm_staged_hoist_probe`
- Shape: B1 H8 L256 D64 block64
- Result: passed previously with max abs error about `0.00439453`.

The first prefilled-LX consumer attempt established that DDL needs an explicit
prefilled marker:

- Without the marker: DDL aborted with `Missing pre-filled transfer in schedule
  tree for this tensor-memory combination`.
- With the marker and allocation relocation: DXP/DDL compiled and executed.

Current prefetch-hoist results:

- v19, overlap with original LX base `278528`: value mismatch
  `2647 / 131072`, max abs `0.5986328125`.
- v20, forced LX base `1625344`: value mismatch worsened to
  `15056 / 131072`, max abs `2.189453125`. DXP also allocated the current
  tile's `lds1` at the forced prefetch base, proving the current-overlap SDSC
  aliases the prefetch dataop destination with the current compute input.
- v22, serial fused prefetch plus future consumer: value mismatch remained
  `2647 / 131072`, max abs `0.5986328125`. This removes the current-overlap
  alias and cross-program LX lifetime from the failure.
- v23-v27, consumer-shaped HBM chunks and STCDPOpHBM primary-DS alignment:
  still value mismatch `2649 / 131072`, max abs `0.5986328125`.

## Interpretation

The remaining prefetch failure is not primarily LX base selection, current-tile
clobbering, or cross-program LX persistence. Those were tested and either
changed the failure in a predictable aliasing way or did not change it.

The durable failure is the contract between an explicit `STCDPOpHBM` dataop and
the future batchmatmul's expected LX input layout. The passing HBM-staged path
uses DXP's native DL HBM input transfer for the batchmatmul. The failing
prefetch path tries to reproduce that transfer as standalone `STCDPOpHBM`
dataops, but the value pattern remains wrong even when those dataops are fused
directly before the future consumer compute.

## Next Direction

The next stack change should expose or synthesize the batchmatmul input's native
HBM-to-LX transfer as a hoistable prefetch primitive, instead of approximating
it from producer-side K/V pieces. In practice that likely means a DXP/DDL
contract for a dataop that reuses the DL input transfer's allocation, loop
offsets, and HBM addressing metadata, then marks the future DL input as
prefilled LX.

# InputFetchNeighbor mb/out probe artifact

Minimal artifact shape for `SPYRE_ONCHIP_MOVE_CARRIER=input_fetch_neighbor`.
`sdsc_1.json` contains the IFN trigger row `[[0, 0, 0, 0]]` on core 0 and logical `mb/out/x` piece metadata.
This directory is a compact unit-test fixture. The second-iteration pod backend
probe used full torch-spyre-generated SDSCs and is recorded in
`artifacts/input_fetch_neighbor_real_mb_out_iter2/`.

That real probe reaches Deeptools IFN but is blocked before value execution by
stock-helper assumptions around `DsTypes::INPUT`, LX pinning, legacy
`coreStateInit_`, loop-order metadata, and bundle `datadscs_` handling.

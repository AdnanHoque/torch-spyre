# Stage 218: Large PT-LX Size Check

## Summary

Extended the `computed_transpose_adds_then_matmul_tuple` benchmark to `8192x8192`
and `16384x16384`. These are square fp16 tensors. The run used fewer timing
iterations because `16384x16384` makes each restickified tensor about 512 MiB.

Artifacts were copied locally under:

```text
artifacts/stage217_ptlx_large/pod
```

## Results

| Size | Stock ms | Stage3B ms | PT-LX ms | PT-LX vs Stock | PT-LX audit |
|---:|---:|---:|---:|---:|---|
| 8192 | 70.734 | 70.705 | 70.560 | 1.002x | skipped |
| 16384 | 317.019 | 317.057 | 317.150 | 1.000x | skipped |

PT-LX audit for both sizes:

```text
producer-endpoint-not-allocator-backed:prototype-default
```

## Interpretation

These large-size numbers are not PT-LX speedups. The PT-LX mixed schedule flag
was enabled, but the compiler did not replace `ReStickifyOpHBM` with the mixed
`ReStickifyOpWithPTLx + STCDPOpLx` artifact. It skipped before replacement and
used the stock path.

The only current executable PT-LX win in this square family remains `2048x2048`.
That size is special because the full-bridge prototype can reserve allocator-
backed LX endpoints and an intermediate per core. Larger shapes require the
streaming tiled PT-LX lowering from Stage216; the planner exists, but lowering
does not yet emit the tile stream.

## 2048 LX Verification

A codegen-only verification run for `2048` generated:

```text
sdsc_1_MixedReStickifyOpWithPTLxConsumer.json
```

and no `sdsc_1_ReStickifyOpHBM.json` for the restickify boundary. The mixed SDSC
contains:

```text
ReStickifyOpWithPTLx
STCDPOpLx
```

The audit row reported:

- `status=patched`
- `replacement_sdsc=1_MixedReStickifyOpWithPTLxConsumer`
- producer endpoint in LX at `0..262144`
- consumer endpoint in LX at `262144..524288`
- intermediate in LX at `524288..786432`
- `value_flow_contract.valid=true`

That is codegen evidence that `2048` is using the PT-LX path. It is still not a
fabric-counter proof of RIU traffic; that would require working HBM/RIU counters.

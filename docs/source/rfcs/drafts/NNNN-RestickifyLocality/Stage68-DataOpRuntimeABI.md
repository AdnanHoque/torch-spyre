# Stage 68: Data-Op Runtime ABI Probe

## Summary

Stage 68 followed up on Stage 67 by asking a narrower question:

> Can we reuse a known-good stock Torch-Spyre/DXP runtime bundle as the argument
> binding harness, but swap in a Deeprt-generated data-op program?

Result: no. The swapped bundle still launches, but the output tensor remains
zero.

This strengthens the Stage 67 conclusion. The blocker is not merely that our
handmade runtime directory was missing a few obvious files. The Deeprt-exported
standalone data-op program is not compatible with the current graph-free
Torch-Spyre/Flex runtime ABI by simply dropping its `init.txt` into a DXP bundle.

## Why This Stage

Stage 67 proved:

- stock DXP-generated `ReStickifyOpHBM` binds runtime tensor arguments and writes
  visible output;
- standalone Deeprt data-op exports launch but do not write visible output;
- LX-only data-op LDS records are not runtime-rebased by just changing
  `segment_`.

One ambiguity remained: maybe the Stage 67 handmade package was too thin. A real
DXP bundle contains files like:

```text
bundle.mlir
execute/<kernel>/pagi.json
execute_dsg.txt
loadmodel_to_device_dsg.txt
loadmodel_to_spad_dsg.txt
loadprogram_to_device/<kernel>-SenProgSend/init.txt
loadprogram_to_device_dsg.txt
loadprogram_to_spad_dsg.txt
sdsc_0_ReStickifyOpHBM.json
segment_size.json
```

So Stage 68 reused exactly that known-good bundle shape.

## Stock Harness Control

The stock captured bundle came from the `isolated_transpose_contiguous_128`
probe:

```text
/tmp/torchinductor_1000800000/tmp134eltm5/inductor-spyre/
  sdsc_fused_clone_t_0_4smvcppm/
```

Launching the stock bundle with argument order `[input, output]` writes visible
transposed values:

```json
{
  "order": "in_out",
  "nonzero": 16383,
  "maxdiff_transpose": 8.0,
  "first8": [0.0, 128.0, 256.0, 384.0, 512.0, 640.0, 768.0, 896.0]
}
```

This confirms the harness binds runtime tensor arguments.

## Data-Op Program Swap

I generated a same-shape `128x128` fp16 `STCDPOpHBM` data-op control from the
prototype data-op generator:

```text
input segment_ = "input"
output segment_ = "output"
PlacementInfo = LX + HBM
hbmSize_ = 256 sticks
coreIDtoANInfo = p1 -> p1
```

Deeprt exported a program:

```text
execute/0_STCDPOpHBM/init.txt      2313 bytes
execute/0_STCDPOpHBM/sdsc.json    28637 bytes
execute/0_STCDPOpHBM/senprog.txt   3078 bytes
execute/0_STCDPOpHBM/smc.txt       1359 bytes
```

The generated `senprog.txt` is not empty. It has L3 and LX activity:

| Term | Count |
|---|---:|
| `L3LU` | 5 |
| `L3SU` | 5 |
| `LXLU` | 2 |
| `LXSU` | 2 |

Representative SMC:

```text
L3_LDMU ...
L3_STMU ...
LX_LDSTIU ...
LX_LDSTU ...
```

Then I copied the known-good stock bundle and replaced only:

```text
loadprogram_to_device/<new-bundle-name>-SenProgSend/init.txt
sdsc_0_ReStickifyOpHBM.json
```

The rest of the stock DSG/segment files were preserved.

Runtime result:

```json
{
  "order": "in_out",
  "nonzero": 0,
  "maxdiff_input": 16384.0,
  "maxdiff_transpose": 16384.0,
  "first8": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

Swapping argument order also left output zero.

## Interpretation

This means the missing piece is deeper than the presence of `execute_dsg.txt` or
`segment_size.json` next to `bundle.mlir`.

The data-op program's generated registers show baked local addresses:

```text
L3LU-LAR: 0
L3SU-LBR: 8192
LXSU R0: 1048320
```

Those are not being patched to the runtime tensor composite addresses passed to
`RuntimeOperationCompute`.

The Spyre KB supports this interpretation. The runtime-facing artifact contract
is SpyreCode/job-plan metadata, not just a SuperDSC JSON plus an init binary.
The runtime needs:

- memory segment layout;
- per-operation DMA/control-block descriptors;
- program frame pointers;
- host correction metadata when addresses or loop bounds are runtime-dependent.

Our Deeprt standalone export gives a program binary, but not the full
Torch-Spyre/Flex ABI metadata needed to bind PyTorch tensor arguments into that
program.

## Deeprt Compute-Op Check

I also tried using the same one-node Deeprt harness on the stock
`ReStickifyOpHBM` compute-op SDSC.

That did not get far enough to test runtime binding:

```text
input_dataops=0 input_dldscs=1 cores=4
Calling DL Scheduler for node: 0_ReStickifyOpHBM
...
terminate called after throwing an instance of 'std::out_of_range'
what(): map::at
```

So the current throwaway harness remains useful for data-op export probing, but
it is not a general replacement for the full DXP/Torch-Spyre compile path.

## Conclusion

Stage 68 does not prove LX-to-LX restickify execution.

It does prove that the next blocker is the runtime ABI between generated
programs and live tensor arguments:

```text
DXP-produced compute-op bundle:
  binds runtime tensors -> writes output

Deeprt-produced standalone data-op init in stock DXP harness:
  launches -> does not bind/write output
```

The in-graph proof path should therefore not continue by swapping raw
`init.txt` files. We need one of:

1. a true Deeprt/SpyreCode export path that includes the job execution plan,
   program frame pointers, and tensor segment binding metadata; or
2. a DXP-accepted DLDSc/DDL representation that stays inside the normal
   Torch-Spyre bundle ABI; or
3. a full compiled graph integration where the data-op consumes a producer-owned
   intermediate already allocated by the compiler, rather than external PyTorch
   tensor arguments.

Option 2 is closest to our existing Torch-Spyre branch work, but the DDL bridge
still needs the source-side LX read contract fixed. Option 1 is likely the
cleaner long-term runtime path if Deeptools already has a supported SpyreCode
export API for data-op graphs.

## Recommended Next Step

Return to the DLDSc/DDL route, not the standalone data-op runtime swap.

The concrete next experiment should be:

1. start from the Stage 62 `computed_transpose_adds_then_matmul` fixture because
   it has one clean `in_graph_computed` restickify;
2. restore an explicit source-side LX allocation/read in the bridge template;
3. compare generated `senprog.txt` against the data-op Stage3B reference and
   require both `LXLU` and `LXSU`;
4. only then rerun hardware.

The acceptance target for the next stage is not speedup. It is:

```text
one in-graph restickify bundle
  + no HBM/L3 in the restickify generated program
  + explicit LXLU and LXSU activity
  + hardware retires
  + output matches stock HBM restickify
```


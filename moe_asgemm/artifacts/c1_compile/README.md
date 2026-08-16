# C1 expert HBM affine advance: backend compile accepted

Status: **accepted compile-only boundary; no generated kernel launched and no
timing collected**.

This artifact fixes the exact address-binding defect found by the preserved
device failure in
`../flat-e2-t64-unit-sum-collapse-device-correctness-blocked-15`: its E=2
loop executed twice but reused expert 0's gate/up/down weights and alpha on
both iterations, producing approximately `2 * expert0`.

## Narrow source fix

`SpyreKernel._general_tile_advance` already preserved the squeezed unit-E
advance and placed it on each expert-copy TensorArg as
`floor(64 * _tile_adv_<copy>_lvl0)`. The first and only loss was
`SpyreKernel.create_op_spec`: it serialized a level symbol only when the raw
output/reduction tiled-dim list was nonempty. Unit E had already been divided
to one and squeezed, so the copy OpSpecs emitted `tiled_symbols=[[]]` and the
bundle could not build an affine address.

The fix:

- treats a real nonzero TensorArg `device_tile_advance_expr` as authoritative;
- detects plain and floor-wrapped coefficients with `coeff_through_floor`;
- serializes the already-minted level symbol without fabricating a tiled LX
  output dimension;
- derives nesting depth from `loop_count` as well as the raw tiled-dim lists;
  and
- fails closed if an emitted level symbol has no matching loop trip count.

The lane-specific compiler/test delta is preserved verbatim in
`source-delta.patch`. No custom DDL, fused epilogue, kernel launch, timing,
commit, or push was used.

## Host tests

Focused tests covering the new helper, floor-wrapped and plain coefficients,
literal-empty raw tiled lists, invariant arguments, missing trip counts, the
existing preserved-unit-E read, and bundle affine machinery passed:

```text
20 passed, 288 deselected
```

The complete coarse-tiling test file passed in the exact pod environment:

```text
307 passed, 1 skipped
```

The independent bundle checker's self-test passes and rejects seven mutated
negative fixtures. It also rejects the preserved broken bundle because that
bundle has zero affine maps.

## Exact emitted bundle

The real backend compiler accepted the fresh E=2, T=64, H=64, F=64, C=1
bundle. `prepare_kernel` and `launch_jobplan` were mocked; the generated runner
was never launched.

The final MLIR contains exactly one deduplicated map:

```mlir
#map_0 = affine_map<(d0)[s0] -> (s0 + 128*d0)>
```

Inside the sole expert loop, exactly four HBM operands advance:

```text
arg2 (gate weights) -> addr0 -> sdsc2
arg3 (up weights)   -> addr1 -> sdsc5
arg4 (down weights) -> addr2 -> sdsc8
arg5 (runtime alpha)-> addr3 -> sdsc10
```

The 128-byte stride is exact: the stickified `[1,64,2,64]` layout interleaves
the E planes, so one expert step is 64 FP16 device elements. Each weight expert
still touches 64 sticks (8192 bytes) per iteration; alpha is physically
stick-padded in the same way.

The strict checker also proves:

- X arg0 is used only by the preheader `sdsc0`;
- accumulator initializer arg1 is used only by preheader `sdsc1`;
- output arg6 is used only by post-loop `sdsc14`;
- every other in-loop SDSC is LX-only and has no bundle operand;
- `sdsc12` is the identity expert contribution;
- `sdsc13` is the loop-carried accumulator add;
- zero local sum OpSpecs/SDSCs;
- zero restickify and HBM-pool operations; and
- one final HBM output.

## Environment and attempts

- Pod: `adnan-cdx-spyre-dev-pf`.
- PCI: `0000:ac:00.0`.
- Isolated source: `/tmp/unit-reduction-affine-fix-host-20260816`.
- Detached base: `65508a025f557663c5694e3596c49b814d87517a`.
- Successful remote output:
  `/tmp/unit-reduction-affine-fix-backend-compile-c1-19`.
- Successful remote compiler cache:
  `/tmp/unit-reduction-affine-fix-cache-backend-c1-19`.

Attempt 18 stopped before the backend compiler because `dxp_standalone` was
not on `PATH`; no DXP, prepare, or launch occurred. Attempt 19 was the single
corrected infrastructure retry, adding the exact DeepTools bin directory and
an explicit fresh `TORCHINDUCTOR_CACHE_DIR`.

## Identities

```text
torch_spyre/_inductor/spyre_kernel.py
  f4753b52356a0e84196516f713eaa4fcd4c6d8a959ef69a50cc1db13c8257d65
tests/inductor/test_coarse_tiling.py
  948b8cf715ef485902052adff453e349d31c81c06f6c447fa57edca4931e6625
14-file tracked production diff stream
  420ef37553f2167f48bebc20800a7596d315a7a299b51a73e0aa049b6b8af32a
source-delta.patch
  9983db28d0892bb1bc6d3f6e96b597aa8a402fe6e6ba199c8fcf68d36d6cd771
experiments/dasx_flat_e2_t64_compile_probe.py
  ca17f3c3db1906d7ee0d033173d8e093a6cf74cd80f786a3af19dd686d7b9f46
experiments/check_c1_bundle_affine.py
  c0974121c76fbd56643bed1d88d01ea656b886e9f90563e6819564fa23c788cb
generated_module.py
  ee8c3ba4e69ba246c5c8ecb1aa79cea389ebafb09a4652192b80398b5de93bad
compile_result.json
  2396680ba99f03eec1df87a381fece481f3acf5ff5973197ab77b63cd6183303
backend-compile-output/bundle.mlir
  977d62e74ca5f4a3ff28a89be32e4531ecb858828b77cedbf21e007c05c0c720
backend-compile-output/sdsc_12.json
  0f49a7b0c5c87f4b7e4b1d1a55222e820e349cf84a7979bdd3bb4c1935e0846d
backend-compile-output/sdsc_13.json
  e3ed4deb7eb98f771686a916a7212145a7ca55d48ada8d85cfc06d20decea128
backend-compile-output/spyreCodeDir/spyrecode.json
  568973ef676601aa9c7d1885f75348798372fe6b2f004be2816bdd85799a92bb
backend-compile-output/spyreCodeDir/init_binary.bin
  2869433c9769bb6dc4103e9c284a9cecc25d56dbfea314be1aeedc84a0be2a34
```

## Next gate

The next authorized step is one untimed C1 numerical correctness rerun using
the unchanged two-alpha harness and this exact source snapshot. It must reuse
one compiled callable and one bundle, compare both outputs and their delta to
fresh FP32 references, and rerun this exact structural checker on that same
bundle. Timing and C32 remain out of scope.

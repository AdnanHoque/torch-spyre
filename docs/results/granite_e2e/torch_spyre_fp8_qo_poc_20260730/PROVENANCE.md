# Provenance

## Results branch

```text
repository: https://github.com/AdnanHoque/torch-spyre.git
branch:     ah/fp8
base before this handoff: 71a99a070b07317e15339052aadc429f489cec23
```

## Experimental Torch-Spyre source

```text
baseline:   PR #2286 head
commit:     a01c627d57ba18bc442d8b5f73086b2778fdc9d4
local tree: /Users/adnan/Documents/Codex/2026-07-28-handoff-sendnn-fp8-matmul-on-spyre/torch-spyre-pr2286-worktree
remote:     /home/adnan/codex-isolated/torch_spyre_pr2286_a01c627d_20260729
diff:       16 files, 502 insertions, 112 deletions
```

The source changes are archived as a patch rather than applied to the results
branch because the two trees do not share a suitable source base.

## DeepTools source

```text
commit: ee2f97a86c609eeb20ea3ad2d48040259d67ded3
source: /home/adnan/codex-isolated/torch_spyre_fp8_deeptools_ee2f97a_20260729
build:  /home/adnan/codex-isolated/torch_spyre_fp8_deeptools_ee2f97a_build_20260729
```

The DeepTools patch was normalized against that exact commit. Its old and new
blob hashes are `7ae0fd3366056b1fb687a39ceda53d92a56fdd9c` and
`446443576a02c199714b67f3df8347e2c311df1a`; `git apply --check` was verified
against the exact commit.

## Pinned device stack

```text
environment: /home/adnan/spyre-envs/image-913f394b4b3f/activate.sh
torch:       2.11.0+aiu.kineto.1.1.2
DeepTools:   +1401 (ee2f97a)
Flex:        +388 (81385a4)
Senlib DD2:  +194 (951e4c4)
cores:       32
corelets:    2
```

The PoC is DD2-only. No 1p5 source, stack, or device result is part of this
package.

## Retrieved remote evidence

OpenShift access was restored on 2026-07-30. The successful FP16 result, the
restricted forced-1x32 raw FP8 result, the pre-corelet-PoC QFP8MB bundle and
SuperDSC, and the post-corelet-PoC compiler log were copied from
`adnan-clc-spyre-dev-pf` into this package. Their original paths were:

```text
FP16 result:
/home/adnan/codex-isolated/torch_spyre_pr2286_a01c627d_20260729/smoke/fp16_m512/result.json

forced 1x32 raw FP8 result:
/home/adnan/codex-isolated/torch_spyre_pr2286_a01c627d_20260729/smoke/fp8_raw_optimized_m512_1x32/result.json

QFP8MB v3 artifact root:
/home/adnan/codex-isolated/torch_spyre_pr2286_a01c627d_20260729/cache_qfp8mb_raw_opt_m512_8x4_v3/inductor-spyre/6bacf05f_sdsc_fused__scaled_mm_quantize_fp8_with_scale_quantize_weight_fp8_with_scale_0_9kzd5b6q

corelet diagnostic log:
/home/adnan/codex-isolated/torch_spyre_pr2286_a01c627d_20260729/smoke/dxp_v3_corelet_preload.log
```

The complete remote cache was not copied. Only the small files needed to
support the stated observations and compiler boundary are archived.

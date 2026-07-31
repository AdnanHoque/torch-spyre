# Fast global-P09 realized WD-chain ledger

Audit date: 2026-07-28
Scope: existing source and artifacts only; no compilation, device launch, timing,
or checkout modification.

Audited treatment:

`/home/adnan/claude-isolated/global_p09_integration_20260728/runs/global_p09_matched_treatment_5x`

This is the treatment identified by the coordinator as the 238.480700 ms run.
That timing is not re-derived here. This audit answers only which boundaries
are present in the actual emitted program.

## Reading the table

- `mb16/even16` means sixteen 32-token shards owned by cores
  `0,2,...,30`.
- `mb8/group4` means eight 64-token cohorts, each replicated to
  `group4(m)={4m,4m+1,4m+2,4m+3}`.
- `mb8 x out4` and `mb8 x in4` use `core=4*mb+out_or_in`.
- `token32` or `mb32` means one token shard per core, `core=token_or_mb`.
- Payload is reported as
  `entries/deliveries/remote-deliveries; total logical bytes/remote logical
  bytes`.
- Every listed Torch shuffle has both input and output allocated with
  `component_="lx"` in its emitted bundle SDSC and an
  `STCDP_FINAL_BEGIN ... LxRelayout` block in `run.log`.

## Per-realized-boundary chain ledger

| Boundary | SenDNN producer WD/owners -> relayout -> consumer WD/owners | Actual Torch producer WD/owners -> emitted payload -> consumer WD/owners | Class |
| --- | --- | --- | --- |
| P09/Q | Initial LayerNorm output `mb16/even16` -> grouped all-gather to `mb8/group4` -> Q BMM `mb8 x out4`. | `buf10`, `mb16/even16` -> layer `9_shuffle`, `16/64/48; 16 MiB/12 MiB` -> `buf11`, `10_batchmatmul`, `mb8 x out4`. | **Exact** |
| P09/K | Same P09 source and destination ownership -> K BMM `mb8 x out4`. | `buf10`, `mb16/even16` -> layer `14_shuffle`, `16/64/48; 16 MiB/12 MiB` -> `buf15`, `15_batchmatmul`, `mb8 x out4`. | **Exact** |
| P06 | Q/rotary `token8 x head4`, `core=4*t+h` -> all-to-all to `token32` -> QK BMM `token32`. | `buf14`, decomposed rotary view `y8 x x4`, `core=4*y+x` -> layer `20_shuffle`, `128/128/96; 4 MiB/3 MiB` -> `buf20`, `21_batchmatmul`, `token32`. Torch also needs producer-completion transfers for its decomposed rotary path. | **Compatible transformed** |
| P09/V | Same P09 source and destination ownership -> V BMM `mb8 x out4`. | `buf10`, `mb16/even16` -> layer `30_shuffle`, `16/64/48; 16 MiB/12 MiB` -> `buf29`, `31_batchmatmul`, `mb8 x out4`. | **Exact** |
| Torch post-attention bridge | No literal SenDNN P01-P14 LX relayout at this semantic boundary. Exact P08 is a different 1 MiB K/restickify edge upstream of nonlinear attention. | `buf40`, 4 MiB post-attention value, `token32` -> layer `41_shuffle`, `128/128/96; 4 MiB/3 MiB` -> `buf41`, `42_identity`, `token8 x head4`, `core=4*t+h`. | **Torch-specific; mismatch to literal P08** |
| P04 | Attention output `mb32` -> grouped all-gather to `mb8/group4` -> output BMM `mb8 x out4`. | `buf43`, `mb32` -> layer `45_shuffle`, `48/128/96; 16 MiB/12 MiB` -> `buf44`, `46_batchmatmul`, `mb8 x out4`. | **Exact** |
| P12 | Pointwise-mul output `mb16/even16` -> all-to-all to `mb8 x out4` -> residual add `mb8 x out4`. | `buf45`, `mb16/even16` -> layer `48_shuffle`, `64/64/48; 4 MiB/3 MiB` -> `buf46`, `49_add`, `mb8 x out4`. | **Exact** |
| P03 / FFN gate | LayerNorm output `mb8 x in4` -> per-cohort grouped all-gather to `mb8/group4` -> gate BMM `mb8 x out4`. | `buf52`, `mb8 x in4` -> layer `56_shuffle`, `48/128/96; 16 MiB/12 MiB` -> `buf53`, `57_batchmatmul`, `mb8 x out4`. | **Exact** |
| P03 / FFN up | Same P03 source and destination ownership -> up BMM `mb8 x out4`. | `buf52`, `mb8 x in4` -> layer `59_shuffle`, `48/128/96; 16 MiB/12 MiB` -> `buf55`, `60_batchmatmul`, `mb8 x out4`. | **Exact** |
| Torch MLP down-projection bridge | SenDNN has no LX relayout here: the corresponding `mul_10_out -> mm_6-BMM_1` handoff is HBM-backed and both operations use `mb8 x out4`. | `buf56`, SwiGLU product `mb32` -> layer `62_shuffle`, `32/64/62; 25 MiB/24.21875 MiB` -> `buf57`, `63_batchmatmul`, `mb16 x out2`, with cohort `m` replicated to cores `m,m+16`. | **Torch-specific; no literal template match** |
| P14 | Final norm `mb8 x out4`; only last-token cohort fragments on cores 28-31 participate -> sparse all-to-all -> last-token hidden vector split 32 ways. | `buf5`, last cohort on cores 28-31 -> final `7_shuffle`, `32/32/31; 8 KiB/7.75 KiB` -> `buf6`, `8_identity`, hidden32 on cores 0-31. | **Exact** |
| P13 | Last-token hidden vector, hidden32 on cores 0-31 -> subset all-gather of the full vector -> LM-head input replicated on output-owner cores 0-27. | `buf6`, hidden32 -> final `9_shuffle`, `32/896/868; 224 KiB/217 KiB` -> `buf7`, `10_batchmatmul`, full input on cores 0-27 and `out28`. | **Exact** |

## Coverage and gaps

The actual fast program has **12 distinct static emitted LX relayout sites**:
ten in the transformer-layer bundle and two in the final bundle.

- Literal SenDNN template sites: 10 total.
  - Exact: 9 sites across P03 (two), P04, P09 (three), P12, P13, and P14.
  - Compatible transformed: one P06 site.
- Torch-specific sites: two, the post-attention bridge (`41_shuffle`) and the
  MLP down-projection bridge (`62_shuffle`).
- Template-class coverage is **7/14**:
  - exact: P03, P04, P09, P12, P13, P14;
  - compatible transformed: P06;
  - unrealized/mismatch: P01, P02, P05, P07, P08, P10, P11.

P09 is the only classification change from the corrected accepted ~245.975 ms
ledger: it moves from mismatch to exact and contributes three actual emitted
sites. It does not change the prior negative classifications:

- P01 has no K-to-QK full-gather payload.
- P02's exact `buf29 -> buf31` V gather is still only a plan; no corresponding
  emitted site exists.
- Exact P08 `buf18 -> buf66` is still only a plan; layer `41_shuffle` is a
  different downstream tensor and is not P08.
- P05/P07/P10/P11 still have no emitted matching boundary.

Therefore `contract.txt` configuration names are not used as realization
evidence.

## Artifact proof and the duplicate site-9 name

The runtime log contains twelve `STCDP_FINAL_BEGIN` blocks in this order:

`layer 9,14,20,30,41,45,48,56,59,62; final 7,9`.

Layer-Q P09 and final P13 both have logical site number 9. The run-directory
debug JSON name is consequently last-writer-collided, but the realization is
not ambiguous:

- the first `run.log` site-9 block has 16 entries and 64 deliveries, matching
  P09/Q;
- the later site-9 block has 32 entries and 896 deliveries, matching P13;
- the layer bundle's `sdsc_9.json` has LX input address 1245184,
  `mb16/even16` source ownership, and LX output address 0 with
  `mb8/group4` ownership;
- the final bundle's `sdsc_9.json` has LX input address 8192, hidden32 source
  ownership, and LX output address 0 replicated on cores 0-27.

Primary evidence:

- `run.log`
- `relayout_plans.jsonl`
- layer bundle:
  `cache/inductor-spyre/83327de4_sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_sum_transpose_unsqueeze_view_0_z5krvabv`
- final bundle:
  `cache/inductor-spyre/55b9ce38_sdsc_fused_clone_div_linear_rms_norm_slice_0_xxgktwch`
- finalized debug payloads:
  `stcdp_after_pcfg_*_shuffle-Relayout.json`

SenDNN geometry is from:

- `/home/adnan/claude-isolated/granite_parity_20260727/torch-spyre/experiments/granite_relayout/artifacts/catalogs/prefill_relayout_templates.json`
- `/home/adnan/claude-isolated/granite_parity_20260727/torch-spyre/experiments/granite_relayout/artifacts/catalogs/sendnn_sdsc_lx_replay_manifest.json`

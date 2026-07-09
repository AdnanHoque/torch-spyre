# SDPA edge classification

Shape: Q `[1,4,512,128]`, K/V `[1,4,4096,128]`.

Only the first generated SDSC bundle is analyzed. Each benchmark generated five
structurally repeated bundles.

The attention dataflow is:

```text
Q_scaled = scale(Q)
K_scaled = scale(K)
S        = Q_scaled @ K_scaled.T
P        = softmax(S)
C        = P @ V
output   = identity(C)
```

## Classification table

`Contract fired` means Torch emitted or encoded the required tensor-distribution
contract. `On-chip result observed` means the generated SDSCs keep that edge in
LX and expose the expected layout operation. Torch's explicit PR2 metadata uses
`realized: false` because Deeptools, not Torch, realizes the physical transfers.

| Edge | Baseline residency | Required communication | Layout transform | Contract fired | On-chip result observed | Artifact evidence |
|---|---|---|---|---|---|---|
| `Q_scaled -> S` (score-matmul LHS) | LX -> LX | No new PR2 collective observed | Same stick layout | No PR2 classification | Already LX in both variants | `sdsc_0` output and `sdsc_3` input remain LX |
| `K_scaled -> S` (score-matmul RHS) | LX -> HBM restickify -> HBM | `all_gather` (`all_gather_replicate`) | `stick_relayout` into the matmul operand layout | **Yes**, explicit `sdsc_3.lxRelayoutClassifications_` entry | **Yes** | `sdsc_2 ReStickifyOpHBM` becomes `ReStickifyOpLx`; `sdsc_3` RHS becomes `1_lx` |
| `S -> softmax` | HBM -> HBM | Not a core-ownership relayout; this is an LX residency/capacity problem | None classified | No | No | `sdsc_3` output and `sdsc_4`/`sdsc_5` inputs remain HBM |
| `P -> C` (value-matmul LHS) | LX -> LX | No new collective required by this run | None | No new classification | Already LX in both variants | `sdsc_8` output feeds `sdsc_9` input in LX |
| `C -> identity` | HBM -> HBM | `all_to_all_shuffle` through the PR1 DLDSC tensor-vs-compute distribution contract | Same-stick ownership reassignment | **Yes**, encoded in allocation vs compute coordinates rather than a PR2 classification row | **Yes** | `sdsc_9` output and `sdsc_10` input change from HBM to LX |
| `identity(C) -> graph output` | HBM output | Required graph-boundary store, not an intermediate relayout | N/A | N/A | Intentionally remains HBM | `sdsc_10` output is HBM in both variants |

## Aggressive LX eligibility

The isolated `allow_all_ops_in_lx_planning=True` variant preserves the PR2
all-gather behavior and additionally keeps the score tensor resident:

| Edge | Narrow relayout variant | Aggressive variant | Additional mechanism | Fired? |
|---|---|---|---|---|
| `K_scaled -> S` | `ReStickifyOpLx`, matmul RHS in LX | Same | PR2 `all_gather + stick_relayout` | Yes in both |
| `S -> max` | score output/input in HBM | score output/input in LX | Expanded LX output eligibility | **Yes** |
| `S -> sub` | score input in HBM | score input in LX | Existing same-view LX persistence after pinning `S` | **Yes** |
| `P -> C` | LX | LX | Existing LX persistence | Already active |
| `C -> identity` | LX | LX | PR1 tensor-vs-compute distribution contract | Yes in both |

The additional HBM round trip removed is:

```text
before: score matmul -> S(HBM) -> max/sub
after:  score matmul -> S(LX)  -> max/sub
```

This changes kernel time from `0.672 ms` to `0.301 ms`. It is a separate
residency optimization, not another communication collective.

## Explicit PR2 contract

The `K_scaled -> S` row is the PR2 all-gather case. The enabled `sdsc_3.json`
records:

| Field | Value |
|---|---|
| `communication_class` | `all_gather` |
| `communication_pattern` | `all_gather_replicate` |
| `materialization_pattern` | `all_gather_replicate_with_layout_conversion` |
| Producer work slices | 4 x 8 |
| Consumer tensor work slices | 4 x 1, replicated across consumer groups |
| `max_fanout` / `max_fanin` | 8 / 8 |
| Logical transfer count | 256 |
| Movement stage | `all_gather_replicate` |
| Conversion stage | `local_restickify_to_kernel` |
| Carrier hint | `lx_all_gather_then_local_restickify` |

The HBM round trip removed by PR2 is therefore:

```text
before: K_scaled(LX) -> ReStickifyOpHBM -> K_operand(HBM) -> score matmul
after:  K_scaled(LX) -> ReStickifyOpLx  -> K_operand(LX)  -> score matmul
```

## Summary artifacts

- `baseline_first_iteration_summarize_sdsc.md`: Jamie-style baseline table.
- `baseline_first_iteration_summarize_sdsc.log`: baseline summarizer stdout.
- `relayout_first_iteration_summarize_sdsc.md`: Jamie-style enabled table.
- `relayout_first_iteration_summarize_sdsc.log`: enabled summarizer stdout.
- `aggressive_first_iteration_summarize_sdsc.md`: aggressive-LX table.
- `aggressive_first_iteration_summarize_sdsc.log`: aggressive-LX summarizer stdout.

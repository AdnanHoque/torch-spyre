# Granite non-attention accepted-stack gate

Snapshot: 2026-07-26

## Result

Attention remains frozen. The accepted non-attention prefill stack is now a
self-contained, hunk-minimal source set at the pinned heads:

- Torch-Spyre: `59545440f0e7091ff1b2f90df63580da1842f3fe`
- FMS: `61bc991b175103e80cb8202b24a66ba7dbe79d1b`
- DeepTools: `406142afb9f080b9271e7c565a757ab8d8b5ed8f`
- LX contract: `DXP_LX_FRAC_AVAIL=0.2`

P12, P13, and P14 remain promoted. P03 and P05 remain rejected by full-model
bit-exactness, and P10/P11 remain dependent on the rejected P05 RMSNorm chain.
No attention edge, commit, push, pull request, merge, or new device bracket was
introduced by this minimization pass.

## Minimal artifacts

| Component | Artifact | SHA-256 | Scope |
| --- | --- | --- | --- |
| Torch | `p12_p13_p14_torch_hunk_minimal.patch` | `b127a68ecbe3db3165acfe2e390975634de4976c0b5e1bc473d21e0e5948feed` | Seven files; 924 insertions, 14 deletions |
| FMS | `p14_fms_hunk_minimal.patch` | `befc6c2ec58be5073e7c5fc97e5204c6437dbd6fc7fa3ae2d1d946f8d91404a5` | Four files; P13/P14 final-stage policy and padded vocabulary |
| DeepTools | `p14_deeptools_hunk_minimal.patch` | `bcd2f7ef2a531cac3cd437667c9e1ff971b0a584bedc0eb4051817ddc0b570e9` | One file; sparse destination-owner/full-participant-union handling |

P12 needs no additional FMS or DeepTools hunk. Its rational destination sizing
and exact sparse repartition are in the Torch patch; the one-file DeepTools
participant-union logic retained for P13/P14 is also compatible with P12.

The Torch patch contains only:

- P12's 16 logical token shards, even-core physical owners, 8x4 residual-add
  destination, and exact `S2:S1 = 1:2` allocation;
- P13's 32-source to 28-owner last-token LM-head all-gather;
- P14's final-token last-cohort to 32-hidden-shard repartition;
- generic dense participant-union and rational allocation support required by
  those exact graph gates;
- ten focused P12/P13/P14 tests plus the pre-existing generic relayout tests.

## Source validation

- Torch patch clean apply-check at `59545440`: pass.
- `git diff --check`: pass.
- Python syntax compilation of every touched Torch file: pass.
- Complete touched Torch test file on `adnan-clc-spyre-dev-pf`:
  `25 passed in 0.67s`.
- P14-minimal FMS focused tests: `3 passed`.
- P14-minimal DeepTools C++17 translation-unit syntax compile: pass.

The P12 additions are explicitly covered by:

1. exact source and destination work divisions;
2. sparse-repartition classification with a `1/2` destination ratio;
3. compiled 16-even-owner to 32-destination piece maps;
4. allocator proof that a 256 KiB S1 produces a 128 KiB S2.

## Authoritative device evidence

The already-completed clean combined T-C-T-C bracket remains authoritative:

- zero mismatches across 887,040 aligned FP16 logit comparisons;
- treatment midpoint kernel sum: `387.619 ms`;
- control midpoint kernel sum: `391.371 ms`;
- midpoint saving: `3.752 ms` (`0.959%`);
- byte-identical treatment payloads for P12, P14, and P13;
- every claimed source and destination endpoint is LX, with no HBM endpoint.

The minimization pass did not rebuild DeepTools or rerun the model. It removed
unrelated/diagnostic hunks while retaining the logic already exercised by the
combined bracket. A release build should rebuild the one-file DeepTools patch
before calling the binary itself hunk-minimal.

## Non-attention disposition

| Edge | Decision | Reason |
| --- | --- | --- |
| P03, RMSNorm output to MLP gate/up | Reject | Large modeled win, but full-model logits drift; work/reduction order changed |
| P05, residual to RMSNorm reduction | Reject | One-layer exact and faster, but isolated full-40 logits drift |
| P10/P11, RMSNorm scalar/scale remaps | Do not promote | Not independent of the rejected P05 arithmetic chain |
| P12, final prefill residual handoff | Promote | Clean bit-exact T-C-T-C, positive timing, LX-only emitted transport |
| P13, last-token vector to LM head | Promote | Clean bit-exact T-C-T-C, 28-owner targeted speedup, decoded LX transport |
| P14, final-token permutation/slice | Promote | Clean bit-exact T-C-T-C, targeted speedup, emitted LX-only transport |
| Decode SwiGLU to down projection | Active next seam | Compact/broadcast-row compile path exists, but physical allocator, post-DXP transport, steady-decode, and full-model multi-token gates remain |

P04 and P01/P02/P06/P08, plus the attention/KV/softmax decode families, remain
parked with attention.

## Remaining parity gap

The accepted non-attention stack is correct but is not close to the SenDNN
prefill gate by itself. Its `387.619 ms` treatment midpoint remains
`195.309 ms` above the `192.310 ms` gate. The next independent non-attention
work is therefore decode SwiGLU-to-down-projection transport and phase-level
submission/lifetime scope, not another replay of P03/P05-style work-division
changes that already failed full-model exactness.


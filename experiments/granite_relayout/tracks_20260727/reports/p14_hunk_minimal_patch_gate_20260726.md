# P14 hunk-minimal patch gate

Snapshot: 2026-07-26

## Result

The accepted P13/P14 source surface has been reapplied and validated from
pristine detached worktrees at the pinned heads:

- Torch-Spyre: `59545440f0e7091ff1b2f90df63580da1842f3fe`
- FMS: `61bc991b175103e80cb8202b24a66ba7dbe79d1b`
- DeepTools: `406142afb9f080b9271e7c565a757ab8d8b5ed8f`

No commit, push, pull request, merge, or device run was performed.

## Patch artifacts

| Artifact | SHA-256 | Size | Scope |
| --- | --- | ---: | --- |
| `p14_torch_hunk_minimal.patch` | `7e50da8e7c7cb29af6893822737ab8eabd6e5c15797a3cef764786f9d45219cd` | 968 lines | Seven Torch files, 636 insertions and 13 deletions |
| `p14_fms_hunk_minimal.patch` | `befc6c2ec58be5073e7c5fc97e5204c6437dbd6fc7fa3ae2d1d946f8d91404a5` | 314 lines | Four FMS files; byte-identical to the previously validated clean FMS patch |
| `p14_deeptools_hunk_minimal.patch` | `bcd2f7ef2a531cac3cd437667c9e1ff971b0a584bedc0eb4051817ddc0b570e9` | 62 lines | One DeepTools file, 23 insertions and 5 deletions |

The former Torch artifact was 1,863 lines and interleaved disabled
P03/P05/P12 infrastructure. The former compatible DeepTools artifact was 519
lines across four files. No added line in the minimized Torch or DeepTools
patch mentions P03, P05, P10, P11, P12, the compact-GQA experiment,
singleton-Y diagnostics, direct-copy NOP diagnostics, or STCDP dump logging.

The remaining Torch changes are limited to:

- P13's 28-owner last-token LM-head work division and subset all-gather;
- P14's final-token source/consumer work divisions, last-cohort selection,
  LX row offset, rational destination allocation, and post-alignment labels;
- generic dense-prefix participant-union handling required by the sparse
  P13/P14 endpoint maps;
- six focused P13/P14 tests.

The remaining DeepTools change keeps the relayout scheduled on the full
source/destination participant union while constructing direct-copy output
addresses and pieces only for the destination allocation's physical owners.

## Clean reapply and validation

The three artifacts were applied, not merely apply-checked, to a second set of
pristine detached worktrees under:

`/home/adnan/codex-isolated/device_parity_tracks_20260726/p14_minimal_20260726/reapply`

All three worktrees passed `git diff --check` at their exact pinned heads.

Focused Torch compiler tests:

```text
6 passed, 15 deselected in 0.66s
```

The entire touched Torch test file also passed in the curation worktree:

```text
21 passed in 0.24s
```

Focused FMS generation-policy tests:

```text
3 passed, 9 deselected in 10.43s
```

The minimized DeepTools translation unit passed a C++17 `-fsyntax-only`
compile using the exact-head build's compile command. The output contained
only pre-existing header warnings.

## Caveats

- The accepted clean P14 T-C-T-C device result remains the authoritative
  correctness, timing, and emitted-transport evidence. It was not repeated
  because this pass removed unrelated and diagnostic hunks without changing
  the retained P13/P14 logic.
- The minimized DeepTools source was syntax-compiled but not linked into a new
  `dxp_standalone`, and no device run used that new source. A promotion build
  should rebuild DeepTools from this exact one-file patch before claiming the
  binary itself is hunk-minimal.
- Dump-only instrumentation from the prior four-file DeepTools overlay is
  intentionally absent. Future transport recapture must enable the standard
  dump path separately; the instrumentation is not part of the functional
  patch.
- The FMS patch is unchanged and still requires
  `materialize_decoupled_head_for_spyre(..., padded_vocab=50176)`.

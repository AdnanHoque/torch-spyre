# Cross-run protocol (the decisive experiment)

Codex's caveat is correct: the program (`init.txt`, dsg, pagi) differs across pods
**even for the control** `decode attn@V 2_8_2_1` (60us on both), so "program differs"
is necessary-not-sufficient. The clean discriminator is to **run one pod's generated
program on the other pod's device/runtime** (swap the artifact, hold the runtime).

## Mechanism (verified on the Claude pod)
`torch_spyre._C.launch_kernel(code_dir, [a,b,out])` replays a pre-generated program
dir directly — no recompile. Verified: replaying the Claude-pod QK^T prefill 1_4_8_1
program reproduces 1631 us (== its compiled timing). `replay_foreign.py` stages any
program dir under its internal `<hash>` name and times it (20 reps).

## Steps
1. Codex exports his full program dirs (sdsc_0.json + bundle.mlir + execute/ +
   loadprogram_to_device/ + *_dsg.txt + segment_size.json) for the 4 splits, intact
   (internal `sdsc_fused_bmm_0_<hash>` subdir names preserved).
2. Claude runs Codex's program on the Claude device:
   `python replay_foreign.py <codex_program_dir> B M N K`
3. Codex runs the committed Claude programs (this dir's `programs/`) on the Codex device.

## Interpretation
- Claude device runs **Codex's** QK^T 1_4_8_1 program **fast (~735us)** -> DeepTools
  **codegen** (his program is genuinely better; the slowdown is in program generation).
- Claude device runs Codex's program **slow (~1635us)** -> **runtime/firmware/device**
  (same program, our runtime executes it slower).
- Control `2_8_2_1` should run ~60us either way (sanity).

## Risk / fallback
The program embeds device pointers (`dev_ptr`, `defaultAddr_`). If the allocator is
deterministic across pods these match and replay is valid; if a foreign program
mis-addresses, fall back to a STRUCTURAL diff (instruction / DMA-op count of the
slow-split program vs the control, mine vs his) — substantive only if the slow-split
programs differ structurally while the control programs are structurally equivalent.

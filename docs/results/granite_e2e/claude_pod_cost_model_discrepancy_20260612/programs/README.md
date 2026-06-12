# Claude-pod POST-SDSC program export

The compiled program (deeptools/flex output) for 4 disputed splits, to diff
deeptools-codegen vs runtime/firmware. `loadprogram_to_device/.../init.txt` is
the actual AIU program sent to the device; `*_dsg.txt` the schedule;
`execute/.../pagi.json` and `segment_size.json` the program metadata. The
hash-named subdir (`sdsc_fused_bmm_0_<hash>`) varies per compile — diff by file
CONTENT, not path. `_sha256.json` per dir lists content hashes.

## Ablations already done on the Claude pod (all negative)
- DT_OPT: unset / allopt / baseopt / explicit-ON / explicit-OFF all give
  ~1626-1636 us for QK^T prefill 1_4_8_1 (codex 735). DT_OPT is NOT the cause.
- flex build: PR#1019 debug-build flex (231MB) vs harvest stock flex both
  ~1600 us. flex is NOT the cause.
- control decode attn@V 2_8_2_1 = 60 us on both pods and under every config ->
  device/firmware runs efficient programs identically; the gap is split-specific.

## Decisive remaining diff
Compare `init.txt` for prefill QK^T 1_4_8_1 (2.2x apart) across pods:
- differs -> deeptools BUILD generates a different/worse program (leading hypothesis;
  codex's measured times track ideal_cycles, mine don't)
- identical -> runtime/firmware executes the same program differently

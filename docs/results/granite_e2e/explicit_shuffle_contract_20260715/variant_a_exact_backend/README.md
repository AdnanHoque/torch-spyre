# Variant A bounded explicit SHUFFLE backend result

## Result

The authoritative SHUFFLE-only fixture lowers successfully through DXP/DCG/DCC.
The backend uses the frontend-planned LX endpoints directly:

- S1: `0x24000`, compact `out=512`, 1024-byte row stride
- S2: `0x44000`, expanded `out=4096`, 8192-byte row stride
- 8 bounded `STCDPOpLx` rows
- 256 logical relations: 224 remote and 32 local
- no HBM movement, dynamic LX allocation, or `ReStickifyOpLx`
- bounded rows `0..7` execute in the SHUFFLE bundle slot before the consumer

The physical verifier also records 29,360,128 remote ring bytes and the emitted
PCFG node counts.

## Diagnosis

The archived `Scheduler failed to find a suitable op mapping` message came from
an older binary. Replaying with the rebuilt isolated binary entered the bounded
lowering and exposed the real failure in DCG piece coverage: generated pieces
read dimensions from the top-level SuperDSC shape, which is empty for this
explicit marker. That produced invalid dimensions such as `in=-1`.

Using the nested DLDSC shape made replay compile, but initially left both
endpoints with the consumer's 8192-byte row stride. The final narrow fix uses
the nested shape only for bounded direct SHUFFLE and explicitly retains the
source LDS as compact `out=512`; S2 remains `out=4096`. Existing non-bounded
relayout behavior is left unchanged.

## Backend change

Only `dxp/SdscRelayoutInsertion.cpp` changes relative to Deeptools
`704c19f8fb7f0cc972f20404f9dd0010895a35e2`.

For an explicit LX-to-LX SHUFFLE whose source partitions one leading stick
dimension and whose destination replicates the full dimension, DXP:

1. recognizes one bounded expanded dimension;
2. preserves compact S1 and full-stride S2 as independent physical layouts;
3. emits one bounded `STCDPOpLx` row per source shard;
4. writes each shard into its explicit S2 window; and
5. replaces the SHUFFLE compute marker with those data-only rows.

The normal relayout path is unchanged. Direct SHUFFLE uses the already planned
S2 addresses, so the dynamic allocation/HBM fallback path is not exercised.

## Evidence

- `deeptools.patch`: readable backend diff with trailing whitespace removed
- `replay.rc`: final replay return code
- `post-sdsc.tar.gz`: all DXP/DCG/DCC debug SDSCs plus `smc.txt`, `init.txt`,
  and `cb.txt`
- `verification.json`, `verification.md`: endpoint, stride, relation, movement,
  and PCFG verification
- `ordering-verification.json`, `ordering-verification.md`: byte-identical
  companion bundle proof that SHUFFLE precedes consumer `batchmatmul`
- `authoritative-fixture/`: immutable input fixture and SHUFFLE-only bundle
- `integration-order-fixture/`: producer/SHUFFLE/consumer bundle used only for
  order verification
- `commands.sh`: exact reproduction commands
- `raw-source-artifacts.tar.gz`: complete byte-for-byte pod artifact directory,
  including the original patch, build/replay logs, raw PCFG, and `post-sdsc/`
- `ARCHIVE_SHA256SUMS`: relocation-safe checksums for the archived copy

## Scope note

The authoritative fixture is intentionally SHUFFLE-only. The companion
three-operation synthetic bundle establishes program order, but its producer
contains an unresolved synthetic symbol and is not used as a full replay gate.
This does not affect the exact SHUFFLE replay, which reaches DCC with return code
zero. Numeric device correctness and performance remain separate goal gates.

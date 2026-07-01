# Comms Collectives Lane Checkpoint - 2026-07-01

This directory records the three parallel communication-class probes from July
1, 2026.

## DLDSC/STCDP Internal Range Path

CDX workspace:

```text
/home/adnan-cdx/codex-isolated/comms_collectives_staged_substick_agent_20260630_225637/deeptools
```

The prototype moved past the old STCDP front-end check for sub-stick pieces:

```text
inpSP.dimToSize["out"] >= stickDim
16 >= 64
```

The new boundary is ring lowering.  `SenPcfgRingDtNode` and the emitted L3 ring
opcode are stick-addressed; the instruction path does not expose source or
destination intra-stick byte offsets.  Direct partial-stick ring transfer is
therefore not a small metadata-only patch.

Latest CDX artifact:

```text
/home/adnan-cdx/codex-isolated/comms_collectives_staged_substick_agent_20260630_225637/runs/staged_substick_ring_offsets_20260701_011119
```

## Explicit Byte-Range Path

CLC workspace:

```text
/home/adnan/codex-isolated/explicit_range_agent_20260630
```

The explicit byte-range prototype lowers four 32-byte ranges into one contiguous
128-byte consumer stick and reaches senulator/backend acceptance:

```text
/home/adnan/codex-isolated/explicit_range_agent_20260630/runs/explicit_range_four_senresolve_ctc_20260701_012808
```

Local files:

```text
explicit_range_diag.txt
explicit_range_semantic_summary.json
```

This proves backend artifact acceptance for the four intended ranges.  It does
not yet prove patterned runtime memory correctness.

## 4-Head Attention Script

dev-pf workspace:

```text
/home/adnan/codex-isolated/attention_4h_probe_20260701_005035
```

Script:

```text
git@github.ibm.com:aviros/test-spyre-scripts.git
05deb9702654f73781b457ed052a3ff69316670f
test_flash_4_head.py
```

Current main fails before scratchpad/SDSC:

```text
buf10 = running_max = torch.maximum(real_max, block_max)
NotImplementedError: buf10 (Pointwise): no mechanism to resolve stick incompatibility
```

The missing mechanism is singleton-stick reduction restickify.  With a runtime
monkey patch restoring that reverted branch, both baseline and scatter-planner
runs reached scratchpad/SDSC and emitted 89 SDSCs.

Observed allocation shift:

```text
baseline singleton patch: 154 hbm, 72 lx
scatter singleton patch:  120 hbm, 106 lx
```

Both singleton-patch runs still failed later in DXP:

```text
DtException: Could not find any suitable dimension mapping
```

Local stderr artifacts:

```text
attention/baseline_current_prescratchpad_stderr.log
attention/baseline_singleton_stderr.log
attention/scatter_singleton_stderr.log
```

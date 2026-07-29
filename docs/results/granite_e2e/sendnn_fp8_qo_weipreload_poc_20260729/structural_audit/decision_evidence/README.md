# Q/O FP8 PoC evidence package

Scope: Granite Q/O projection (`K=N=4096`) on DD2 only. No 1p5 evidence is
included.

Contents:

- `stock_m512_workdiv_decision.log`: compact stock work-division interposer
  output.
- `weipreload0_m512_workdiv_decision.log`: compact treatment output.
- `qo_fp8_workplan_all_m.tsv`: machine-readable stock plan map for
  `M=1..2048` plus the M=512 treatment structure.
- `qo_fp8_workplan_all_m.md`: condensed interpretation and evidence paths.
- `SOURCE_MECHANISM.md`: exact DeepTools source path from preload modeling to
  the rejected work split.

The clean M=512 treatment run independently establishes numerical correctness
and timing. The decision and structure interposers reached the relevant
compiler decisions/artifact generation but later encountered an interposer
teardown boundary. Their logs are used only for compiler-decision evidence;
no timing or correctness is attributed to those processes.

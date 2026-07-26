# Artifact ledger

## Included in this branch

- Full-model SenDNN and Torch-Spyre run logs, metrics, and gzip traces.
- SenDNN full-model compiler export archive.
- SenDNN full-model post-LXOpt SDSC/perfdsc archive.
- Relayout template catalog and expanded replay manifest.
- SMC/ISA summaries, attribution tables, init-packet summaries, and Torch BMM
  init-packet evidence.
- P02 runs `_c`, `_d`, and `_e`: logs, allocation/plan dumps, available SDSC,
  generated logits, and correct reference logits.
- Representative P06 pre/post-DXP relayout payloads proving the debug method.
- Gap analysis, SMC analysis, charts, and both report builders.
- Complete DeepTools all-to-all common-refinement handoff and example bundle.

All checked-in files are covered by `provenance/SHA256SUMS`.

## Logged but not duplicated

These are regenerable or too large/noisy for source control:

| Artifact | Original location | Reason |
| --- | --- | --- |
| Granite weights/tokenizer | `/tmp/models/granite-3.3-8b-instruct` | External model payload |
| Torch uncompressed full trace | local `work/full-model-artifacts/torch_spyre/*.trace.json` | 108 MiB; gzip copy included |
| SenDNN uncompressed full trace | local `work/full-model-artifacts/sendnn/*.trace.json` | gzip copy included |
| Torch Inductor caches | `$ROOT/runs/*/cache` | Regenerable, many generated files |
| Compiler exports | `$ROOT/runs/*/export` | Regenerable; selected SenDNN archive included |
| All raw run trees | `$ROOT/runs` | More than 15,000 files; top-level index included |
| Full SenDNN init tensors | local `work/smc-study/init_inputs` | 69 MiB; summaries and compact Torch archive included |
| DeepTools source/build | `$ROOT/deeptools`, `$ROOT/deeptools-build` | Exact base head and dirty overlay included |

Use `tools/index_artifacts.py PATH --output inventory.json` to produce a
recursive path/size/SHA256 manifest when moving or deleting any original tree.


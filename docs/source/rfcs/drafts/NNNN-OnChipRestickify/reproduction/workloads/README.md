# Workload A/B research harnesses

Per-workload A/B harnesses behind the on-chip core-to-core RFC. Each directory
holds a baseline `torch.compile(backend="inductor")` workload, the offline
handoff classification (`edges.md` / `eligibility.md`), and a grounded on-chip
speedup `projection.md` derived from the measured per-MB anchor. The attention
dirs also carry a bespoke MIXED splice plus a device validation harness.

These are **research harnesses, not production code.** They carry machine-specific
absolute paths (`/home/adnan/...` interpreter, `/tmp/...` spliced bundles, the
`/tmp/val-boot` import shim) wired the way the original `/tmp` worktree ran them.
The shared `reproduction/env.sh` parameterizes the paths used by the main
reproduction harness; the workload device scripts here were not retrofitted to it
and reference the spliced-bundle dirs (which are intentionally NOT vendored — too
large) by absolute path. To re-run, override the interpreter, `PYTHONPATH`, and
the `ONCHIP_DIR`/`SCRIPT`/`DIR` paths at the top of each `.sh`. Device steps run
SOLO (single shared accelerator).

## Workloads

| Dir | Baseline workload | Projection / classification |
|---|---|---|
| `transformer_block` | decode block (RMSNorm/SDPA/SwiGLU) + `edge_classifier.py` | `projection.md`, `edges.md` |
| `moe_block` | MoE block + `moe_ffn_workload.py` (expert bmm) | `projection.md`, `edges.md` |
| `moe_routing` | dispatch/combine/router as permutation matmuls | `projection.md`, `eligibility.md` |
| `mamba2` | SSD block + 3 microbenches | `projection.md` |
| `attention` | SDPA seq=64 + QK^T->softmax splice + devval | `projection.md` |
| `attention_512` | SDPA seq=512 splice + devval | (numbers in `attention/projection.md`) |

Consolidated numbers (measured + projected) are in the RFC's
`PerformanceResults.md`. Projected speedups are summarized there from each
`projection.md`. Excluded: large spliced-bundle dirs and `senprog`/`init.txt`
artifacts.

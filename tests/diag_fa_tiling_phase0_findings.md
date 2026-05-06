# FA tiling via decomposition — Phase 0 findings

## TL;DR

**FA tiling via torch_spyre decomposition is fundamentally slower
than the reference attention on AIU** because of per-tile launch
floor overhead. The path is closed for solo torch_spyre work.

To deliver FA tiling on AIU, the entire tile loop must fuse into a
single SDSC kernel. That requires deeptools-side work (custom kernel
template) or runtime-side work (kernel-binary prefetch). Neither is
solo torch_spyre territory.

## What I measured

For Llama 70B-style attention shapes, compared three configurations:

- **Reference**: `bmm(softmax(bmm(Q, K^T)), V)`. Materializes the full
  M×M attention matrix.
- **FA tiled (k_tile=128)**: K-tile loop with online softmax, manually
  unrolled in eager Python. Each tile compiles to its own kernel.
- **FA single-tile (k_tile=M)**: same FA-2 algorithm but with one
  iteration — no actual tiling. Tests the algorithm overhead
  independently of the loop.

Results:

| n_heads | M | reference | FA single-tile | FA k_tile=128 |
|---:|---:|---:|---:|---:|
| 8 | 256 | 15.1 ms | 23.5 ms (1.5× slower) | 55.2 ms |
| 8 | 512 | 15.6 ms | 24.8 ms (1.6× slower) | 105.9 ms |
| 8 | 1024 | 17.8 ms | 26.6 ms (1.5× slower) | 209.7 ms |
| 32 | 1024 | 28.6 ms | 35.7 ms (1.3× slower) | 230.0 ms |

**Two findings layer on top of each other:**

1. **Single-tile FA is already 1.3-1.6× slower** than reference. The
   FA-2 algorithm has more tensor ops (matmul + amax + where + exp + exp
   + matmul + accumulation) than the reference (matmul + softmax + matmul).
   Each op is a kernel launch on AIU; LF is per-launch.
2. **Each additional tile multiplies the overhead** by another ~5 ms.
   8 tiles at k_tile=128 → 8 × 26 ms = 208 ms for what reference
   computes in 18 ms.

## Why each tile is so expensive

Phase 0 of Project B established that AIU's per-op launch floor is
~3 ms, and **LF stacks on the HMI pipeline** (it's not free,
it's serial with operand fetch). Each compiled call to fa_step:

- 6 internal tensor ops, fused by Inductor into ~3-4 kernels
- 3-4 kernel launches × 3 ms LF each = ~12 ms minimum per call
- Plus actual HMI + compute time

For a multi-tile FA loop with N iterations: 12+ ms overhead per tile
× N tiles. The per-tile HMI savings (smaller intermediate S matrix)
are tiny compared to LF cost.

## Workarounds I tried

### Use torch.where instead of torch.maximum
torch.maximum is `UnimplementedOp` on the Spyre backend. Worked around
by using `torch.where(m_tile > m_state, m_tile, m_state)`. **Lesson**:
worth filing as a backend gap to fix, but not blocking.

### fp16 throughout (no fp32 promotion)
The Spyre backend doesn't support `to_dtype IEEE_FP32` inside
compile graphs. Worked around by keeping running state in fp16.
Slight accuracy hit on long sequences but acceptable for shorter ones.
**Lesson**: also worth filing as a backend gap.

### Pre-tile outside compile
Initial attempt put `k[start:end]` slicing inside the compiled
function. Hit `Could not find a host dimension matching stick expr d2 + 128`
in the stickify pass — Inductor on AIU can't lower dynamic-offset
slicing. Worked around by slicing in eager Python and passing pre-
tiled tensors to the compiled `fa_step`. **Lesson**: dynamic-shape
slicing is a backend gap.

### Larger k_tile (fewer tiles)
Sweep showed even k_tile=M (one big tile, no real FA tiling) is
slower than reference. The per-tile-step overhead is structural —
the FA-2 algorithm's extra ops cost LFs that the reference doesn't
pay.

## Why this differs from the FA prototype's prediction

The earlier FA prototype (in `joint_swp_ws_fa_prototype.py`) predicted
a 1.36× speedup from FA-tiled attention on Llama 70B M=2048. That
prototype assumed:
- Each FA stage runs as one fused kernel inside one launch (single LF)
- Cross-tile pipeline of PT and SFP is possible

In a real torch_spyre decomposition, **each FA op gets its own kernel
launch with its own LF**. The "fused tile kernel" assumption doesn't
hold for a Python-decomposition implementation. It would hold for a
custom SDSC kernel.

This calibration provides a 4th datapoint in the calibration thread:
the joint SWP+WS prototype assumed real FA on AIU; this Phase 0 shows
that real FA on AIU via decomposition isn't viable.

## Where this leaves the project

The brainstorm originally listed FA tiling as a 4-8 week torch_spyre
project. **That estimate doesn't hold** — the path is closed. To
ship FA tiling on AIU you need:

**Path A: Custom FA SDSC kernel** (deeptools work)
- 6+ months. Write a single SDSC kernel that does the tile loop
  internally, with all the FA-2 state machine inlined. Fuses
  matmul + softmax + matmul into one launch.
- Out of solo torch_spyre scope.

**Path B: Kernel-binary prefetch** (runtime work)
- Hide LF between ops by prefetching kernel binaries during prior
  op execution. Same project I flagged out of Project B. Would
  enable FA-via-decomposition at lower cost (LF amortized).
- Also out of solo torch_spyre scope. Same deeptools/runtime
  partnership needed.

**Path C: Close the project**
- Accept that AIU's current attention path (materialized M×M) is what
  ships, even though it's HMI-heavy. Focus on other levers.

## Backend gaps surfaced (worth filing as torch_spyre issues)

While prototyping, I hit three backend gaps that could be filed as
small fix-it issues independent of the FA tiling question:

1. **torch.maximum → UnimplementedOp**. Workaround: torch.where.
   Should be a 1-day fix in the Spyre lowering.
2. **to_dtype IEEE_FP32 unsupported inside compile graphs**. Blocks
   any fp16 → fp32 accumulation pattern. Several real workloads need
   this. Probably bigger fix.
3. **Dynamic-offset slicing fails stickify**. Limits any tile-based
   decomposition. Probably moderate fix in the stickify pass.

These are independent of FA tiling and worth raising even if the
project closes.

## Updated brainstorm — what's still on the table for solo torch_spyre

After both HMI BW investigation (Jamie's PR already covers it) and
FA tiling (closed as solo project), the remaining torch_spyre-only
projects from the original brainstorm:

| project | status | block-level estimate |
|---|---|---:|
| Fix SDPA-to-bmm regression (`spyre__sdpa_overrideable`) | viable | 30-50% on attention |
| Broaden k_fast heuristic (extends PR 1933) | viable | 2-8% block on prefill |
| LX-fit aware splits | viable | preventative (avoids 10× regressions) |
| Operator fusion audit | needs investigation | ? |
| Fix `torch.maximum` UnimplementedOp | small fix | enables FA-2 patterns |
| Fix to_dtype IEEE_FP32 | medium fix | enables fp32 accumulation patterns |
| Fix dynamic-offset slicing in stickify | medium fix | enables tile-based decomp |

**Strongest candidate still standing**: the SDPA regression fix.
Smallest scope, real bug, evidence-grounded (calibration data).

## Files

- `diag_fa_tiling_phase0.py` — the prototype
- `diag_fa_tiling_phase0_results.txt` — k_tile sweep results
- This doc — findings

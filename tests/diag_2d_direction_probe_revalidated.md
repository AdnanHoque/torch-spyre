# 2D direction probe — re-validated with cold-cache + meta-finding

## TL;DR

Original 2D direction probe gave correct CONCLUSION ("no direction
lever") but wrong NUMBERS (all perms at ~10.9 ms). Re-validation with
cold cache between variants gives correct numbers:

- default (2-hop column): **5.20 ms**
- k_fast (1-hop row): **4.30 ms**
- col_dir (1-hop column): **4.30 ms**

**Hop count is the lever (1.20×, 5.20→4.30 ms). Direction is NOT.**
k_fast already captures the win. PR 1932 is still real — just with
1.20× speedup, not the 2.78× reported in original validation.

**Meta-finding**: my earlier probes were partially confounded by
caching. Inductor caches compiled kernels keyed on FX graph hash;
within one Python process, switching permutations through the
patcher hits the cache without recompiling. The first compiled
variant served all subsequent ones with its kernel.

## What happened

On the PR 1932 branch (where built-in k_fast is wired into
generate_sdsc directly), measuring the same shape under different
flag/perm combinations:

| variant | wall ms |
|---|---:|
| builtin k_fast ON, no manual perm | 4.27 |
| **builtin OFF, manual k_fast (cache hot)** | 5.14 |

Same JSON output should produce same wall, but they didn't. Cause:
the first compile (builtin ON) cached an optimized kernel; the
manual probe's second compile produced different SDSC JSON but
hit the cached kernel from the first run.

Verified by clearing `/tmp/torchinductor_adnan/` between variants:

| variant (cold cache) | wall ms |
|---|---:|
| builtin OFF, manual k_fast | **4.37** |

Now matches the built-in k_fast wall (4.27 ms) within noise.

## What this means for the 2D direction question

Re-running with cold cache between variants:

| permutation | k-pair 2D position | wall ms (3-run median) |
|---|---|---:|
| default | 2 hops column | 5.19 |
| k_fast (PR 1932) | 1 hop row | 4.30 |
| col_dir | 1 hop column | 4.30 |

**Hop count is the lever (1.20× from 2→1 hop). Direction symmetric.**

This is the same conclusion as the original probe but with correct
absolute numbers. The 2D direction lever does not exist; k_fast
already captures the optimal placement.

## What this means for k_fast PR (1932)

The PR's win is real but smaller than originally claimed:

| metric | original validation | today (cold-cache) |
|---|---:|---:|
| +id (default) wall | 10.93 ms | 5.19 ms |
| +kf (k_fast) wall | 3.94 ms | 4.30 ms |
| speedup | 2.78× | 1.20× |

Default emission has gotten ~2× faster (deeptools updates).
k_fast still helps, but by 20% not 178%.

**Worth re-validating PR 1932's measurements before merging** —
the headline claims may be stale by 2×.

## Multicast permutation: re-validated

Re-tested the multicast permutation with cold cache between
variants on the same shape (128, 8192, 8192) under (8,4,1):

| permutation | wall ms (cold cache) |
|---|---:|
| identity | 3.98 |
| m_adj | 3.91 |
| reversed | 3.95 |

Spread: 1.6%. **Conclusion holds — multicast permutation is not a
lever.** Cache wasn't masking a hidden effect there.

## Meta-finding: cache poisoning in compile probes

Many of my prior diag probes ran multiple compile-and-bench variants
in a single Python process. Within a process, Inductor caches
compiled kernels by FX graph hash. If two variants produce the same
FX graph (which they do when the only difference is post-compile
SDSC patching), the cache serves the first variant's kernel for all.

This means:
- The first variant in each process gets a fresh compile
- Subsequent variants hit cache and silently use the first variant's
  kernel
- Walls appear identical across variants regardless of patcher

Probes affected:
- 2D direction probe (this doc): conclusion correct, numbers wrong
- Multicast permutation probe: conclusion correct (re-validated above)
- Possibly others — anything that patched generate_sdsc post-compile

**Going forward**: probes that vary post-compile SDSC patches MUST
clear `/tmp/torchinductor_adnan/` between variants OR run each
variant in a separate Python process. The diag-branch probes that
used `_force_split` (a planner-side override that DOES affect FX
graph hash) were not affected.

## What this changes about the brainstorm

Same conclusions as before, with refined confidence:

1. **Multicast permutation**: still closed. Cold-cache verification
   confirms.
2. **Inter-op alignment**: still closed (placement-independent).
3. **2D direction lever**: closed. Hop count matters, direction
   doesn't.
4. **k_fast PR 1932**: still useful but smaller win than originally
   claimed. Should re-validate before merging.

The strongest remaining candidates for new solo torch_spyre work
are unchanged:
1. Investigate / lift the 6-tensor fusion cap (issue #827)
2. LX residency planner
3. Fix SDPA-to-bmm regression
4. Cost-model-driven planner heuristic

## Files

- `diag_2d_direction_probe.py` + results — original (cache-confounded numbers)
- This doc — re-validation + meta-finding on cache poisoning

# Co-assignment — the Inductor-only on-chip SwiGLU win

The value-correct alternative to the data-op reshard (Path A). Where the reshard
moves the matmul output across work-divisions with an `STCDPOpLx` data-op
(dxp-gated, and — as Path A proved — value-broken by a DCG EBR packing bug),
co-assignment **removes the cross-division edge entirely** by making the
element-wise consumers inherit the matmul's split. No data-op, no dxp gate, no
deeptools change.

## Mechanism

The `matmul→pointwise` HBM round-trip exists only because the matmul `(m4,n8)`
and the pointwise chain (pure-M default) are divided **independently**. After the
cost model picks the matmul split, `apply_coassign` (monkeypatch on
`passes.cost_model_matmul_division`) BFS-walks the element-wise (`Pointwise`)
consumer chain and commits the matmul's `{mb:4,out:8}` split to each consumer
(mapped onto the consumer iter-space by matching dim extents,
`apply_splits` → `op_it_space_splits`), returning them as **preassigned** so
`work_distribution` honors it. The edge becomes **same-division same-core**: each
consumer core reads exactly the tile its own core produced. Element-wise ops are
split-agnostic, so the propagation is value-preserving by construction.

This is Stage 1 (split propagation → same-shard hand-off). Stage 2 (future) folds
in `onchip_softmax_chain.apply_lx_flip` — the proven 1.88× base-pointer flip — to
make the now-same-core edge LX-resident.

## Result (2026-06-18)

- **Perf (`/tmp/spyre-perf-suite`, `fms_granite_micro.swiglu_unfused`, profiler
  stack):** `[COASSIGN]` flips `neg` + all 5 element-wise ops to `{mb:4,out:8}` =
  m4n8 (matching the matmul). Kernel **12.9 ms vs 13.9 ms** unfused baseline ≈
  **~7% faster on stock dxp** (no data-op, no patched dxp).
- **Value-correctness (`spyre-perf-suite-aisw`, `granite_micro_bench.swiglu_unfused`,
  seed 0, 1×512×4096):** device co-assign output vs CPU eager —
  **max_abs_diff = 0.0059, mean_abs_diff = 0.00081, `allclose(1e-2,1e-2)=True`**
  (device mean/std −0.00341/0.1064 vs eager −0.00339/0.1055). fp16-noise level —
  **value-correct.** The exact opposite of the data-op reshard's ≈0 corruption
  (`mean|out|≈0.0001`, see `../reshard/PATH_A_PROGRESS.md`): the reshard moves data
  to the wrong place, co-assignment never moves it at all.

## Why it wins where steering (A1) lost

A1 forced the **producer** matmul to pure-M to align it with the pure-M
pointwise → matmul slow (1.4–1.6× regression). Co-assignment instead forces the
cheap, memory-bound **consumer** to the matmul's `(m4,n8)` → the matmul keeps its
fast split and the hand-off goes same-core. Captures the reshard's goal (keep
m×n, kill the hand-off) with zero data-ops.

## Files

- `coassign.py` — the pass (`apply_coassign`).
- `save_coassign_out.py` — lean device value-capture (compile + one forward + save).
- `check_maxerr.py` — offline diff of the saved output vs CPU eager.
- `run_coassign_device.sh` — device wrapper (stock dxp + harvest stack; run SOLO).

See `../reshard/PATH_A_PROGRESS.md` for the reshard (Path A) verdict this
supersedes.

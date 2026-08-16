# Validation

## Source checks

```text
git diff --check
python -m compileall -q experiments moe_asgemm/tools
ruff check experiments moe_asgemm/tools
```

## Focused compiler tests

The integrated prototype was developed with focused coverage in:

```text
tests/inductor/test_coarse_tiling.py
tests/inductor/test_core_mapping.py
tests/inductor/test_scratchpad_solver.py
tests/inductor/test_work_division_hint.py
```

The reduced acceptance sequence is:

```text
1. source emission
2. exact C1 structural checker
3. real DeepTools compile with no kernel launch
4. two-alpha device correctness
5. full-shape C32 source and bundle structure
6. full-shape correctness
7. timing
```

## Fail-closed full-shape conditions

- Exactly one source bundle and one wrapper call.
- Exactly one flat expert loop and no temporal token loop.
- Exactly one X HBM-to-LX preheader.
- X, gate, and up share the same all-32-core row map.
- Exactly four loop-advanced HBM inputs: Wg, Wu, Wd, and alpha.
- No internal HBM pool allocation.
- No HBM restickify.
- All internal compute and accumulator storage is LX.
- Runtime alpha is `[128,512,1]` and applied after down.
- Exactly one final HBM output.
- One callable responds correctly to identity, permutation, and hot-eight alpha.

## Evidence integrity

```text
shasum -a 256 -c moe_asgemm/SHA256SUMS
```

Do not accept timing when any structural or correctness condition fails.

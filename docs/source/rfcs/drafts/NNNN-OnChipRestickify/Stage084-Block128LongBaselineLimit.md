# Stage084 - Block128 Long Baseline Limit

## Question

Are the `B1 H2 D64 block128` failures at L768/L1024 caused by the
loader-specialized K/V prefetch artifact?

## Result

No. The non-warpspec baseline variant fails the same rows with the same mismatch
summaries:

```text
variant: onchip_hbm_kv_layout_xform

B1 H2 L768 D64 block128:
  status: failed
  mismatched: 63 / 98304
  max abs: 0.25390625 at (0, 0, 463, 51)

B1 H2 L1024 D64 block128:
  status: failed
  mismatched: 2 / 131072
  max abs: 0.120361328125 at (0, 1, 728, 8)
```

The warpspec variant fails with the same mismatch counts and worst indices.
The long block128 rows should therefore stay out of the warpspec promotion gate
until the underlying block128 long-sequence baseline is corrected.

## Interpretation

The current block128 promotion boundary is:

```text
promote: B1 H2 D64 block128 L256,L384,L512
exclude: B1 H2 D64 block128 L768,L1024
```

This is not a reason to roll back the loader-core schedule. It is a separate
baseline correctness limit in the block128 long-sequence path.

# Variant A bundle-order verification

Status: **pass**

| Bundle index | File | Operation |
|---:|---|---|
| 0 | `sdsc_0.json` | `ReStickifyOpHBM` producer |
| 1 | `sdsc_1.json` | explicit `SHUFFLE` marker |
| 2 | `sdsc_2.json` | consumer `batchmatmul` |

The integration SHUFFLE is byte-identical to the authoritative SHUFFLE-only
fixture. DXP replaces that bundle slot with data rows `0..7` on every core.
Because `sdsc_execute` operations are ordered in the bundle, all eight bounded
rows complete before the following consumer `batchmatmul` begins.

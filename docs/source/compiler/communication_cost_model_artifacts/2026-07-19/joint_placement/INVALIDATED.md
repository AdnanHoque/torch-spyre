# Historical `joint_all` result is invalid for performance claims

The timing and structural files in this directory are preserved byte-for-byte
as forensic history. Their `success` filenames mean that the original gates
returned zero; they do not establish value correctness under the strengthened
test and must not be cited as placement-performance evidence.

The independent audit was later amended with this superseded verdict. Its
original byte-for-byte SHA-256 was
`c12cccf9805c8b68ee40bb8bf9ba140230cd74af474f109ad63710191d349656`;
the package manifest covers the amended file.

The prototype changed the physical mapping attached to the synthetic K
restickify/source view while leaving the actual scaled-K producer at the default
mapping. Both used the same local LX allocation and no materialized transfer
separated them. The consumer therefore interpreted produced bytes under a
different core-to-logical-slice mapping.

A later high-contrast device-versus-CPU gate found:

```text
mismatches at atol=rtol=1e-2: 241384 / 262144
maximum absolute error:       0.943985
mean absolute error:          0.135225
```

The oracle reproduced the same incorrect normal output bit-for-bit, confirming
that the oracle mechanism preserved graph semantics but not that those semantics
matched CPU. Changing only the source view reproduced the failure; changing the
actual producer and source together restored correctness.

Consequences:

- the historical `10.3867 us`, `4.864%`, and `1.05113x` timing claims are
  withdrawn;
- the non-causal projection derived from those timings is withdrawn;
- the offline `2,048 -> 672` hop-unit and `40 -> 16` load changes remain route
  proxies only; and
- the replacement evidence is the corrected coherent-placement factorial in
  the adjacent 2026-07-20 package.

The production legality invariant exposed by this failure is:

> Every unshuffled local-LX producer-to-consumer edge must use the same realized
> physical mapping. A mapping change requires an explicit materialized bridge.

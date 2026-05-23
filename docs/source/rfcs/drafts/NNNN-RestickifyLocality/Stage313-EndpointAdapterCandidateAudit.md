# Stage 313: Endpoint Adapter Candidate Audit

## Summary

The native PT-LX endpoint adapter contract now records the existing diagnostic
lowering candidates and why none of them is production-ready yet.

## Candidate Status

| Candidate | Useful property | Production blocker |
|---|---|---|
| `native-64x64-tiles` | Uses the native local PT-LX transform descriptor. | Output is still native-shaped and needs a consumer endpoint adapter. |
| `direct-64x64-tiles` | Can match the consumer descriptor shape in some cases. | Lacks a proven remote-fragment coordinate map. |
| `validgap-consumer-64x64-tiles` | Can force-validate the consumer descriptor with sparse valid-gap aliasing. | Lacks hardware value proof. |

## Why This Matters

This prevents the implementation from accidentally treating one diagnostic
property as the full solution.  The production path needs all of these at once:

```text
remote producer fragment gather
native local PT-LX transform
consumer-readable LX endpoint
hardware value correctness
```

No existing diagnostic helper proves all four.  The next implementation step is
therefore not to turn one on, but to lower the planned adapter explicitly:

```text
native PT-LX tile workspace -> consumer LX endpoint
```

and validate that adapter in isolation.

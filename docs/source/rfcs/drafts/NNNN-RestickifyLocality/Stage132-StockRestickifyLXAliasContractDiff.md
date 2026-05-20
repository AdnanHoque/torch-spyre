# Stage 132: Stock Restickify LX Alias Contract Diff

## Goal

Evaluate the newer prototype path:

```text
producer add -> stock ReStickifyOpHBM with input/output patched to LX -> consumer
```

This path now compiles and launches for the high-signal 2048 case, but the
values are wrong. The purpose of this stage was to compare the generated SDSC
contracts against the clean stock HBM restickify path and identify whether the
failure is a small metadata mismatch or a deeper ownership mismatch.

## Compared Artifacts

Clean stock HBM baseline:

```text
/tmp/stage132-bench-baseline/kernel_code/computed_transpose_adds_then_matmul_tuple_512/0001_sdsc_fused_add_t_0
```

Wrong-but-launching LX alias:

```text
/tmp/stage131-stock-lx-alias-tuple/kernel_code/computed_transpose_adds_then_matmul_tuple_512/0001_sdsc_fused_add_t_0
/tmp/stage131-stock-lx-alias-run-2048-normalmap/kernel_code/computed_transpose_adds_then_matmul_2048/0001_sdsc_fused_add_t_0
```

The tested graph shape was the computed-input restickify pattern:

```text
u = x + y.t() + z.t()
return u, u @ d
```

The tuple return lets us inspect whether the restickified bridge output is
already wrong before the downstream matmul.

## SDSC Contract Difference

The clean HBM path uses HBM as a global layout boundary. The producer writes a
global tensor, and the restickify op can read whatever logical region its own
work slice requires from that global tensor.

For the 512 HBM baseline:

| Op | Work split | Relevant allocation |
|---|---:|---|
| producer `sdsc_0_add` | `mb:32,out:1` | output HBM layout `mb,out` |
| `sdsc_1_ReStickifyOpHBM` | `mb:4,out:8` | input HBM layout `out,mb`; output HBM layout `mb,out` |
| consumer `sdsc_2_add` | `mb:32,out:1` | input HBM layout `mb,out` |

The LX alias path patches the stock restickify tensors to LX, but it does not
change the physical ownership contract. Each core now reads a local LX address,
not a global HBM address.

For the 512 LX alias:

| Op | Work split | Relevant allocation |
|---|---:|---|
| producer `sdsc_0_add` | `mb:32,out:1` | output LX layout `mb,out` |
| restickify alias | `mb:4,out:8` | input LX layout `out,mb`; output LX layout `mb,out` |
| consumer `sdsc_2_add` | `mb:32,out:1` | input LX layout `mb,out` |

For the 2048 LX alias:

| Op | Work split | Relevant allocation |
|---|---:|---|
| producer `sdsc_0_add` | `mb:32,out:1` | output LX layout `mb,out` |
| restickify alias | `mb:1,out:32` | input LX layout `out,mb`; output LX layout `mb,out` |
| consumer | `mb:32,out:1` | input LX layout `mb,out` |

The local LX start addresses collapse to one address in the JSON because the
address is local to each core. That is correct for a local tensor, but it means
the same address on core 0 and core 31 names different physical scratchpad
storage.

## Ownership Overlap

The failure is visible by comparing the logical region a producer core owns in
its local LX with the logical region the restickify work slice tries to read
from that same local LX address.

For the 512 alias case:

| Core | Producer local region | Restickify local need | Local overlap |
|---:|---|---|---:|
| 0 | `mb [0,16), out [0,512)` | `mb [0,128), out [0,64)` | `0.125` |
| 1 | `mb [16,32), out [0,512)` | `mb [128,256), out [0,64)` | `0.0` |
| 7 | `mb [112,128), out [0,512)` | `mb [384,512), out [64,128)` | `0.0` |
| 15 | `mb [240,256), out [0,512)` | `mb [384,512), out [192,256)` | `0.0` |
| 31 | `mb [496,512), out [0,512)` | `mb [384,512), out [448,512)` | `0.125` |

Summary:

```text
min/avg/max local overlap = 0.0 / 0.03125 / 0.125
```

For the 2048 alias case:

```text
producer split:    mb:32,out:1
restickify split:  mb:1,out:32
local overlap:     0.03125 on every core
```

Example core 0:

```text
producer owns:      mb [0,64),    out [0,2048)
restickify needs:   mb [0,2048),  out [0,64)
```

Only 1/32 of the restickify input needed by that core is actually present in
that core's local LX. The rest would have to come from other cores.

## Interpretation

The stock HBM path is value-correct because HBM is a global exchange point. The
restickify core can read the global source region matching its own work split.

The LX alias path is not value-correct because it replaces that global exchange
point with per-core local scratchpad addresses while leaving the restickify work
ownership unchanged. A local LX alias is only sound when the producer core and
restickify core own the same logical region, or when the bridge explicitly
fetches the missing pieces from remote cores.

So the wrong values are not surprising. The prototype is asking each core to
read a local tensor region that mostly lives in other cores' LX.

## Deeptools Context

Deeptools does have a first-class LX restickify opfunc. In the installed
template:

```text
/opt/ibm/spyre/deeptools/share/ddc/ddl_templates/restickify_sen1p5.ddl
```

both opfuncs are bound:

```text
opFuncName="ReStickifyOpHBM"
opFuncName="ReStickifyOpLx"
```

and the LX allocation path is explicit:

```text
memory="lx", data_connect="lxlu_input"
memory="lx", data_connect="lxsu_input"
```

The sysconfig also lists both `ReStickifyOpLx` and `ReStickifyOpHBM` as known
opfuncs.

This means the next reuse attempt should not blindly alias a stock
`ReStickifyOpHBM` SDSC to LX. It should use the stock LX restickify contract, or
an explicit remote-LX bridge, with producer/restickify/consumer ownership made
consistent.

## Recommendation

Stop treating the stock-LX-alias path as a correctness candidate by itself. It
is useful as a diagnostic because it proves the bundle can launch at 2048, but
it violates the per-core ownership contract.

There are three viable follow-up paths:

1. Generate a real `ReStickifyOpLx` SDSC through the `restickify_sen1p5.ddl`
   contract, rather than hand-patching a post-lowered HBM SDSC.
2. Use an explicit internal-edge data-op/bridge that fetches the missing remote
   LX regions before the consumer reads them.
3. Allow direct LX aliasing only under a locality certificate proving
   producer-owned and restickify-needed logical regions match for every core.

Path 3 is the narrow Stage 3B case. The general LX-to-LX restickify case needs
path 1 or path 2.

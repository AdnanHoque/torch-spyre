# Project E — Bidirectional ring exploitation, closed by code reading

## TL;DR

The IBM AIU hardware has two counter-rotating data rings (CW + CCW,
128 B wide each). We hypothesized that codegen might use only one
direction, leaving up to 2× ring bandwidth unexploited. Closing this
project without a card-time probe, because **the ring-direction lever
isn't exposed at the torch_spyre layer**.

## What we found by reading the code

Searched the torch_spyre Python codegen, the C++ extension, and the
deeptools / dxp_standalone backend for any references to:

- ring direction (CW, CCW, clockwise, counter-clockwise)
- ring scheduling or transfer routing
- L3-LU / L3-SU direction parameters
- transfer-direction hints in any IR or runtime API

**Nothing.** Cross-core transfers are abstracted at the codegen layer
as "neighbor sharing" — the codegen only specifies *which cores* need
to share *what data*. The actual ring routing (CW vs CCW) is decided
at one of two layers neither of which torch_spyre touches:

1. The deeptools/dxp_standalone backend's instruction encoding
2. The RIU (Ring Interface Unit) hardware routing logic

## Implication

Even if we ran a probe and confirmed "only one ring is used today,"
**we couldn't ship a fix from torch_spyre.** Any optimization to
exploit both rings would require:

- Either: modifying deeptools to expose a per-transfer direction hint
  to codegen (large project, outside scope)
- Or: modifying the RIU routing (firmware / hardware change, way
  outside scope)

## What this confirms about the project pattern

This is the second time we've found that a measurable hardware lever
is hidden behind an abstraction layer that torch_spyre doesn't
control — the first being the cross-call weight preload mechanism
(slide 86 of the architecture doc) which doesn't fire for
torch.compile-driven matmul because it lives in a different runtime
path.

Both cases reinforce the same point: **we're working in one layer of
the stack, and the most productive future work is probably in the
layers we haven't touched** — Inductor scheduler, graph rewrites, or
the runtime / driver pathway.

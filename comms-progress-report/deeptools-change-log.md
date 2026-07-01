# Deeptools change log

Every backend change the comms-collectives work needs, why it is needed, and what
breaks without it. Changes land on the deeptools fork, never as a PR. Each entry is
verified against live code paths (not dead code) before it is proposed.

Status legend: **applied** (committed on a fork branch) · **designed** (rationale
below, not yet coded) · **verify** (a claimed-to-exist capability to confirm live
before building on it).

---

## 1. Tighten the mixed-SDSC routing — designed

**What.** In the DXP execution router (`dxp/dxp.cpp`), only route an SDSC through the
mixed data-op + DL path (`runDcgForDataOpsDlOps`) when it has *both* a populated
per-core schedule *and* a non-empty data-op list. A scheduled but pure-DL SDSC must
stay on the standalone-DL path (`runDcgForDlOpsStandalone`).

**Why.** With a scheduled-but-pure-DL SDSC, the mixed path is entered with no data-op
work and mis-lowers. This surfaced as a pure-DL DXP failure on the Granite prefill run
once the relayout classes were enabled.

**What breaks without it.** Enabling the collective path regresses ordinary scheduled
DL SDSCs — the block fails to compile before it ever reaches the relayout gap.

---

## 2. Generalize loop-scoped operand fetch off graph-inputs — designed (all-gather)

**What.** The loop-scoped operand-fetch generator (`runDcgForInputFetchNeighbor` →
`generatePcfgIRForDataOpInpFetch` → `fillDataDSCForInputFetchNeighbor`) is (a) reachable
only from a standalone tool (`dcg/tools/dcg_inpfetch_standalone.cpp`), not from the
compile pipeline, and (b) hard-checks `DsTypes::INPUT` (`inputNeighFetchOp.cpp`). The
change: wire it into the DXP path so an SDSC can reach it, and generalize it off the
hard-coded input type to the operand ds-type selected by the frontend's operand ordinal.

**Why.** The value-side attention operand that needs an all-gather is a `DsTypes::KERNEL`
matmul operand, not a graph input, so the only backend op that would do the correct
loop-scoped movement refuses it. Full-resident materialization is the alternative and it
fails: it needs the whole slice per co-splitting core, overflows LX, and falls back to an
HBM restickify or errors with a corelet-cardinality mismatch.

**What breaks without it.** The all-gather class cannot be lowered on-chip at all; the
operand keeps spilling to HBM. Route the replication through the already-live multicast
share-group path rather than adding a new ring primitive.

**Verify before building.** The cross-device `AllGather`/`AllReduce` ops exist but are
multi-AIU (world-size, rank) — *not* a single-AIU on-chip primitive. Do not wire to them.

---

## 3. Derive a layout-restickify from a coordinate+form mismatch — verify first (restickify)

**What.** The on-chip layout-transform op (`ReStickifyOpLx`) is a live, executable op,
but the relayout-insertion pass (`dxp/SdscRelayoutInsertion.cpp`) only ever synthesizes
the ownership-move op and assumes the input and output layouts match. A layout-changing
edge needs a pass that reads a source/dest stick-form contract and emits the transform.

**Why.** The scores → value edge changes the stick form (a dimension flips from a stick
dim to the contraction dim), so a pure ownership move is the wrong op.

**Verify before building.** The DSM base optimizer already self-promotes the HBM
restickify to its LX form (`baseOptimizer/lxopt.cpp` ~3798) when both operands are
LX-resident. Before adding any pass, A/B whether that promotion already fires on this
exact computed-activation edge. If it does, the frontend opfunc-swap prototype is
redundant and no new backend pass is needed. If it does not, this pass is the fix.

**What breaks without it (if the promotion does not fire).** The layout-restickify edge
keeps its HBM round-trip; the frontend rename lands on a real op but the pass that would
choose it never runs.

---

## Not a backend change: cardinality classification

The frontend currently hand-labels each collective (scatter / broadcast / all-gather).
That label is derivable in the backend from the two per-core distributions the relayout
insertion already reads, and one coordinate-driven path already covers both scatter and
multicast. Recommendation is to demote the frontend label to a backend-derived property
rather than grow a backend branch per class. Recorded here so it is not mistaken for a
missing backend capability — it is a contract simplification, not new code.

# DeepTools source mechanism

Pinned production source reference: DeepTools `ee2f97a`.

File:
`dsm/workOptimizer/baseOptimizer/workdivopt.cpp`

## 1. The global preload switch gates preload-favorable modeling

Lines 111-177 inspect static inputs per compute quanta. The decisive switch is
at lines 172-175:

```cpp
if (dscGlobal.doWeiPreload) {
  isPreloadFavorable = (totMemBoundStaticSize < myPreloadMemCap) ? 1 : 0;
} else {
  isPreloadFavorable = 0;
}
```

## 2. A preload-favorable quanta gets a static-vs-dynamic preference

Lines 852-904 calculate `preferStaticOrDynamic[cq]` when the quanta has a
static tensor and preload is favorable. The calculation uses dynamic tensor
size, static tensor size, memory capacity, and the number of memory units
(lines 865-899).

## 3. That preference becomes a hard split cap

Lines 1075-1086 form a product constraint over dimensions shared by static and
dynamic tensors:

```cpp
DimSplitConstraint newConstr(
    sIdDims, DimSplitConstraint::CombineOp::PRODUCT,
    DimSplitConstraint::CompareOp::LE,
    numMemUnits / meta_data_.preferStaticOrDynamic.at(cq));
dimSpConstraints.push_back(newConstr);
```

The M=512 stock decision log shows the resulting recovery-2 constraint is
`product({MB,X,Y,OUT}) <= 1`.

## 4. The hard cap rejects the copied 32-core partition

Lines 1481-1518 copy each candidate parent split and validate all hard
constraints. On rejection, the candidate split is reverted:

```cpp
isSplitLegal = currConstraint.checkConstraint(
    currSplit, DimSplitConstraint::Softness::HARD);
...
if (!isSplitLegal) {
  currSplit.at(dimId) = incomingSplit;
}
```

Observed decisions:

- Stock recovery 2: input candidate `MB=8, OUT=4`; hard cap `<=1`; output
  `MB=1, OUT=1`.
- `weipreload=0`: the hard cap is absent; output `MB=8, OUT=4`.

This precisely locates the gap in preload-aware work-division modeling for the
second FP8 scale-recovery operation. It is not a limitation of the FP8 BMM or
of two-corelet legality.

PoC workaround: `DT_OPT=autopilot=1,weipreload=0`.

Production-shaped direction: keep useful weight preload for the FP8 BMM, but
exclude or soften the static-reuse product cap for broadcast scale recovery
when its dynamic `M x N` output cost dominates the tiny scale tensor. Validate
the resulting split across non-Granite shapes.

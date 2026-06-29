# Reduced buf21 fit repro

Derived from buf21 attention value repro. Shrinks N x_=1 and in_=16 while preserving out_=128 and consumer mb=32 / producer out=32 mismatch. Intended to test whether current resident relayout lowering works when full post-relayout materialization fits in LX.

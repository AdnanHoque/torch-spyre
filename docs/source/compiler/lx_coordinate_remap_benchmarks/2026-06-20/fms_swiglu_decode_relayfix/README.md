# fms_swiglu_decode_relayfix

spyre-perf-suite `jamie/dev` FMS empty-weight fused SwiGLU, relay-fix build, B=1 S=1 E=4096.

This decode-shaped run is included as a control. The coordinate-remap planner does not emit remap rows for this shape, and kernel time is effectively unchanged.

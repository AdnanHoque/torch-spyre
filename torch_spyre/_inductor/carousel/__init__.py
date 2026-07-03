# Copyright 2026 The Torch-Spyre Authors. Apache 2.0
"""Carousel lowerings: rotating-operand matmul (prefill) and ring-native
flash-decode (decode). Frontend policy/plan/operator; backend realizes movement.
Research-track: gated behind TS_ENABLE_CAROUSEL. See carousel-rfc-impl/."""
from . import weight_carousel, kv_carousel, fold, comm_cost, reference, probes, roofline  # noqa: F401

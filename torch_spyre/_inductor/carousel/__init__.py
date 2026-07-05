# Copyright 2026 The Torch-Spyre Authors. Apache 2.0
"""Carousel lowerings: ring-native flash-decode fold.

Frontend policy/plan/operator; the backend realizes the movement. This is the
LSE ring-fold slice: fold.py holds the hop schedule + cost, reference.py the
device-free NumPy validator. Research-track. See carousel-rfc-impl/."""
from . import fold, reference  # noqa: F401

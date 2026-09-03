"""Cadence and radial-binning resolution calculations."""

from __future__ import annotations

import math


def displacement_mm(speed_kms: float, cadence_seconds: float) -> float:
    """Distance travelled between frames, in Mm."""

    return float(speed_kms) * float(cadence_seconds) / 1000.0


def quantization_speed_kms(radial_bin_mm: float, cadence_seconds: float) -> float:
    """Speed corresponding to exactly one radial bin per image interval."""

    return float(radial_bin_mm) * 1000.0 / float(cadence_seconds)


def endpoint_quantization_sigma_kms(radial_bin_mm: float, fit_interval_seconds: float) -> float:
    """Endpoint error scale for two independently half-bin-quantized positions."""

    return math.sqrt(2.0) * 0.5 * float(radial_bin_mm) * 1000.0 / float(fit_interval_seconds)


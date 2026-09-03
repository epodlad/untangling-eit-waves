"""Reproducible utilities for the *Untangling EIT Waves* Review."""

from .cadence import displacement_mm, quantization_speed_kms
from .energy import (
    front_fraction,
    full_sun_flux_erg_cm2_s,
    heating_requirement_percent,
    kinetic_energy_erg,
    velocity_perturbation_kms,
)

__all__ = [
    "displacement_mm",
    "quantization_speed_kms",
    "kinetic_energy_erg",
    "velocity_perturbation_kms",
    "full_sun_flux_erg_cm2_s",
    "heating_requirement_percent",
    "front_fraction",
]

__version__ = "1.0.0"

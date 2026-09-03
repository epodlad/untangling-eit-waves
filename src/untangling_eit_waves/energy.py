"""Transparent weak-compression front-energy estimates."""

from __future__ import annotations

import numpy as np


PROTON_MASS_G = 1.67262192369e-24
SOLAR_RADIUS_CM = 6.957e10
SECONDS_PER_DAY = 86400.0


def mass_density_g_cm3(electron_density_cm3: float, mass_per_electron: float = 1.2) -> float:
    """Approximate fully ionized coronal mass density from electron density."""

    return float(mass_per_electron) * PROTON_MASS_G * float(electron_density_cm3)


def velocity_perturbation_kms(phase_speed_kms: float, density_compression: float) -> float:
    """Weak-compression estimate delta-v ~= v_phase * delta-n/n."""

    return float(phase_speed_kms) * float(density_compression)


def kinetic_energy_erg(
    electron_density_cm3: float,
    phase_speed_kms: float,
    density_compression: float,
    volume_cm3: float,
    mass_per_electron: float = 1.2,
) -> float:
    """Return 0.5 rho delta-v^2 V in cgs units.

    This is a model-dependent kinetic estimate. It is not a thermal-energy
    measurement and does not include magnetic, conductive, or radiative terms.
    """

    rho = mass_density_g_cm3(electron_density_cm3, mass_per_electron)
    delta_v_cm_s = velocity_perturbation_kms(phase_speed_kms, density_compression) * 1.0e5
    return 0.5 * rho * delta_v_cm_s**2 * float(volume_cm3)


def kinetic_energy_range_erg(
    electron_density_cm3: float,
    phase_speed_kms: float,
    compression_range: tuple[float, float],
    volume_cm3: float,
) -> tuple[float, float]:
    """Evaluate the weak-compression energy at two compression bounds."""

    values = [
        kinetic_energy_erg(electron_density_cm3, phase_speed_kms, c, volume_cm3)
        for c in compression_range
    ]
    return float(np.min(values)), float(np.max(values))


def full_sun_flux_erg_cm2_s(
    event_energy_erg: float,
    events_per_day: float,
    solar_radius_cm: float = SOLAR_RADIUS_CM,
) -> float:
    """Full-Sun surface-averaged flux for an event population.

    This converts transported event energy into a surface flux.  Interpreting
    it as heat additionally requires an explicit dissipated fraction.
    """

    area_cm2 = 4.0 * np.pi * float(solar_radius_cm) ** 2
    return float(event_energy_erg) * float(events_per_day) / (area_cm2 * SECONDS_PER_DAY)


def heating_requirement_percent(
    flux_erg_cm2_s: float,
    requirement_erg_cm2_s: float = 1.0e5,
    dissipated_fraction: float = 1.0,
) -> float:
    """Percentage of a heating requirement after a stated dissipated fraction."""

    fraction = float(dissipated_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("dissipated_fraction must lie between zero and one")
    return 100.0 * fraction * float(flux_erg_cm2_s) / float(requirement_erg_cm2_s)


def front_fraction(front_energy_erg: float, released_energy_erg: float) -> float:
    """Return E_front/E_release for paired measurements of the same event."""

    released = float(released_energy_erg)
    if released <= 0.0:
        raise ValueError("released_energy_erg must be positive")
    return float(front_energy_erg) / released

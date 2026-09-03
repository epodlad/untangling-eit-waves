#!/usr/bin/env python3
"""Regenerate the data-derived and quantitative manuscript figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*parts: str) -> None:
    subprocess.run([sys.executable, *parts], cwd=ROOT, check=True)


def main() -> None:
    run("figure1/scripts/make_fig1_real_channels.py")
    run("figures/scripts/make_fig4_speed_measurements.py")
    run("figures/scripts/make_fig5_nemo_front_dimming.py")
    run("figures/scripts/make_fig6_observation_operator.py")
    run("figures/scripts/make_fig7_energy_geometry.py")
    run("figures/scripts/make_fig8_energy_partition_scaling.py")
    run("energy_partition/scripts/make_partition_figure.py")
    run(
        "radial_azimuthal/scripts/make_radial_azimuthal_phase_figure.py",
        "--rotation-summary",
        "radial_azimuthal/data/event_20170403_pattern_rotation_summary.json",
        "--csv",
        "radial_azimuthal/data/radial_azimuthal_phase_measurements.csv",
        "--output",
        "paper/figures/fig10_radial_azimuthal_phase.png",
    )


if __name__ == "__main__":
    main()

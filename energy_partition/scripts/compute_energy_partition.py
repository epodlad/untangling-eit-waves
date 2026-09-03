#!/usr/bin/env python3
"""Compute transparent energy and mass proxies for the 2017-04-03 pilot.

Observed products (AIA intensities, areas, and ridge speed) are kept separate
from DEM-inferred column emission measures and from quantities that require
line-of-sight depths, filling factors, or a wave model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "derived"
MASK_META = DERIVED / "event_20170403_mask_metadata.json"
DEM_CURVES = DERIVED / "event_20170403_region_dem_curves.npz"

KB = 1.380649e-16  # erg K^-1
MP = 1.67262192369e-24  # g
MU_E = 1.2  # mass per electron in proton-mass units
GMSUN_OVER_RSUN = 1.9077e15  # erg g^-1


def region_index(labels, region, stage):
    return int(np.where(labels == f"{region}:{stage}")[0][0])


def dem_moments(curves, region, stage, tlo=10**5.5, thi=10**7.3):
    labels = curves["labels"]
    idx = region_index(labels, region, stage)
    temp = curves["temperature_mid_K"]
    dt = np.diff(curves["temperature_edges_K"])
    sel = (temp >= tlo) & (temp <= thi)
    dem = np.clip(curves["dem_cm5_K"][idx], 0.0, None)
    em = np.sum(dem[sel] * dt[sel])
    tmean = np.sum(dem[sel] * temp[sel] * dt[sel]) / em
    return dem, em, tmean


def source_excess(curves, stage="early"):
    labels = curves["labels"]
    ipre = region_index(labels, "source", "pre")
    iev = region_index(labels, "source", stage)
    temp = curves["temperature_mid_K"]
    dt = np.diff(curves["temperature_edges_K"])
    pre = np.clip(curves["dem_cm5_K"][ipre], 0.0, None)
    event = np.clip(curves["dem_cm5_K"][iev], 0.0, None)
    excess = np.clip(event - pre, 0.0, None)
    em = np.sum(excess * dt)
    tmean = np.sum(excess * temp * dt) / em
    return em, tmean


def thermal_energy(em_column, temperature, area, depth, filling=1.0):
    # E_th = 3 n_e k_B T f A L and EM = n_e^2 f L.
    return 3.0 * KB * temperature * area * np.sqrt(em_column * filling * depth)


def dimming_mass(em0, em1, area, depth, filling=1.0):
    if em1 >= em0:
        return 0.0
    density_deficit = np.sqrt(filling / depth) * (np.sqrt(em0) - np.sqrt(em1))
    return MU_E * MP * density_deficit * area * depth


def front_kinetic(em0, em1, area, depth, speed):
    # Constant LOS depth and filling factor are assumed across the front.
    compression = np.sqrt(em1 / em0)
    ne0 = np.sqrt(em0 / depth)
    delta_v = speed * (compression - 1.0)
    volume = area * depth
    return 0.5 * (MU_E * MP * ne0) * delta_v**2 * volume


def main():
    meta = json.loads(MASK_META.read_text())
    curves = np.load(DEM_CURVES)

    # 1) Compact heating.  The early 14:20 snapshot is preferred because it
    # is unsaturated in the hot channels.  Area spans projected to a simple
    # radial foreshortening correction.  Depth is tied to sqrt(area).
    source_em, source_t = source_excess(curves, "early")
    source_info = meta["masks"]["source"]
    source_a_lo = source_info["projected_area_cm2"]
    source_a_hi = source_info["surface_area_cm2"]
    source_l_lo = np.sqrt(source_a_lo)
    source_l_hi = np.sqrt(source_a_hi)
    source_e_lo = thermal_energy(source_em, source_t, source_a_lo, source_l_lo)
    source_e_hi = thermal_energy(source_em, source_t, source_a_hi, source_l_hi)
    source_e_c = np.sqrt(source_e_lo * source_e_hi)

    # 2) Dimming/ejecta mass and mechanical-energy proxy.  The LOS depth is
    # not observed and is bracketed by 30--100 Mm.  The speed range is the
    # published SWAP off-limb feature range, not a direct velocity measurement
    # of the dimming plasma.
    _, dim_em0, _ = dem_moments(curves, "dimming", "pre")
    _, dim_em1, _ = dem_moments(curves, "dimming", "late")
    dim_info = meta["masks"]["dimming"]
    dim_area = [dim_info["projected_area_cm2"], dim_info["surface_area_cm2"]]
    dim_depth = [3.0e9, 1.0e10]
    masses = [dimming_mass(dim_em0, dim_em1, a, l) for a in dim_area for l in dim_depth]
    mass_lo, mass_hi = min(masses), max(masses)
    mass_c = np.sqrt(mass_lo * mass_hi)
    ejecta_v = [1.90e7, 3.70e7]  # cm s^-1
    mech = [0.5 * m * v**2 + GMSUN_OVER_RSUN * m for m in masses for v in ejecta_v]
    mech_lo, mech_hi = min(mech), max(mech)
    mech_c = np.sqrt(mech_lo * mech_hi)

    # 3) Front kinetic component in the reconstructed sector segment.  The full fixed shell is primary.  The
    # brightness-selected subset is an intentionally biased sensitivity test.
    speeds = [4.05e7, 4.84e7]  # AIA 171 ridge and published/reconstructed SWAP
    depths = [3.0e9, 1.0e10]
    front_values = {}
    for region in ["front_geometric", "front_bright_subset"]:
        _, em0, _ = dem_moments(curves, region, "pre")
        _, em1, _ = dem_moments(curves, region, "late")
        info = meta["masks"][region]
        areas = [info["projected_area_cm2"], info["surface_area_cm2"]]
        energies = [front_kinetic(em0, em1, a, l, v) for a in areas for l in depths for v in speeds]
        front_values[region] = {
            "em_pre_cm-5": em0,
            "em_late_cm-5": em1,
            "em_ratio": em1 / em0,
            "density_compression": np.sqrt(em1 / em0),
            "delta_n_over_n": np.sqrt(em1 / em0) - 1.0,
            "kinetic_low_erg": min(energies),
            "kinetic_central_erg": np.sqrt(min(energies) * max(energies)),
            "kinetic_high_erg": max(energies),
        }

    front_lo = front_values["front_geometric"]["kinetic_low_erg"]
    front_primary_hi = front_values["front_geometric"]["kinetic_high_erg"]
    front_selection_hi = front_values["front_bright_subset"]["kinetic_high_erg"]
    front_c = np.sqrt(front_lo * front_selection_hi)

    # Control-region temporal baseline.
    control_ratios = []
    for region in ["front_control_inner", "front_control_outer"]:
        _, c0, _ = dem_moments(curves, region, "pre")
        _, c1, _ = dem_moments(curves, region, "late")
        control_ratios.append(c1 / c0)
    control_ratio = float(np.sqrt(np.prod(control_ratios)))

    rows = [
        {
            "component": "compact_source_thermal_excess_early",
            "low": source_e_lo,
            "central": source_e_c,
            "high": source_e_hi,
            "units": "erg",
            "status": "DEM-inferred; volume/filling-factor model",
        },
        {
            "component": "dimming_mass_deficit",
            "low": mass_lo,
            "central": mass_c,
            "high": mass_hi,
            "units": "g",
            "status": "DEM-inferred; LOS-depth/filling-factor model",
        },
        {
            "component": "ejecta_mechanical_proxy",
            "low": mech_lo,
            "central": mech_c,
            "high": mech_hi,
            "units": "erg",
            "status": "model proxy; dimming mass paired with published feature speed",
        },
        {
            "component": "front_segment_kinetic_full_fixed_shell",
            "low": front_lo,
            "central": front_values["front_geometric"]["kinetic_central_erg"],
            "high": front_primary_hi,
            "units": "erg",
            "status": "weak-compression kinetic model",
        },
        {
            "component": "front_segment_kinetic_including_bright_subset_sensitivity",
            "low": front_lo,
            "central": front_c,
            "high": front_selection_hi,
            "units": "erg",
            "status": "sector-local model range; bright-subset upper sensitivity is selection-biased",
        },
    ]

    csv_path = DERIVED / "event_20170403_energy_partition.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    details = {
        "event": meta["event"],
        "times": {"pre": "14:04 UT", "early": "14:20 UT", "late": "14:44 UT"},
        "measured_inputs": {
            "aia171_ridge_speed_km_s": 405.0,
            "swap_comparison_speed_km_s": 484.0,
            "source_projected_area_cm2": source_a_lo,
            "source_foreshortening_corrected_area_cm2": source_a_hi,
        },
        "dem_inferences": {
            "source_early_excess_em_cm-5": source_em,
            "source_excess_dem_weighted_temperature_K": source_t,
            "dimming_em_ratio_late_pre": dim_em1 / dim_em0,
            "front_full": front_values["front_geometric"],
            "front_bright_subset": front_values["front_bright_subset"],
            "front_control_geometric_mean_em_ratio_late_pre": control_ratio,
        },
        "model_assumptions": {
            "filling_factor": 1.0,
            "dimming_and_front_depth_cm": depths,
            "dimming_feature_speed_cm_s": ejecta_v,
            "front_phase_speed_cm_s": speeds,
            "density_relation": "n_e=sqrt(EM/L)",
            "weak_front_relation": "delta_v=v_ph*(sqrt(EM1/EM0)-1)",
            "thermal_energy": "3*k_B*T*A*sqrt(EM*f*L)",
        },
        "results": rows,
        "interpretation_limit": (
            "The source thermal excess is not the total released magnetic energy, and the front mask "
            "covers one reconstructed sector segment rather than the full global front. Therefore "
            "E_front/E_release is not measured by this EUV-only pilot."
        ),
    }
    json_path = DERIVED / "event_20170403_energy_partition_details.json"
    json_path.write_text(json.dumps(details, indent=2))

    # Compact plot: numerical ranges and epistemic status are visible together.
    energy_rows = [r for r in rows if r["units"] == "erg"]
    labels = [
        "compact source\nthermal excess",
        "ejecta/dimming\nmechanical proxy",
        "front segment\nfixed shell",
        "front segment\nmask sensitivity",
    ]
    low = np.array([r["low"] for r in energy_rows])
    cen = np.array([r["central"] for r in energy_rows])
    high = np.array([r["high"] for r in energy_rows])
    colors = ["#d73027", "#fdae61", "#2c7bb6", "#74add1"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(labels))
    ax.errorbar(
        x,
        cen,
        yerr=np.vstack([cen - low, high - cen]),
        fmt="none",
        ecolor="0.25",
        elinewidth=1.5,
        capsize=5,
        zorder=1,
    )
    ax.scatter(x, cen, s=95, c=colors, edgecolor="white", linewidth=0.8, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(2e25, 3e30)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Energy [erg]")
    ax.set_title("2017-04-03 pilot: same-event energy proxies")
    ax.grid(axis="y", which="both", alpha=0.22)
    ax.text(
        0.01,
        0.02,
        "Points are not equally direct measurements; see status table.",
        transform=ax.transAxes,
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(DERIVED / "event_20170403_energy_partition.png", dpi=200)
    plt.close(fig)

    print(csv_path)
    print(json_path)
    for row in rows:
        print(
            f"{row['component']}: {row['low']:.3e} -- {row['high']:.3e} "
            f"{row['units']} (central {row['central']:.3e})"
        )
    print(
        "front full delta_n/n=",
        front_values["front_geometric"]["delta_n_over_n"],
        "bright=",
        front_values["front_bright_subset"]["delta_n_over_n"],
        "control ratio=",
        control_ratio,
    )


if __name__ == "__main__":
    main()

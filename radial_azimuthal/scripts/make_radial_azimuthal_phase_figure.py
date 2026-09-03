#!/usr/bin/env python3
"""Create the observational radial--azimuthal EUV-front phase diagram."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rad_s_to_deg_min(value: float) -> float:
    return value * 180.0 / math.pi * 60.0


def derived_tangential(radial: float, total: float | None) -> tuple[float | None, float | None]:
    if total is None or total < radial:
        return None, None
    tangential = math.sqrt(max(0.0, total * total - radial * radial))
    return tangential, tangential / radial


def literature_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pod = [
        ("1997-05-12 PB SE-1", "04:50-05:07", 258.0, 2.06e-4, 263.0, "SE"),
        ("1997-05-12 PB SE-2", "05:07-05:24", 225.0, 5.48e-4, 365.0, "SE"),
        ("1997-05-12 PB NW-2", "05:07-05:24", 225.0, 2.64e-4, 265.0, "NW"),
        ("1997-05-12 PB NW-3", "05:24-05:41", 258.0, 2.26e-4, 313.0, "NW"),
    ]
    for label, interval, radial, omega, total, component in pod:
        tangential, chi = derived_tangential(radial, total)
        rows.append(
            {
                "event_measurement": label,
                "interval_utc": interval,
                "radial_speed_km_s": radial,
                "radial_speed_uncertainty_km_s": "",
                "omega_deg_min": rad_s_to_deg_min(omega),
                "omega_rad_s": omega,
                "omega_low_deg_min": "",
                "omega_high_deg_min": "",
                "total_pattern_speed_km_s": total,
                "derived_tangential_pattern_speed_km_s": tangential,
                "derived_spirality_vphi_over_vr": chi,
                "sense_or_component": component,
                "measurement_status": "published measurement",
                "observable_definition": "weighted centre of a localized running-difference intensity region",
                "radial_speed_definition": "successive radial maxima in the same EIT image pair",
                "source": "Podladchikova & Berghmans (2005), Tables I-IV",
            }
        )

    rows.extend(
        [
            {
                "event_measurement": "1997-05-12 Attrill peak",
                "interval_utc": "05:07-05:24",
                "radial_speed_km_s": 225.0,
                "radial_speed_uncertainty_km_s": "",
                "omega_deg_min": 44.0 / 17.0,
                "omega_rad_s": math.radians(44.0) / (17.0 * 60.0),
                "omega_low_deg_min": "",
                "omega_high_deg_min": "",
                "total_pattern_speed_km_s": "",
                "derived_tangential_pattern_speed_km_s": "",
                "derived_spirality_vphi_over_vr": "",
                "sense_or_component": "CCW",
                "measurement_status": "published measurement",
                "observable_definition": "phase shift of a deprojected base-difference ring-intensity peak",
                "radial_speed_definition": "same-interval value from Podladchikova & Berghmans (2005)",
                "source": "Attrill et al. (2007), Section 3.1; radial speed from Podladchikova & Berghmans (2005)",
            },
            {
                "event_measurement": "1997-04-07 Attrill peaks",
                "interval_utc": "14:12-14:21",
                "radial_speed_km_s": 255.0,
                "radial_speed_uncertainty_km_s": 50.0,
                "omega_deg_min": -22.0 / 9.0,
                "omega_rad_s": -math.radians(22.0) / (9.0 * 60.0),
                "omega_low_deg_min": "",
                "omega_high_deg_min": "",
                "total_pattern_speed_km_s": "",
                "derived_tangential_pattern_speed_km_s": "",
                "derived_spirality_vphi_over_vr": "",
                "sense_or_component": "CW",
                "measurement_status": "published measurements, mixed radial interval",
                "observable_definition": "phase shift of two deprojected base-difference ring-intensity peaks",
                "radial_speed_definition": "event-average surface speed, not a simultaneous angular-fit speed",
                "source": "Attrill et al. (2007), Section 3.1; Thompson et al. (1999), Figure 4",
            },
        ]
    )
    return rows


def add_current_event(rows: list[dict[str, object]], summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows.append(
        {
            "event_measurement": "2017-04-03 AIA 171 lower sector",
            "interval_utc": "14:30-14:44",
            "radial_speed_km_s": 444.5,
            "radial_speed_uncertainty_km_s": 39.5,
            "omega_deg_min": summary["omega_deg_min_median"],
            "omega_rad_s": summary["omega_rad_s_median"],
            "omega_low_deg_min": summary["omega_deg_min_p16"],
            "omega_high_deg_min": summary["omega_deg_min_p84"],
            "total_pattern_speed_km_s": "",
            "derived_tangential_pattern_speed_km_s": "",
            "derived_spirality_vphi_over_vr": "",
            "sense_or_component": "not resolved",
            "measurement_status": "non-detection under analysis systematics",
            "observable_definition": "moving-annulus centroid of standardized AIA 171 intensity",
            "radial_speed_definition": "bracket between AIA 171 ridge (405) and published SWAP mean (484)",
            "source": "this work; reconstructed sector, original SWAP mask unavailable",
        }
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotation-summary", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = literature_rows()
    add_current_event(rows, args.rotation_summary)
    write_csv(args.csv, rows)

    fig, ax = plt.subplots(figsize=(9.8, 6.2), constrained_layout=True)
    pod = rows[:4]
    attrill = rows[4:6]
    current = rows[6]

    ax.scatter(
        [float(r["radial_speed_km_s"]) for r in pod],
        [float(r["omega_deg_min"]) for r in pod],
        s=72,
        marker="o",
        color="#167d9a",
        edgecolor="white",
        linewidth=0.8,
        label="Podladchikova--Berghmans localized regions",
        zorder=4,
    )
    point_tags = ("SE-1", "SE-2", "NW-2", "NW-3")
    label_offsets = ((5, 5), (5, 5), (5, 5), (5, -13))
    for r, tag, offset in zip(pod, point_tags, label_offsets, strict=True):
        ax.annotate(
            tag,
            (float(r["radial_speed_km_s"]), float(r["omega_deg_min"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            color="#0c4f61",
        )

    ax.errorbar(
        [float(r["radial_speed_km_s"]) for r in attrill],
        [float(r["omega_deg_min"]) for r in attrill],
        xerr=[
            [0.0, float(attrill[1]["radial_speed_uncertainty_km_s"])],
            [0.0, float(attrill[1]["radial_speed_uncertainty_km_s"])],
        ],
        fmt="^",
        ms=9,
        color="#c34a36",
        ecolor="#c34a36",
        capsize=3,
        label="Attrill deprojected ring peaks",
        zorder=5,
    )

    x = float(current["radial_speed_km_s"])
    y = float(current["omega_deg_min"])
    xerr = float(current["radial_speed_uncertainty_km_s"])
    ylow = y - float(current["omega_low_deg_min"])
    yhigh = float(current["omega_high_deg_min"]) - y
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=np.asarray([[ylow], [yhigh]]),
        fmt="D",
        ms=7.5,
        mfc="white",
        mec="#4d4d4d",
        ecolor="#4d4d4d",
        capsize=4,
        label="2017-04-03: systematic range crosses zero",
        zorder=6,
    )

    ax.axhline(0.0, color="0.3", lw=1.0, ls=":")
    ax.annotate(
        "Same 12 May interval;\nmethod changes the angular rate",
        xy=(225.0, 2.59),
        xytext=(305, 2.45),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="0.25"),
        fontsize=9,
        ha="left",
    )
    ax.text(
        0.02,
        0.025,
        "Observational phase space: azimuthal intensity-pattern drift, not plasma rotation\nand not a fast/slow MHD-mode boundary.",
        transform=ax.transAxes,
        fontsize=8.8,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.92),
    )
    ax.set_xlabel(r"Radial front speed $v_r$ (km s$^{-1}$)")
    ax.set_ylabel(r"Azimuthal pattern rate $\Omega_{\rm pat}$ (deg min$^{-1}$)")
    ax.set_xlim(150, 520)
    ax.set_ylim(-3.2, 3.35)
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.3, frameon=True, borderaxespad=0.0)
    ax.set_title("Radial propagation and azimuthal drift are distinct observables")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()

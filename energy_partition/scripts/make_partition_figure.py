#!/usr/bin/env python3
"""Create the manuscript figure for the same-event partition pilot."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "derived"
REPOSITORY = ROOT.parent
PAPER_FIGURE = REPOSITORY / "paper" / "figures" / "fig9_same_event_partition.pdf"


def main():
    masks = np.load(DERIVED / "event_20170403_masks.npz")
    with (DERIVED / "event_20170403_energy_partition.csv").open() as f:
        rows = {r["component"]: r for r in csv.DictReader(f)}

    fig = plt.figure(figsize=(11.4, 5.1))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.34)
    ax = fig.add_subplot(grid[0, 0])
    ratio = masks["ratio171_late"]
    image = ax.imshow(ratio, origin="lower", cmap="RdBu_r", vmin=0.72, vmax=1.28)
    regions = [
        ("source", "#f9a825", "compact source"),
        ("dimming", "#00acc1", "dimming"),
        ("front_geometric", "#43a047", "front sector"),
    ]
    for key, color, label in regions:
        ax.contour(masks[key].astype(float), levels=[0.5], colors=[color], linewidths=2.0)
        ax.plot([], [], color=color, lw=4, label=label)
    ax.set_xlim(350, 1020)
    ax.set_ylim(200, 940)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) Fixed regions in AIA 171 Å late/pre ratio", loc="left", fontsize=11)
    ax.legend(loc="lower left", framealpha=0.92, fontsize=9)
    cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.set_title("ratio", fontsize=8.5, pad=5)

    ax = fig.add_subplot(grid[0, 1])
    keys = [
        "compact_source_thermal_excess_early",
        "ejecta_mechanical_proxy",
        "front_segment_kinetic_full_fixed_shell",
        "front_segment_kinetic_including_bright_subset_sensitivity",
    ]
    labels = [
        "source thermal\nexcess",
        "ejecta/dimming\nmechanical proxy",
        "front sector\nfixed shell",
        "front sector\nmask sensitivity",
    ]
    colors = ["#d73027", "#fdae61", "#2c7bb6", "#74add1"]
    markers = ["o", "s", "D", "D"]
    low = np.array([float(rows[k]["low"]) for k in keys])
    cen = np.array([float(rows[k]["central"]) for k in keys])
    high = np.array([float(rows[k]["high"]) for k in keys])
    x = np.arange(len(keys))
    for i in range(len(keys)):
        ax.errorbar(
            x[i],
            cen[i],
            yerr=np.array([[cen[i] - low[i]], [high[i] - cen[i]]]),
            fmt=markers[i],
            markersize=8.5,
            color=colors[i],
            markeredgecolor="white",
            markeredgewidth=0.8,
            ecolor="0.3",
            elinewidth=1.4,
            capsize=4,
        )
    ax.set_yscale("log")
    ax.set_ylim(2e25, 3e30)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel("Energy [erg]")
    ax.set_title("(b) Inferred and model-dependent ranges", loc="left", fontsize=11)
    ax.grid(axis="y", which="both", alpha=0.22)
    ax.text(
        0.50,
        0.02,
        "Sector-local front values; not total wave energy.",
        transform=ax.transAxes,
        fontsize=8.2,
        color="0.3",
        ha="center",
    )

    fig.suptitle("Same-event pilot: AR 12644, 3 April 2017", fontsize=13, y=0.99)
    fig.savefig(DERIVED / "event_20170403_partition_manuscript.png", dpi=220, bbox_inches="tight")
    fig.savefig(PAPER_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

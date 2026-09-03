#!/usr/bin/env python3
"""Invert six AIA channels for fixed event regions.

The input intensities are region means in DN s^-1.  A 10% response/model
floor is used to regularize the inversion; it is not interpreted as an
independent random error between observing times.  Temporal changes are
therefore also checked against fixed control regions in the next analysis
stage.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from demregpy import dn2dem


ROOT = Path(__file__).resolve().parents[1]
PHOT = ROOT / "derived" / "event_20170403_region_photometry.csv"
RESP = ROOT / "calibration" / "aia_temperature_response_20170403.npz"
OUT = ROOT / "derived"

CHANNELS = np.array([94, 131, 171, 193, 211, 335])
REGIONS = [
    "source",
    "dimming",
    "front_geometric",
    "front_bright_subset",
    "front_control_inner",
    "front_control_outer",
]
STAGES = ["pre", "early", "late"]


def read_photometry():
    rows = list(csv.DictReader(PHOT.open()))
    indexed = {
        (r["region"], r["stage"], int(r["channel_angstrom"])): r for r in rows
    }
    values, errors, labels = [], [], []
    for region in REGIONS:
        for stage in STAGES:
            block = [indexed[(region, stage, int(ch))] for ch in CHANNELS]
            mean = np.array([float(r["mean_dn_s"]) for r in block])
            mad = np.array([float(r["spatial_mad_dn_s"]) for r in block])
            npix = np.array([float(r["n_pixels"]) for r in block])
            # The images have been reduced by four in each dimension, and the
            # corona is spatially correlated.  N_eff=Npix/16 is deliberately
            # conservative for the random component.  Ten per cent is a
            # response/model floor used by the inversion.
            n_eff = np.maximum(npix / 16.0, 1.0)
            # spatial_mad_dn_s is already the Gaussian-equivalent 1.4826*MAD.
            err = np.sqrt((mad / np.sqrt(n_eff)) ** 2 + (0.10 * mean) ** 2)
            values.append(mean)
            errors.append(err)
            labels.append((region, stage))
    return np.array(values), np.array(errors), labels


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dn, edn, labels = read_photometry()
    response = np.load(RESP)
    if not np.array_equal(response["channels_angstrom"], CHANNELS):
        raise RuntimeError("Temperature-response channels do not match photometry")
    tresp = response["response_channel_temperature"].T
    tresp_logt = response["logt"]

    # 0.1-dex bins over the interval to which the six coronal AIA channels
    # provide useful constraints for this quiet-coronal/eruptive event.
    temp_edges = np.logspace(5.5, 7.3, 19)
    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn,
        edn,
        tresp,
        tresp_logt,
        temp_edges,
        reg_tweak=1.0,
        max_iter=20,
        nmu=42,
        warn=False,
    )

    temp_mid = np.sqrt(temp_edges[:-1] * temp_edges[1:])
    dtemp = np.diff(temp_edges)
    dem_pos = np.clip(dem, 0.0, None)
    em = np.sum(dem_pos * dtemp, axis=-1)
    tmean = np.sum(dem_pos * temp_mid * dtemp, axis=-1) / np.maximum(em, 1e-99)
    residual_rms = np.sqrt(np.mean(((dn_reg - dn) / edn) ** 2, axis=-1))

    summary_path = OUT / "event_20170403_region_dem_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "region",
                "stage",
                "em_cm-5",
                "dem_weighted_temperature_K",
                "chisq",
                "normalized_residual_rms",
                "negative_dem_fraction",
            ]
        )
        for i, (region, stage) in enumerate(labels):
            writer.writerow(
                [
                    region,
                    stage,
                    f"{em[i]:.8e}",
                    f"{tmean[i]:.8e}",
                    f"{np.asarray(chisq).reshape(-1)[i]:.6g}",
                    f"{residual_rms[i]:.6g}",
                    f"{np.mean(dem[i] < 0):.6g}",
                ]
            )

    np.savez_compressed(
        OUT / "event_20170403_region_dem_curves.npz",
        labels=np.array([f"{a}:{b}" for a, b in labels]),
        channels_angstrom=CHANNELS,
        dn_s=dn,
        dn_error_s=edn,
        reconstructed_dn_s=dn_reg,
        temperature_edges_K=temp_edges,
        temperature_mid_K=temp_mid,
        dem_cm5_K=dem,
        dem_error_cm5_K=edem,
        elogt=elogt,
        chisq=chisq,
    )

    colors = {"pre": "0.35", "early": "#2b83ba", "late": "#d7191c"}
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4), sharex=True)
    for ax, region in zip(axes.flat, REGIONS):
        for stage in STAGES:
            i = labels.index((region, stage))
            ax.step(
                np.log10(temp_mid),
                dem_pos[i],
                where="mid",
                color=colors[stage],
                lw=1.7,
                label=stage,
            )
        ax.set_yscale("log")
        ax.set_ylim(1e14, 3e22)
        ax.set_title(region.replace("_", " "))
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\log_{10} T\;[\mathrm{K}]$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"DEM [cm$^{-5}$ K$^{-1}$]")
    fig.suptitle("AIA six-channel regional DEM: 2017-04-03")
    fig.tight_layout()
    fig.savefig(OUT / "event_20170403_region_dem.png", dpi=180)
    plt.close(fig)

    print(summary_path)
    for i, (region, stage) in enumerate(labels):
        print(
            f"{region:24s} {stage:5s} EM={em[i]:.3e} "
            f"T={tmean[i]/1e6:.2f} MK chi={np.asarray(chisq).reshape(-1)[i]:.2f}"
        )


if __name__ == "__main__":
    main()

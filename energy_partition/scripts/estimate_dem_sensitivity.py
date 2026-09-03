#!/usr/bin/env python3
"""Monte-Carlo sensitivity of regional DEM changes.

The experiment separates a channel-wise 10% calibration/response factor,
shared by all times, from conservative region-mean random scatter.  It tests
whether temporal ratios survive plausible perturbations; it is not a complete
systematic uncertainty budget for DEM inversion or line-of-sight geometry.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from demregpy import dn2dem


ROOT = Path(__file__).resolve().parents[1]
PHOT = ROOT / "derived" / "event_20170403_region_photometry.csv"
RESP = ROOT / "calibration" / "aia_temperature_response_20170403.npz"
CURVES = ROOT / "derived" / "event_20170403_region_dem_curves.npz"
OUT = ROOT / "derived" / "event_20170403_dem_sensitivity.json"
CHANNELS = np.array([94, 131, 171, 193, 211, 335])
NREP = 200


def percentile_dict(values):
    q = np.nanpercentile(values, [16, 50, 84])
    return {"p16": float(q[0]), "median": float(q[1]), "p84": float(q[2])}


def main():
    nominal = np.load(CURVES)
    labels = nominal["labels"]
    rows = list(csv.DictReader(PHOT.open()))
    index = {(r["region"], r["stage"], int(r["channel_angstrom"])): r for r in rows}
    dn, random_error = [], []
    for label in labels:
        region, stage = str(label).split(":")
        block = [index[(region, stage, int(ch))] for ch in CHANNELS]
        mean = np.array([float(r["mean_dn_s"]) for r in block])
        mad = np.array([float(r["spatial_mad_dn_s"]) for r in block])
        npix = np.array([float(r["n_pixels"]) for r in block])
        n_eff = np.maximum(npix / 16.0, 1.0)
        dn.append(mean)
        # spatial_mad_dn_s is already the Gaussian-equivalent 1.4826*MAD.
        random_error.append(mad / np.sqrt(n_eff))
    dn = np.array(dn)
    random_error = np.array(random_error)

    rng = np.random.default_rng(12644)
    shared_channel_factor = np.clip(rng.normal(1.0, 0.10, size=(NREP, 1, 6)), 0.65, 1.35)
    perturbed = dn[None, :, :] * shared_channel_factor
    perturbed += rng.normal(size=perturbed.shape) * random_error[None, :, :]
    perturbed = np.clip(perturbed, 1e-5, None)
    inv_error = np.sqrt(random_error[None, :, :] ** 2 + (0.10 * perturbed) ** 2)

    response = np.load(RESP)
    temp_edges = nominal["temperature_edges_K"]
    dem, _, _, chisq, _ = dn2dem(
        perturbed,
        inv_error,
        response["response_channel_temperature"].T,
        response["logt"],
        temp_edges,
        reg_tweak=1.0,
        max_iter=15,
        nmu=42,
        warn=False,
    )
    dem = np.clip(dem, 0.0, None)
    temp = nominal["temperature_mid_K"]
    dt = np.diff(temp_edges)
    em = np.sum(dem * dt, axis=-1)

    def ii(region, stage):
        return int(np.where(labels == f"{region}:{stage}")[0][0])

    metrics = {}
    for region in [
        "dimming",
        "front_geometric",
        "front_bright_subset",
        "front_control_inner",
        "front_control_outer",
    ]:
        ratio = em[:, ii(region, "late")] / em[:, ii(region, "pre")]
        metrics[f"{region}_em_ratio_late_pre"] = percentile_dict(ratio)

    pre = dem[:, ii("source", "pre"), :]
    early = dem[:, ii("source", "early"), :]
    excess = np.clip(early - pre, 0.0, None)
    excess_em = np.sum(excess * dt, axis=-1)
    excess_t = np.sum(excess * temp * dt, axis=-1) / np.maximum(excess_em, 1e-99)
    metrics["source_early_excess_em_cm-5"] = percentile_dict(excess_em)
    metrics["source_early_excess_dem_weighted_temperature_K"] = percentile_dict(excess_t)
    metrics["chisq_all"] = percentile_dict(np.asarray(chisq).reshape(-1))

    OUT.write_text(
        json.dumps(
            {
                "n_repetitions": NREP,
                "random_model": "region MAD with N_eff=Npix/16",
                "systematic_model": "10% channel factor shared across all stages",
                "scope": "DEM temporal-change sensitivity only; excludes geometry/depth/filling-factor uncertainty",
                "metrics": metrics,
            },
            indent=2,
        )
    )
    print(OUT)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

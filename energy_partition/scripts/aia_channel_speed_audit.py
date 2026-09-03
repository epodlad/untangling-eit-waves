#!/usr/bin/env python3
"""Exploratory same-sector AIA 171/193 front-speed audit.

The script keeps the measurement hierarchy explicit:

1. It constructs exposure-normalized, approximately differentially derotated
   sector profiles on the solar surface.
2. It finds a *joint* 171+193 ridge, so the two channels are compared at the
   same locations rather than being allowed to select unrelated features.
3. It also reports free single-channel ridge fits as a feature-selection
   diagnostic.  These are not yet paper measurements because the SWAP sectors
   have only been reconstructed from the published figure.

This is intentionally a small dependency-light analysis for the reduced
1024-pixel SIDC AIA Level-1.5 synoptic data set.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, map_coordinates


RSUN_KM = 696_340.0


@dataclass(frozen=True)
class Frame:
    time: datetime
    image: np.ndarray
    header: fits.Header
    path: str


def read_frames(data_root: Path, date: str, wavelength: int) -> list[Frame]:
    pattern = data_root / date / f"{wavelength:04d}" / "*.fits"
    frames: list[Frame] = []
    seen: set[datetime] = set()
    for path in sorted(glob.glob(str(pattern))):
        with fits.open(path, memmap=False) as hdul:
            header = hdul[1].header.copy()
            time = datetime.fromisoformat(header["DATE-OBS"].rstrip("Z"))
            if time in seen:
                continue
            seen.add(time)
            exposure = float(header["EXPTIME"])
            image = np.asarray(hdul[1].data, dtype=np.float64) / exposure
        frames.append(Frame(time, image, header, path))
    return frames


def observer_grid(header: fits.Header) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny = int(header["NAXIS2"])
    nx = int(header["NAXIS1"])
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    x = ((xx + 1.0 - float(header["CRPIX1"])) * float(header["CDELT1"]) + float(header.get("CRVAL1", 0.0))) / float(header["RSUN_OBS"])
    y = ((yy + 1.0 - float(header["CRPIX2"])) * float(header["CDELT2"]) + float(header.get("CRVAL2", 0.0))) / float(header["RSUN_OBS"])
    rr = x * x + y * y
    z = np.sqrt(np.clip(1.0 - rr, 0.0, None))
    z[rr > 1.0] = np.nan
    return x, y, z


def pixel_to_vector(header: fits.Header, xpix: float, ypix: float) -> np.ndarray:
    x = ((xpix + 1.0 - float(header["CRPIX1"])) * float(header["CDELT1"]) + float(header.get("CRVAL1", 0.0))) / float(header["RSUN_OBS"])
    y = ((ypix + 1.0 - float(header["CRPIX2"])) * float(header["CDELT2"]) + float(header.get("CRVAL2", 0.0))) / float(header["RSUN_OBS"])
    z = math.sqrt(max(0.0, 1.0 - x * x - y * y))
    v = np.array([x, y, z], dtype=float)
    return v / np.linalg.norm(v)


def derotate_to_reference(frame: Frame, reference_time: datetime, x: np.ndarray, y: np.ndarray, z: np.ndarray, rotation_rate_deg_day: float) -> np.ndarray:
    dt_days = (frame.time - reference_time).total_seconds() / 86400.0
    angle = np.deg2rad(rotation_rate_deg_day * dt_days)
    ca, sa = np.cos(angle), np.sin(angle)
    # A co-rotating point moves toward solar west (positive HPLN) with time.
    xt = x * ca + z * sa
    yt = y
    h = frame.header
    col = (xt * float(h["RSUN_OBS"]) - float(h.get("CRVAL1", 0.0))) / float(h["CDELT1"]) + float(h["CRPIX1"]) - 1.0
    row = (yt * float(h["RSUN_OBS"]) - float(h.get("CRVAL2", 0.0))) / float(h["CDELT2"]) + float(h["CRPIX2"]) - 1.0
    valid = np.isfinite(z) & (col >= 0.0) & (col <= frame.image.shape[1] - 1.0) & (row >= 0.0) & (row <= frame.image.shape[0] - 1.0)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    out[valid] = map_coordinates(frame.image, [row[valid], col[valid]], order=1, mode="nearest")
    return out


def sector_geometry(header: fits.Header, source_xy: tuple[float, float], target_xy: tuple[float, float], half_width_deg: float) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = observer_grid(header)
    points = np.stack([x, y, z], axis=-1)
    source = pixel_to_vector(header, *source_xy)
    target = pixel_to_vector(header, *target_xy)
    tangent = target - np.dot(target, source) * source
    tangent /= np.linalg.norm(tangent)

    dot = np.einsum("...i,i->...", points, source)
    distance_mm = np.arccos(np.clip(dot, -1.0, 1.0)) * RSUN_KM / 1000.0
    projected = points - dot[..., None] * source
    projected_norm = np.linalg.norm(projected, axis=-1)
    direction = projected / np.where(projected_norm[..., None] > 0, projected_norm[..., None], np.nan)
    angle = np.arccos(np.clip(np.einsum("...i,i->...", direction, tangent), -1.0, 1.0))
    mask = np.isfinite(z) & (angle <= np.deg2rad(half_width_deg))
    return distance_mm, mask


def radial_profiles(frames: list[Frame], source_xy: tuple[float, float], target_xy: tuple[float, float], half_width_deg: float, bin_mm: float, rotation_rate: float, reference_time: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    header = frames[0].header
    x, y, z = observer_grid(header)
    distance, sector = sector_geometry(header, source_xy, target_xy, half_width_deg)
    edges = np.arange(0.0, 1120.0 + bin_mm, bin_mm)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which = np.digitize(distance[sector], edges) - 1
    profiles = []
    for frame in frames:
        image = derotate_to_reference(frame, reference_time, x, y, z, rotation_rate)
        values = image[sector]
        profile = np.full(centers.shape, np.nan)
        for j in range(centers.size):
            v = values[which == j]
            if np.count_nonzero(np.isfinite(v)) >= 5:
                profile[j] = np.nanmedian(v)
        profiles.append(profile)
    times_min = np.array([(f.time - reference_time).total_seconds() / 60.0 for f in frames])
    return times_min, centers, np.asarray(profiles)


def prepare_derotated(frames: list[Frame], reference_time: datetime, rotation_rate: float) -> tuple[np.ndarray, np.ndarray, fits.Header]:
    header = frames[0].header
    x, y, z = observer_grid(header)
    images = [derotate_to_reference(frame, reference_time, x, y, z, rotation_rate) for frame in frames]
    times = np.array([(frame.time - reference_time).total_seconds() / 60.0 for frame in frames])
    return times, np.asarray(images, dtype=np.float32), header


def profiles_from_images(times: np.ndarray, images: np.ndarray, header: fits.Header, source_xy: tuple[float, float], target_xy: tuple[float, float], half_width_deg: float, bin_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance, sector = sector_geometry(header, source_xy, target_xy, half_width_deg)
    edges = np.arange(0.0, 1120.0 + bin_mm, bin_mm)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which = np.digitize(distance[sector], edges) - 1
    profiles = np.full((len(times), len(centers)), np.nan, dtype=float)
    for j in range(centers.size):
        use = which == j
        if np.count_nonzero(use) >= 5:
            profiles[:, j] = np.nanmedian(images[:, sector][:, use], axis=1)
    return times, centers, profiles


def standardize_profiles(profiles: np.ndarray, times: np.ndarray) -> np.ndarray:
    logp = np.log(np.clip(profiles, 1e-6, None))
    temporal_background = np.nanmedian(logp, axis=0, keepdims=True)
    signal = logp - temporal_background
    med = np.nanmedian(signal, axis=0, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(signal - med), axis=0, keepdims=True)
    standardized = signal / np.where(mad > 0.005, mad, 0.005)
    standardized = gaussian_filter(np.nan_to_num(standardized, nan=0.0), sigma=(0.45, 0.8))
    return np.clip(standardized, -6.0, 6.0)


def line_score(signal: np.ndarray, times: np.ndarray, distance: np.ndarray, velocity: float, r0: float, t0: float, time_window: tuple[float, float]) -> float:
    use = (times >= time_window[0]) & (times <= time_window[1])
    if np.count_nonzero(use) < 5:
        return -np.inf
    ridge = r0 + velocity * (times[use] - t0) * 60.0 / 1000.0
    if np.any((ridge < distance[0]) | (ridge > distance[-1])):
        return -np.inf
    vals = np.array([np.interp(r, distance, row) for r, row in zip(ridge, signal[use])])
    # Reward a persistent bright ridge, while limiting domination by one frame.
    return float(np.mean(np.clip(vals, -1.0, 4.0)) - 0.20 * np.std(vals))


def fit_ridge(signal: np.ndarray, times: np.ndarray, distance: np.ndarray, time_window: tuple[float, float], r0_bounds: tuple[float, float], t0: float = 0.0) -> tuple[float, float, float]:
    best = (-np.inf, np.nan, np.nan)
    for velocity in np.arange(220.0, 681.0, 5.0):
        for r0 in np.arange(r0_bounds[0], r0_bounds[1] + 0.1, 5.0):
            score = line_score(signal, times, distance, velocity, r0, t0, time_window)
            if score > best[0]:
                best = (score, velocity, r0)
    return best[1], best[2], best[0]


def aligned_channel(times_a: np.ndarray, prof_a: np.ndarray, times_b: np.ndarray, prof_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ia, ib, common = [], [], []
    for j, t in enumerate(times_a):
        k = int(np.argmin(np.abs(times_b - t)))
        if abs(times_b[k] - t) <= 0.25:  # AIA channels are sequential, not simultaneous.
            ia.append(j)
            ib.append(k)
            common.append(0.5 * (t + times_b[k]))
    return np.asarray(common), prof_a[ia], prof_b[ib]


def run_one(prepared171: tuple[np.ndarray, np.ndarray, fits.Header], prepared193: tuple[np.ndarray, np.ndarray, fits.Header], source_xy: tuple[float, float], target_xy: tuple[float, float], half_width: float, bin_mm: float, time_window: tuple[float, float], r0_bounds: tuple[float, float]) -> dict[str, float]:
    t171, images171, header171 = prepared171
    t193, images193, header193 = prepared193
    t171, dist, p171 = profiles_from_images(t171, images171, header171, source_xy, target_xy, half_width, bin_mm)
    t193, dist2, p193 = profiles_from_images(t193, images193, header193, source_xy, target_xy, half_width, bin_mm)
    if not np.allclose(dist, dist2):
        raise RuntimeError("Channel distance grids differ")
    times, p171, p193 = aligned_channel(t171, p171, t193, p193)
    s171, s193 = standardize_profiles(p171, times), standardize_profiles(p193, times)
    joint = 0.5 * (s171 + s193)
    vj, rj, sj = fit_ridge(joint, times, dist, time_window, r0_bounds)
    v171, r171, s171_best = fit_ridge(s171, times, dist, time_window, r0_bounds)
    v193, r193, s193_best = fit_ridge(s193, times, dist, time_window, r0_bounds)
    return {
        "target_y": target_xy[1], "half_width_deg": half_width, "bin_mm": bin_mm,
        "rotation_deg_day": 13.2, "joint_v_kms": vj, "joint_r0_mm": rj,
        "joint_score": sj, "v171_kms": v171, "r171_mm": r171,
        "score171": s171_best, "v193_kms": v193, "r193_mm": r193,
        "score193": s193_best,
    }


def plot_case(path: Path, prepared171: tuple[np.ndarray, np.ndarray, fits.Header], prepared193: tuple[np.ndarray, np.ndarray, fits.Header], source_xy: tuple[float, float], target_xy: tuple[float, float], half_width: float, bin_mm: float, time_window: tuple[float, float], r0_bounds: tuple[float, float], swap_velocity: float, swap_r0: float, title: str) -> None:
    t171, images171, header171 = prepared171
    t193, images193, header193 = prepared193
    t171, dist, p171 = profiles_from_images(t171, images171, header171, source_xy, target_xy, half_width, bin_mm)
    t193, _, p193 = profiles_from_images(t193, images193, header193, source_xy, target_xy, half_width, bin_mm)
    times, p171, p193 = aligned_channel(t171, p171, t193, p193)
    signals = [standardize_profiles(p171, times), standardize_profiles(p193, times)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True, constrained_layout=True)
    for ax, signal, wave in zip(axes, signals, (171, 193)):
        v, r0, score = fit_ridge(signal, times, dist, time_window, r0_bounds)
        mesh = ax.pcolormesh(times, dist, signal.T, shading="nearest", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        use = np.linspace(time_window[0], time_window[1], 100)
        ax.plot(use, r0 + v * use * 60.0 / 1000.0, color="gold", lw=2.3, label=f"AIA fit: {v:.0f} km/s")
        ax.plot(use, swap_r0 + swap_velocity * use * 60.0 / 1000.0, color="black", lw=2.0, ls="--", label=f"SWAP: {swap_velocity:.0f} km/s")
        ax.set_title(f"AIA {wave} Å; score={score:.2f}")
        ax.set_xlim(-4, 20)
        ax.set_ylim(80, 1050)
        ax.set_xlabel("Minutes after 14:28 UT")
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("Great-circle distance from source (Mm)")
    fig.colorbar(mesh, ax=axes, label="temporally standardized log intensity")
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("qa/aia_speed_audit"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Approximate reconstruction of the two April-3 published SWAP sectors.
    # The target-y ensemble explicitly carries the dominant geometrical uncertainty.
    settings = []
    for label, central_y, r0_bounds in [
        ("sector15_upper", 840.0, (160.0, 260.0)),
        ("sector16_lower", 760.0, (300.0, 410.0)),
    ]:
        for target_y in [central_y - 20.0, central_y, central_y + 20.0]:
            for half_width in [7.0, 9.0, 11.0]:
                settings.append((label, target_y, half_width, 10.0, r0_bounds))

    reference_time = datetime.fromisoformat("2017-04-03T14:28:00")
    prepared171 = prepare_derotated(read_frames(args.data_root, "2017-04-03", 171), reference_time, 13.2)
    prepared193 = prepare_derotated(read_frames(args.data_root, "2017-04-03", 193), reference_time, 13.2)

    rows = []
    for label, target_y, half_width, bin_mm, r0_bounds in settings:
        row = run_one(prepared171, prepared193, (890.0, 631.0), (512.0, target_y), half_width, bin_mm, (0.0, 18.0), r0_bounds)
        row["sector"] = label
        rows.append(row)

    csv_path = args.out / "event2_speed_ensemble.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_case(args.out / "sector15_central_time_distance.png", prepared171, prepared193, (890.0, 631.0), (512.0, 840.0), 9.0, 10.0, (0.0, 18.0), (160.0, 260.0), 457.0, 213.0, "2017-04-03 reconstructed SWAP sector 15 (upper)")
    plot_case(args.out / "sector16_central_time_distance.png", prepared171, prepared193, (890.0, 631.0), (512.0, 760.0), 9.0, 10.0, (0.0, 18.0), (300.0, 410.0), 484.0, 355.0, "2017-04-03 reconstructed SWAP sector 16 (lower)")

    for label in ["sector15_upper", "sector16_lower"]:
        subset = [r for r in rows if r["sector"] == label]
        print(label)
        for key in ["joint_v_kms", "v171_kms", "v193_kms"]:
            values = np.asarray([r[key] for r in subset])
            print(f"  {key}: median={np.median(values):.0f}; 16-84%={np.percentile(values,16):.0f}-{np.percentile(values,84):.0f}; range={values.min():.0f}-{values.max():.0f}")


if __name__ == "__main__":
    main()

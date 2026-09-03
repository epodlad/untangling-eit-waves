#!/usr/bin/env python3
"""Measure azimuthal drift of the 2017-04-03 AIA 171 A front pattern.

This is deliberately an image-pattern measurement.  It follows the radial
ridge already identified in the reconstructed lower SWAP sector, forms an
azimuthal intensity profile in a moving annulus, and measures the weighted
centroid of the positive profile.  It does not interpret the centroid drift as
plasma rotation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


RSUN_KM = 696_340.0


@dataclass(frozen=True)
class Frame:
    time: datetime
    image: np.ndarray
    header: fits.Header
    path: Path


@dataclass(frozen=True)
class TrackSetting:
    source_xy: tuple[float, float]
    target_xy: tuple[float, float]
    radial_speed_km_s: float
    r0_mm: float
    half_width_mm: float
    angular_window_deg: float
    smoothing_deg: float


def read_frames(data_root: Path, date: str, wavelength: int) -> list[Frame]:
    frames: list[Frame] = []
    seen: set[datetime] = set()
    for path in sorted((data_root / date / f"{wavelength:04d}").glob("*.fits")):
        with fits.open(path, memmap=False) as hdul:
            hdu = next(h for h in hdul if getattr(h, "data", None) is not None and h.data.ndim == 2)
            header = hdu.header.copy()
            time = datetime.fromisoformat(str(header["DATE-OBS"]).rstrip("Z"))
            if time in seen:
                continue
            seen.add(time)
            image = np.asarray(hdu.data, dtype=np.float64) / float(header["EXPTIME"])
        frames.append(Frame(time, image, header, path))
    return frames


def observer_grid(header: fits.Header) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = int(header["NAXIS2"]), int(header["NAXIS1"])
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
    vector = np.asarray([x, y, z], dtype=float)
    return vector / np.linalg.norm(vector)


def bilinear_sample(image: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    r1 = np.clip(r0 + 1, 0, image.shape[0] - 1)
    c1 = np.clip(c0 + 1, 0, image.shape[1] - 1)
    r0 = np.clip(r0, 0, image.shape[0] - 1)
    c0 = np.clip(c0, 0, image.shape[1] - 1)
    fr, fc = row - r0, col - c0
    return (
        image[r0, c0] * (1 - fr) * (1 - fc)
        + image[r1, c0] * fr * (1 - fc)
        + image[r0, c1] * (1 - fr) * fc
        + image[r1, c1] * fr * fc
    )


def prepare_derotated(
    frames: list[Frame], reference_time: datetime, rotation_rate_deg_day: float
) -> tuple[np.ndarray, np.ndarray, fits.Header]:
    header = frames[0].header
    x, y, z = observer_grid(header)
    output = []
    for frame in frames:
        dt_days = (frame.time - reference_time).total_seconds() / 86400.0
        angle = np.deg2rad(rotation_rate_deg_day * dt_days)
        xt = x * np.cos(angle) + z * np.sin(angle)
        col = (xt * float(header["RSUN_OBS"]) - float(header.get("CRVAL1", 0.0))) / float(header["CDELT1"]) + float(header["CRPIX1"]) - 1.0
        row = (y * float(header["RSUN_OBS"]) - float(header.get("CRVAL2", 0.0))) / float(header["CDELT2"]) + float(header["CRPIX2"]) - 1.0
        valid = np.isfinite(z) & (col >= 0) & (col <= frame.image.shape[1] - 1) & (row >= 0) & (row <= frame.image.shape[0] - 1)
        corrected = np.full(x.shape, np.nan, dtype=np.float64)
        corrected[valid] = bilinear_sample(frame.image, row[valid], col[valid])
        output.append(corrected)
    times = np.asarray([(frame.time - reference_time).total_seconds() / 60.0 for frame in frames])
    return times, np.asarray(output, dtype=np.float32), header


def gaussian_smooth_1d(values: np.ndarray, sigma: float, axis: int = 1) -> np.ndarray:
    radius = max(1, int(math.ceil(4 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis, values)


def local_geometry(
    header,
    source_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle distance and signed bearing about the eruption centre."""
    x, y, z = observer_grid(header)
    points = np.stack([x, y, z], axis=-1)
    source = pixel_to_vector(header, *source_xy)
    target = pixel_to_vector(header, *target_xy)
    e0 = target - np.dot(target, source) * source
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(source, e0)
    e1 /= np.linalg.norm(e1)

    dot = np.einsum("...i,i->...", points, source)
    distance_mm = np.arccos(np.clip(dot, -1.0, 1.0)) * RSUN_KM / 1000.0
    tangent = points - dot[..., None] * source
    norm = np.linalg.norm(tangent, axis=-1)
    direction = tangent / np.where(norm[..., None] > 0, norm[..., None], np.nan)
    bearing_deg = np.rad2deg(
        np.arctan2(
            np.einsum("...i,i->...", direction, e1),
            np.einsum("...i,i->...", direction, e0),
        )
    )
    return distance_mm, bearing_deg


def standardized_log_signal(images: np.ndarray) -> np.ndarray:
    log_image = np.log(np.clip(images.astype(float), 1e-6, None))
    median = np.nanmedian(log_image, axis=0, keepdims=True)
    residual = log_image - median
    mad = 1.4826 * np.nanmedian(np.abs(residual), axis=0, keepdims=True)
    signal = residual / np.where(mad > 0.008, mad, 0.008)
    return np.clip(signal, -6.0, 6.0)


def angular_profiles(
    times_min: np.ndarray,
    signal: np.ndarray,
    header,
    setting: TrackSetting,
    bin_deg: float = 1.0,
    geometry: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    distance, bearing = geometry if geometry is not None else local_geometry(header, setting.source_xy, setting.target_xy)
    centers = np.arange(-setting.angular_window_deg, setting.angular_window_deg + 0.1, bin_deg)
    edges = np.concatenate(([centers[0] - bin_deg / 2], centers + bin_deg / 2))
    profiles = np.full((len(times_min), len(centers)), np.nan)
    for i, (time_min, frame) in enumerate(zip(times_min, signal, strict=True)):
        ridge_mm = setting.r0_mm + setting.radial_speed_km_s * time_min * 60.0 / 1000.0
        annulus = (
            np.isfinite(distance)
            & np.isfinite(bearing)
            & (np.abs(distance - ridge_mm) <= setting.half_width_mm)
            & (np.abs(bearing) <= setting.angular_window_deg + bin_deg)
        )
        which = np.digitize(bearing[annulus], edges) - 1
        values = frame[annulus]
        for j in range(len(centers)):
            selected = values[which == j]
            if np.count_nonzero(np.isfinite(selected)) >= 5:
                profiles[i, j] = np.nanmedian(selected)
    sigma_bins = setting.smoothing_deg / bin_deg
    filled = np.where(np.isfinite(profiles), profiles, 0.0)
    weights = gaussian_smooth_1d(np.isfinite(profiles).astype(float), sigma_bins, axis=1)
    smooth = gaussian_smooth_1d(filled, sigma_bins, axis=1)
    smooth = smooth / np.where(weights > 0.15, weights, np.nan)
    return centers, smooth


def centroids(
    times_min: np.ndarray,
    angles_deg: np.ndarray,
    profiles: np.ndarray,
    fit_window: tuple[float, float],
    centroid_half_width_deg: float = 18.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    measured = np.full(len(times_min), np.nan)
    strength = np.full(len(times_min), np.nan)
    peak_angle = np.full(len(times_min), np.nan)
    previous = 0.0
    for i, (time_min, profile) in enumerate(zip(times_min, profiles, strict=True)):
        if not (fit_window[0] <= time_min <= fit_window[1]) or not np.any(np.isfinite(profile)):
            continue
        search = np.isfinite(profile) & (np.abs(angles_deg - previous) <= 20.0)
        if np.count_nonzero(search) < 3:
            search = np.isfinite(profile)
        candidates = np.where(search, profile, -np.inf)
        peak = int(np.argmax(candidates))
        peak_angle[i] = angles_deg[peak]
        use = np.isfinite(profile) & (np.abs(angles_deg - angles_deg[peak]) <= centroid_half_width_deg)
        local = profile[use]
        baseline = np.nanpercentile(local, 35)
        weights = np.clip(local - baseline, 0.0, None)
        if np.sum(weights) <= 0:
            continue
        measured[i] = float(np.sum(angles_deg[use] * weights) / np.sum(weights))
        strength[i] = float(np.nanmax(local) - baseline)
        previous = measured[i]
    return measured, strength, peak_angle


def fit_rate(times_min: np.ndarray, angle_deg: np.ndarray, strength: np.ndarray) -> tuple[float, float, float, int]:
    use = np.isfinite(angle_deg) & np.isfinite(strength)
    if np.count_nonzero(use) < 5:
        return math.nan, math.nan, math.nan, int(np.count_nonzero(use))
    x = times_min[use]
    y = angle_deg[use]
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (slope * x + intercept)
    dof = max(1, len(x) - 2)
    sigma2 = float(np.sum(residual**2) / dof)
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    slope_error = float(np.sqrt(covariance[0, 0]))
    rms = float(np.sqrt(np.mean(residual**2)))
    return float(slope), slope_error, rms, len(x)


def analyse_setting(
    times_min: np.ndarray,
    signal: np.ndarray,
    header,
    setting: TrackSetting,
    fit_window: tuple[float, float],
    geometry: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angles, profiles = angular_profiles(times_min, signal, header, setting, geometry=geometry)
    center, strength, peaks = centroids(times_min, angles, profiles, fit_window)
    slope, slope_error, rms, n = fit_rate(times_min, center, strength)
    result = {
        "source_x_px": setting.source_xy[0],
        "source_y_px": setting.source_xy[1],
        "target_y_px": setting.target_xy[1],
        "radial_speed_km_s": setting.radial_speed_km_s,
        "r0_mm": setting.r0_mm,
        "annulus_half_width_mm": setting.half_width_mm,
        "angular_window_deg": setting.angular_window_deg,
        "smoothing_deg": setting.smoothing_deg,
        "omega_deg_min": slope,
        "formal_omega_error_deg_min": slope_error,
        "omega_rad_s": slope * np.pi / 180.0 / 60.0,
        "fit_rms_deg": rms,
        "n_frames": n,
    }
    return result, angles, profiles, center, peaks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-time", default="2017-04-03T14:28:00")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    reference_time = datetime.fromisoformat(args.reference_time)
    frames = read_frames(args.data_root, "2017-04-03", 171)
    times, images, header = prepare_derotated(frames, reference_time, 13.2)
    keep = (times >= -2.0) & (times <= 20.0)
    times, images = times[keep], images[keep]
    signal = standardized_log_signal(images)
    fit_window = (2.0, 16.0)

    settings: list[TrackSetting] = []
    for target_y in (740.0, 760.0, 780.0):
        for speed in (355.0, 405.0, 455.0):
            for r0 in (330.0, 355.0, 380.0):
                for width in (30.0, 45.0, 60.0):
                    settings.append(
                        TrackSetting(
                            source_xy=(890.0, 631.0),
                            target_xy=(512.0, target_y),
                            radial_speed_km_s=speed,
                            r0_mm=r0,
                            half_width_mm=width,
                            angular_window_deg=42.0,
                            smoothing_deg=3.0,
                        )
                    )

    rows: list[dict[str, float]] = []
    preferred = TrackSetting((890.0, 631.0), (512.0, 760.0), 405.0, 355.0, 45.0, 42.0, 3.0)
    preferred_data = None
    geometry_cache = {
        target_y: local_geometry(header, (890.0, 631.0), (512.0, target_y))
        for target_y in (740.0, 760.0, 780.0)
    }
    for setting in settings:
        result, angles, profiles, center, peaks = analyse_setting(
            times,
            signal,
            header,
            setting,
            fit_window,
            geometry=geometry_cache[setting.target_xy[1]],
        )
        rows.append(result)
        if setting == preferred:
            preferred_data = (result, angles, profiles, center, peaks)

    csv_path = args.out / "event_20170403_pattern_rotation_ensemble.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    valid = np.asarray([row["omega_deg_min"] for row in rows], dtype=float)
    valid = valid[np.isfinite(valid)]
    summary = {
        "observable": "azimuthal drift of the AIA 171 A intensity-pattern centroid",
        "interpretation_limit": "pattern speed, not plasma velocity",
        "reference_time": args.reference_time,
        "fit_window_min": list(fit_window),
        "ensemble_size": len(rows),
        "valid_ensemble_size": int(len(valid)),
        "omega_deg_min_median": float(np.nanmedian(valid)),
        "omega_deg_min_p16": float(np.nanpercentile(valid, 16)),
        "omega_deg_min_p84": float(np.nanpercentile(valid, 84)),
        "omega_rad_s_median": float(np.nanmedian(valid) * np.pi / 180.0 / 60.0),
    }
    assert preferred_data is not None
    result, angles, profiles, center, peaks = preferred_data
    summary["preferred"] = result
    (args.out / "event_20170403_pattern_rotation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    mesh = ax0.pcolormesh(times, angles, profiles.T, shading="nearest", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax0.plot(times, center, "o-", color="#ffd54a", lw=1.8, ms=4.2, label="weighted centroid")
    use = np.isfinite(center) & (times >= fit_window[0]) & (times <= fit_window[1])
    if np.count_nonzero(use) >= 2:
        xline = np.linspace(fit_window[0], fit_window[1], 100)
        slope = result["omega_deg_min"]
        intercept = float(np.nanmean(center[use] - slope * times[use]))
        ax0.plot(xline, slope * xline + intercept, color="black", ls="--", lw=1.5, label=f"fit {slope:+.2f} deg/min")
    ax0.set_xlabel("Minutes after 14:28 UT")
    ax0.set_ylabel("Bearing from reconstructed sector 16 (deg)")
    ax0.set_title("Moving-annulus azimuthal profile")
    ax0.legend(loc="best", fontsize=9)
    fig.colorbar(mesh, ax=ax0, label="standardized log intensity")

    ax1.hist(valid, bins=40, color="#168aad", alpha=0.85)
    ax1.axvline(summary["omega_deg_min_median"], color="black", lw=2, label="ensemble median")
    ax1.axvspan(summary["omega_deg_min_p16"], summary["omega_deg_min_p84"], color="#ffd166", alpha=0.45, label="16-84%")
    ax1.axvline(0.0, color="0.4", lw=1, ls=":")
    ax1.set_xlabel("Pattern angular rate (deg/min)")
    ax1.set_ylabel("Number of analysis variants")
    ax1.set_title("Sensitivity to centre, ridge and annulus")
    ax1.legend(loc="best", fontsize=9)
    fig.suptitle("2017-04-03 AIA 171 A: azimuthal drift is an image-pattern observable")
    fig.savefig(args.out / "event_20170403_pattern_rotation_diagnostic.png", dpi=190)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

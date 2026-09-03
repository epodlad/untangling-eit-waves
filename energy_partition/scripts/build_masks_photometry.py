#!/usr/bin/env python3
"""Build fixed source, dimming, and front masks and extract six-channel photometry."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter, label
from skimage.morphology import remove_small_objects

from aia_geometry import (
    derotate_to_reference,
    distance_and_sector,
    observer_grid,
    projected_pixel_area_cm2,
    read_frame,
)


CHANNELS = [94, 131, 171, 193, 211, 335]
TAGS = {"pre": "140400", "early": "142000", "late": "144400"}
SOURCE_XY = (887.6, 637.3)
SECTOR16_TARGET_XY = (512.0, 760.0)
REFERENCE_TIME = datetime.fromisoformat("2017-04-03T14:28:00")


def find_path(root: Path, channel: int, tag: str) -> Path:
    matches = list((root / f"{channel:04d}").glob(f"*_{tag}.fits"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {channel} Å file with tag {tag}; found {len(matches)}")
    return matches[0]


def keep_seeded_components(grow: np.ndarray, seed: np.ndarray, min_size: int) -> np.ndarray:
    components, count = label(grow)
    keep = np.zeros_like(grow, dtype=bool)
    for component_id in range(1, count + 1):
        component = components == component_id
        if component.sum() >= min_size and np.any(component & seed):
            keep |= component
    return keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        stage: {
            channel: read_frame(find_path(args.data_root, channel, tag))
            for channel in CHANNELS
        }
        for stage, tag in TAGS.items()
    }
    images = {
        stage: {
            channel: derotate_to_reference(frame, REFERENCE_TIME)
            for channel, frame in stage_frames.items()
        }
        for stage, stage_frames in frames.items()
    }

    header = frames["pre"][171].header
    _, _, mu = observer_grid(header)
    distance_mm, sector16 = distance_and_sector(
        header, SOURCE_XY, SECTOR16_TARGET_XY, half_width_deg=9.0
    )
    disk = np.isfinite(mu) & (mu >= 0.08)

    smooth = {
        stage: {channel: gaussian_filter(image, 1.2) for channel, image in channel_images.items()}
        for stage, channel_images in images.items()
    }
    ratio94_early = smooth["early"][94] / np.maximum(smooth["pre"][94], 0.20)
    ratio131_early = smooth["early"][131] / np.maximum(smooth["pre"][131], 0.50)
    diff94_early = smooth["early"][94] - smooth["pre"][94]
    diff131_early = smooth["early"][131] - smooth["pre"][131]
    ratio171_late = smooth["late"][171] / np.maximum(smooth["pre"][171], 5.0)

    source = (
        disk
        & (distance_mm <= 100.0)
        & (
            ((ratio94_early >= 1.50) & (diff94_early >= 5.0))
            | ((ratio131_early >= 1.50) & (diff131_early >= 15.0))
        )
    )
    source = remove_small_objects(binary_closing(source, iterations=1), max_size=5)
    source = binary_dilation(source, iterations=1) & disk & (distance_mm <= 110.0)

    dimming_domain = disk & (distance_mm >= 65.0) & (distance_mm <= 350.0)
    dimming_grow = dimming_domain & (ratio171_late <= 0.82)
    dimming_seed = dimming_domain & (ratio171_late <= 0.70) & (distance_mm <= 280.0)
    dimming = keep_seeded_components(dimming_grow, dimming_seed, min_size=15)
    dimming = remove_small_objects(binary_closing(dimming, iterations=1), max_size=19)
    dimming &= ~source

    front_geometric = disk & sector16 & (distance_mm >= 780.0) & (distance_mm <= 850.0)
    front_bright = front_geometric & (ratio171_late >= 1.02)
    control_inner = disk & sector16 & (distance_mm >= 700.0) & (distance_mm < 760.0)
    control_outer = disk & sector16 & (distance_mm > 870.0) & (distance_mm <= 930.0)

    masks = {
        "source": source,
        "dimming": dimming,
        "front_geometric": front_geometric,
        "front_bright_subset": front_bright,
        "front_control_inner": control_inner,
        "front_control_outer": control_outer,
    }
    if any(mask.sum() == 0 for mask in masks.values()):
        empty = [name for name, mask in masks.items() if mask.sum() == 0]
        raise RuntimeError(f"Empty masks: {empty}")

    np.savez_compressed(
        args.output_dir / "event_20170403_masks.npz",
        **{name: mask.astype(np.uint8) for name, mask in masks.items()},
        distance_mm=distance_mm.astype(np.float32),
        mu=mu.astype(np.float32),
        ratio171_late=ratio171_late.astype(np.float32),
    )

    area_pixel = projected_pixel_area_cm2(header)
    mask_metadata = {}
    for name, mask in masks.items():
        mask_metadata[name] = {
            "n_pixels": int(mask.sum()),
            "projected_area_cm2": float(mask.sum() * area_pixel),
            "surface_area_cm2": float(np.nansum(area_pixel / mu[mask])),
            "distance_min_Mm": float(np.nanmin(distance_mm[mask])),
            "distance_max_Mm": float(np.nanmax(distance_mm[mask])),
            "mu_median": float(np.nanmedian(mu[mask])),
        }
    metadata = {
        "event": "2017-04-03 AR 12644",
        "reference_time": REFERENCE_TIME.isoformat(),
        "source_xy_reduced_pixels": SOURCE_XY,
        "sector16_target_xy_reduced_pixels": SECTOR16_TARGET_XY,
        "sector16_half_width_deg": 9.0,
        "front_radial_bounds_Mm": [780.0, 850.0],
        "mask_definitions": {
            "source": "Within 110 Mm; early 94/131 hot-channel increase; morphology cleaned.",
            "dimming": "171 Å late/pre hysteresis: seed <=0.70, grow <=0.82, 65-350 Mm.",
            "front_geometric": "Full reconstructed SWAP sector 16 shell at 780-850 Mm.",
            "front_bright_subset": "Diagnostic subset of geometric shell with 171 late/pre >=1.02.",
        },
        "masks": mask_metadata,
    }
    (args.output_dir / "event_20170403_mask_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    rows = []
    for region, mask in masks.items():
        for stage in TAGS:
            for channel in CHANNELS:
                values = images[stage][channel][mask]
                values = values[np.isfinite(values)]
                exposure = float(frames[stage][channel].header["EXPTIME"])
                raw_equivalent = values * exposure
                median = float(np.nanmedian(values))
                mad = float(1.4826 * np.nanmedian(np.abs(values - median)))
                rows.append(
                    {
                        "region": region,
                        "stage": stage,
                        "nominal_tag": TAGS[stage],
                        "date_obs": frames[stage][channel].time.isoformat(),
                        "channel_angstrom": channel,
                        "exposure_s": exposure,
                        "n_pixels": int(values.size),
                        "mean_dn_s": float(np.nanmean(values)),
                        "median_dn_s": median,
                        "spatial_mad_dn_s": mad,
                        "saturation_fraction_gt15000_dn": float(np.mean(raw_equivalent >= 15000.0)),
                    }
                )
    with (args.output_dir / "event_20170403_region_photometry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    label_image = np.zeros_like(mu, dtype=int)
    label_image[dimming] = 1
    label_image[front_geometric] = 2
    label_image[source] = 3
    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    ax.imshow(ratio171_late, origin="lower", cmap="RdBu_r", vmin=0.70, vmax=1.30)
    overlay = np.ma.masked_where(label_image == 0, label_image)
    ax.imshow(overlay, origin="lower", cmap=ListedColormap(["#25c6da", "#77dd77", "#ffd84d"]), alpha=0.55, vmin=1, vmax=3)
    for name, color, mask in [
        ("dimming", "#00acc1", dimming),
        ("front", "#43a047", front_geometric),
        ("compact source", "#f9a825", source),
    ]:
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.4)
        ax.plot([], [], color=color, lw=4, label=name)
    ax.set_xlim(350, 1020)
    ax.set_ylim(200, 940)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_title("2017-04-03 AIA 171 Å late/pre ratio with fixed energy-partition masks")
    fig.savefig(args.output_dir / "event_20170403_masks.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

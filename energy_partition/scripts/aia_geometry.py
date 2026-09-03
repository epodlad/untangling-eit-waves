"""Small geometry helpers for the reduced full-disk AIA pilot data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates


RSUN_KM = 696_340.0


@dataclass(frozen=True)
class Frame:
    time: datetime
    image_dn_s: np.ndarray
    image_dn: np.ndarray
    header: fits.Header
    path: Path


def read_frame(path: Path) -> Frame:
    with fits.open(path, memmap=False) as hdul:
        hdu = next(h for h in hdul if getattr(h, "data", None) is not None and h.data.ndim == 2)
        header = hdu.header.copy()
        image_dn = np.asarray(hdu.data, dtype=np.float64)
    exposure = float(header["EXPTIME"])
    time = datetime.fromisoformat(str(header["DATE-OBS"]).replace("Z", ""))
    return Frame(time, image_dn / exposure, image_dn, header, path)


def observer_grid(header: fits.Header) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = int(header["NAXIS2"]), int(header["NAXIS1"])
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    x = (
        (xx + 1 - float(header["CRPIX1"])) * float(header["CDELT1"])
        + float(header.get("CRVAL1", 0.0))
    ) / float(header["RSUN_OBS"])
    y = (
        (yy + 1 - float(header["CRPIX2"])) * float(header["CDELT2"])
        + float(header.get("CRVAL2", 0.0))
    ) / float(header["RSUN_OBS"])
    rr = x * x + y * y
    z = np.sqrt(np.clip(1 - rr, 0, None))
    z[rr > 1] = np.nan
    return x, y, z


def pixel_to_vector(header: fits.Header, xpix: float, ypix: float) -> np.ndarray:
    x = (
        (xpix + 1 - float(header["CRPIX1"])) * float(header["CDELT1"])
        + float(header.get("CRVAL1", 0.0))
    ) / float(header["RSUN_OBS"])
    y = (
        (ypix + 1 - float(header["CRPIX2"])) * float(header["CDELT2"])
        + float(header.get("CRVAL2", 0.0))
    ) / float(header["RSUN_OBS"])
    z = math.sqrt(max(0.0, 1 - x * x - y * y))
    vector = np.asarray([x, y, z], dtype=float)
    return vector / np.linalg.norm(vector)


def derotate_to_reference(
    frame: Frame,
    reference_time: datetime,
    rotation_rate_deg_day: float = 13.2,
) -> np.ndarray:
    x, y, z = observer_grid(frame.header)
    dt_days = (frame.time - reference_time).total_seconds() / 86400
    angle = np.deg2rad(rotation_rate_deg_day * dt_days)
    xt = x * np.cos(angle) + z * np.sin(angle)
    yt = y
    header = frame.header
    col = (
        (xt * float(header["RSUN_OBS"]) - float(header.get("CRVAL1", 0.0)))
        / float(header["CDELT1"])
        + float(header["CRPIX1"])
        - 1
    )
    row = (
        (yt * float(header["RSUN_OBS"]) - float(header.get("CRVAL2", 0.0)))
        / float(header["CDELT2"])
        + float(header["CRPIX2"])
        - 1
    )
    valid = (
        np.isfinite(z)
        & (col >= 0)
        & (col <= frame.image_dn_s.shape[1] - 1)
        & (row >= 0)
        & (row <= frame.image_dn_s.shape[0] - 1)
    )
    output = np.full(x.shape, np.nan, dtype=np.float64)
    output[valid] = map_coordinates(frame.image_dn_s, [row[valid], col[valid]], order=1, mode="nearest")
    return output


def distance_and_sector(
    header: fits.Header,
    source_xy: tuple[float, float],
    target_xy: tuple[float, float],
    half_width_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = observer_grid(header)
    points = np.stack([x, y, z], axis=-1)
    source = pixel_to_vector(header, *source_xy)
    target = pixel_to_vector(header, *target_xy)
    tangent = target - np.dot(target, source) * source
    tangent /= np.linalg.norm(tangent)
    dot = np.einsum("...i,i->...", points, source)
    distance_mm = np.arccos(np.clip(dot, -1, 1)) * RSUN_KM / 1000
    projected = points - dot[..., None] * source
    norm = np.linalg.norm(projected, axis=-1)
    direction = projected / np.where(norm[..., None] > 0, norm[..., None], np.nan)
    angle = np.arccos(np.clip(np.einsum("...i,i->...", direction, tangent), -1, 1))
    sector = np.isfinite(z) & (angle <= np.deg2rad(half_width_deg))
    return distance_mm, sector


def projected_pixel_area_cm2(header: fits.Header) -> float:
    pixel_radian = np.deg2rad(abs(float(header["CDELT1"])) / 3600)
    pixel_cm = float(header["DSUN_OBS"]) * 100 * pixel_radian
    return pixel_cm**2

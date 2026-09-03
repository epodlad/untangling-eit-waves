"""Small WCS helpers for pixel-to-Sun coordinate transfer.

These routines handle pointing, roll, plate scale, and reference-pixel
differences encoded by FITS WCS. They do not perform solar differential
rotation. Register time-separated frames to a common time before extracting a
sector. For rigorous transformations between substantially separated
observers, install the optional SunPy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io.fits import Header
from astropy.wcs import WCS


REQUIRED_WCS_KEYS = (
    "CRPIX1",
    "CRPIX2",
    "CRVAL1",
    "CRVAL2",
    "CDELT1",
    "CDELT2",
    "CTYPE1",
    "CTYPE2",
    "RSUN_OBS",
)


@dataclass(frozen=True)
class WCSCheck:
    valid: bool
    missing: tuple[str, ...]
    ctype: tuple[str, str]
    scale: tuple[float, float]


def check_wcs(header: Header) -> WCSCheck:
    """Report whether the minimum solar image WCS is present."""

    missing = tuple(key for key in REQUIRED_WCS_KEYS if key not in header)
    ctype = (str(header.get("CTYPE1", "")), str(header.get("CTYPE2", "")))
    scale = (float(header.get("CDELT1", np.nan)), float(header.get("CDELT2", np.nan)))
    solar_axes = ctype[0].startswith("HPLN") and ctype[1].startswith("HPLT")
    finite_scale = bool(np.all(np.isfinite(scale)) and np.all(np.asarray(scale) != 0))
    return WCSCheck(not missing and solar_axes and finite_scale, missing, ctype, scale)


def require_valid_wcs(header: Header) -> None:
    check = check_wcs(header)
    if not check.valid:
        raise ValueError(
            f"Incomplete/unsupported solar WCS: missing={check.missing}, "
            f"CTYPE={check.ctype}, CDELT={check.scale}"
        )


def pixel_to_world(header: Header, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert zero-based pixels to the two native helioprojective WCS axes."""

    require_valid_wcs(header)
    world = WCS(header, naxis=2).all_pix2world(np.asarray(x), np.asarray(y), 0)
    return np.asarray(world[0]), np.asarray(world[1])


def world_to_pixel(header: Header, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert native helioprojective WCS coordinates to zero-based pixels."""

    require_valid_wcs(header)
    pixel = WCS(header, naxis=2).all_world2pix(np.asarray(lon), np.asarray(lat), 0)
    return np.asarray(pixel[0]), np.asarray(pixel[1])


def transfer_pixels(
    source_header: Header,
    target_header: Header,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transfer points between near-simultaneous, near-Earth solar images.

    Pixel coordinates are never copied directly. They are converted through
    the physical helioprojective axes of each FITS WCS.
    """

    lon, lat = pixel_to_world(source_header, x, y)
    return world_to_pixel(target_header, lon, lat)


def plate_scale_arcsec(header: Header) -> tuple[float, float]:
    """Return absolute plate scales in arcsec/pixel regardless of CUNIT."""

    require_valid_wcs(header)
    wcs = WCS(header, naxis=2)
    units = [str(unit) for unit in wcs.wcs.cunit]
    factors = []
    for unit, value in zip(units, wcs.wcs.cdelt):
        if unit in ("deg", "degree"):
            factors.append(abs(float(value)) * 3600.0)
        elif unit in ("arcsec", "arcsecond"):
            factors.append(abs(float(value)))
        else:
            raise ValueError(f"Unsupported WCS angular unit: {unit}")
    return factors[0], factors[1]

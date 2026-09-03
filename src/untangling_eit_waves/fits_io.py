"""FITS I/O for the compressed SIDC AIA synoptic products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class Frame:
    """One exposure-normalized image and its science header."""

    path: Path
    time: datetime
    image: np.ndarray
    header: fits.Header


def parse_fits_time(value: str) -> datetime:
    """Parse an ISO FITS timestamp and always return an aware UTC datetime."""

    text = str(value).strip().replace("Z", "+00:00")
    result = datetime.fromisoformat(text)
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def science_hdu(hdul: fits.HDUList):
    """Return the first 2-D image HDU.

    SIDC synoptic files store the compressed image in extension 1 and leave the
    primary HDU empty. Searching for the first 2-D image is more robust than
    hard-coding an extension number.
    """

    for hdu in hdul:
        data = getattr(hdu, "data", None)
        if data is not None and np.ndim(data) == 2:
            return hdu
    raise ValueError("FITS file contains no 2-D science image")


def read_frame(path: str | Path, normalize_exposure: bool = True) -> Frame:
    """Read a compressed AIA frame and optionally divide by exposure time."""

    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        hdu = science_hdu(hdul)
        header = hdu.header.copy()
        image = np.asarray(hdu.data, dtype=np.float64)
    if normalize_exposure:
        exposure = float(header.get("EXPTIME", 1.0))
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError(f"Invalid exposure time in {path}: {exposure}")
        image = image / exposure
    return Frame(path, parse_fits_time(header["DATE-OBS"]), image, header)


def read_unique_frames(paths: Iterable[str | Path]) -> list[Frame]:
    """Read frames in time order and discard duplicate observation times."""

    by_time: dict[datetime, Frame] = {}
    for path in sorted(map(Path, paths)):
        frame = read_frame(path)
        by_time.setdefault(frame.time, frame)
    return [by_time[key] for key in sorted(by_time)]


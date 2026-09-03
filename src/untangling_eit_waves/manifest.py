"""Manifest and checksum helpers for the archived AIA FITS subset."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .fits_io import read_frame
from .wcs_geometry import check_wcs


SIDC_ROOT = "https://sdo.oma.be/data/aia_synoptic"


def sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_url(date: str, channel: str, filename: str) -> str:
    year, month, day = date.split("-")
    return f"{SIDC_ROOT}/{channel}/{year}/{month}/{day}/{filename}"


def build_rows(data_root: str | Path) -> list[dict[str, str | int | float]]:
    root = Path(data_root)
    rows: list[dict[str, str | int | float]] = []
    seen_times: set[tuple[str, str, str]] = set()
    for path in sorted(root.rglob("*.fits")):
        if " (1)" in path.name:
            continue
        relative = path.relative_to(root)
        if len(relative.parts) < 3:
            raise ValueError(f"Expected date/channel/file layout, got {relative}")
        date, channel = relative.parts[0], relative.parts[1]
        frame = read_frame(path, normalize_exposure=False)
        key = (date, channel, frame.time.isoformat())
        if key in seen_times:
            raise ValueError(f"Duplicate observation time: {key}")
        seen_times.add(key)
        check = check_wcs(frame.header)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "source_url": source_url(date, channel, path.name),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "date_obs": frame.time.isoformat().replace("+00:00", "Z"),
                "wavelength_angstrom": int(frame.header.get("WAVELNTH", int(channel))),
                "shape_y": frame.image.shape[0],
                "shape_x": frame.image.shape[1],
                "exptime_s": float(frame.header.get("EXPTIME", 0.0)),
                "wcs_valid": str(check.valid).lower(),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], output: str | Path) -> None:
    if not rows:
        raise ValueError("No manifest rows")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

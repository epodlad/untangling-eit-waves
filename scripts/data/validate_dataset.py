#!/usr/bin/env python3
"""Validate checksums, compressed-image readability, WCS, and core cadence."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from untangling_eit_waves.fits_io import read_frame
from untangling_eit_waves.manifest import sha256
from untangling_eit_waves.wcs_geometry import check_wcs


CORE_WINDOWS = {
    ("2017-04-01", "0171"): ("2017-04-01T21:30:00+00:00", "2017-04-01T22:06:00+00:00"),
    ("2017-04-01", "0193"): ("2017-04-01T21:30:00+00:00", "2017-04-01T22:06:00+00:00"),
    ("2017-04-03", "0171"): ("2017-04-03T14:12:00+00:00", "2017-04-03T14:50:00+00:00"),
    ("2017-04-03", "0193"): ("2017-04-03T14:12:00+00:00", "2017-04-03T14:50:00+00:00"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    errors: list[str] = []
    times: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    for row in rows:
        path = args.data_root / row["relative_path"]
        if not path.exists():
            errors.append(f"missing: {row['relative_path']}")
            continue
        if sha256(path) != row["sha256"]:
            errors.append(f"checksum: {row['relative_path']}")
            continue
        try:
            frame = read_frame(path, normalize_exposure=False)
            check = check_wcs(frame.header)
            if frame.image.shape != (1024, 1024):
                errors.append(f"shape {frame.image.shape}: {row['relative_path']}")
            if not check.valid:
                errors.append(f"WCS: {row['relative_path']}")
            parts = Path(row["relative_path"]).parts
            match = re.search(r"\.(\d{8})_(\d{6})\.fits$", parts[-1])
            if not match:
                errors.append(f"filename time: {row['relative_path']}")
            else:
                nominal = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(
                    tzinfo=timezone.utc
                )
                times[(parts[0], parts[1])].append(nominal)
        except Exception as exc:
            errors.append(f"read {row['relative_path']}: {exc}")

    cadence = {}
    for key, (start_text, end_text) in CORE_WINDOWS.items():
        start, end = datetime.fromisoformat(start_text), datetime.fromisoformat(end_text)
        selected = sorted(t for t in set(times[key]) if start <= t <= end)
        gaps = [(b - a).total_seconds() for a, b in zip(selected, selected[1:])]
        expected = int((end - start).total_seconds() / 120) + 1
        ok = len(selected) == expected and all(gap == 120 for gap in gaps)
        cadence["/".join(key)] = {"frames": len(selected), "expected": expected, "two_minute_grid": ok}
        if not ok:
            errors.append(f"cadence: {key}, frames={len(selected)}, gaps={gaps}")

    report = {
        "manifest_records": len(rows),
        "errors": errors,
        "cadence": cadence,
        "status": "PASS" if not errors else "FAIL",
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recreate the reduced AIA subset from its exact public-source manifest."""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from untangling_eit_waves.manifest import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    for index, row in enumerate(rows, start=1):
        target = args.output / row["relative_path"]
        expected = row["sha256"]
        if target.exists() and not args.overwrite and sha256(target) == expected:
            print(f"[{index:03d}/{len(rows):03d}] verified {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(
            row["source_url"],
            headers={"User-Agent": "untangling-eit-waves/1.0.0"},
        )
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        actual = sha256(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {target}: {actual} != {expected}")
        partial.replace(target)
        print(f"[{index:03d}/{len(rows):03d}] downloaded {target}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a provenance/checksum manifest from a date/channel FITS tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from untangling_eit_waves.manifest import build_rows, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = build_rows(args.data_root)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} unique FITS records to {args.output}")


if __name__ == "__main__":
    main()

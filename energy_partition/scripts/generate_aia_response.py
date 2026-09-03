#!/usr/bin/env python3
"""Generate a compact time-dependent AIA temperature-response product."""

from __future__ import annotations

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from aiatresp import aia_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obstime", default="2017-04-03T14:20:00")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    response = aia_response(obstime=args.obstime)
    channels = np.asarray([int(str(ch).removeprefix("A")) for ch in response.channels])
    logt = np.asarray(response.logt, dtype=np.float64)
    matrix = np.asarray(response.response, dtype=np.float64)
    if matrix.shape != (len(channels), len(logt)):
        raise ValueError(f"Unexpected response shape {matrix.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "observation_time": args.obstime,
        "units": str(response.units),
        "aiatresp_version": version("aiatresp"),
        "aiapy_version": version("aiapy"),
        "channels_angstrom": channels.tolist(),
        "note": "Model/calibration product, not an observed quantity.",
    }
    np.savez_compressed(
        args.output,
        channels_angstrom=channels,
        logt=logt,
        response_channel_temperature=matrix,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["log10_temperature_K", *[f"AIA_{ch}_DN_cm5_s-1_pix-1" for ch in channels]])
        for i, value in enumerate(logt):
            writer.writerow([f"{value:.5f}", *[f"{matrix[j, i]:.9e}" for j in range(len(channels))]])

    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

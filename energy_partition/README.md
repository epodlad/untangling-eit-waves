# Same-event energy-partition pilot

This directory reproduces the controlled AR 12644 analysis for 3 April 2017.
Compact heating, strong dimming, and one propagating-front sector are evaluated
with a common WCS geometry. Direct observables, DEM inferences, and
model-dependent energy conversions remain explicitly separated.

## Reported ranges

| Component | Range | Status |
| --- | ---: | --- |
| Early compact-source thermal excess | 1.5e29--1.0e30 erg | DEM plus area/depth/filling-factor model |
| Strong-dimming mass deficit | 2.1e12--1.1e13 g | DEM plus line-of-sight depth; lower bound |
| Dimming/ejecta mechanical proxy | 4.4e27--2.8e28 erg | Dimming mass paired with a published feature speed |
| Front kinetic component in one fixed sector | 5.9e25--2.0e26 erg | Weak-compression model; not total wave energy |
| Bright-mask front sensitivity | up to 6.9e26 erg | Selection-sensitive upper test |

The full front-sector DEM enhancement is 3.4%, corresponding to
`delta_n/n = 1.7%` if line-of-sight depth is unchanged. Its magnitude is close
to control-region temporal variability. The values are not a closed energy
budget and do not measure `E_front/E_release`.

## Files

```text
scripts/       WCS, masks, DEM, energy, figure, and speed-measurement programs
calibration/   compact AIA temperature-response arrays and metadata
derived/       masks, photometry, DEM products, and energy results
manifests/     public AIA manifest and validation report
```

The reduced AIA FITS files are distributed in the accompanying Zenodo archive.
They are not duplicated in the GitHub repository.

## Reproduction

From the repository root, create an environment and install the full analysis
dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-full.txt
```

With the Zenodo data unpacked at `DATA/raw/aia_synoptic`, rebuild the fixed
masks and photometry:

```bash
python energy_partition/scripts/build_masks_photometry.py \
  --data-root DATA/raw/aia_synoptic/2017-04-03 \
  --output-dir energy_partition/derived
```

Then run:

```bash
python energy_partition/scripts/run_region_dem.py
python energy_partition/scripts/estimate_dem_sensitivity.py
python energy_partition/scripts/compute_energy_partition.py
python energy_partition/scripts/make_partition_figure.py
```

The checked compact temperature response is included. A full response rebuild
with `generate_aia_response.py` additionally requires `aiatresp` and its
official time-dependent calibration inputs. The regional DEM step additionally
requires `demregpy`.

Run the matched-cadence AIA speed test from the directory above the date
folders. The requested output directory is generated locally and is not part
of the archived release:

```bash
python energy_partition/scripts/aia_channel_speed_audit.py \
  --data-root DATA/raw/aia_synoptic \
  --out energy_partition/validation/aia_channel_speeds
```

The free AIA 193 ridge is an observation-operator result under strongly
shortened exposure times, not evidence for a second MHD mode. Exact historical
SWAP Level-1 files and original masks would be required for a definitive
cross-instrument comparison.

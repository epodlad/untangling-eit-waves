# Untangling EIT Waves

Reproducibility materials for the single-author Solar Physics Review

> **Untangling EIT Waves: What a Measured Speed Actually Traces**  
> Olena Podladchikova

The Review separates four levels that must not be conflated: the physical
structure being tracked, the MHD response, the driver history, and the
observation operator. The software and data in this repository preserve the
same distinction between direct observables, derived quantities, and
model-dependent estimates.

## Scientific scope

- The repeated SWAP interval value near 834 km s-1 is audited against the
  published 91.6-Mm radial-ring width and approximately 110-s image interval.
- AIA cadence, WCS geometry, exposure history, and ridge selection are tested
  without treating passband-dependent speeds as automatic MHD-mode labels.
- The 3 April 2017 pilot compares compact-source heating, dimming depletion,
  and one front-sector segment under a common WCS and DEM protocol.
- Mini-front and global-front energy values remain heterogeneous,
  model-dependent anchors. They are not fitted as a universal power law.
- `E_front = C E_release^alpha` is retained as a falsifiable future scaling
  hypothesis, not as a measured relation.

## Repository contents

```text
paper/                 submitted manuscript, bibliography, PDF, and figures
src/                   reusable cadence, energy, FITS, WCS, and manifest tools
tests/                 unit tests for the core calculations
scripts/               one-command figure rebuild
figures/scripts/       generators for quantitative and explanatory figures
figure1/               signed SWAP/AIA display inputs and Figure 1 generator
energy_partition/      same-event DEM, masks, speed audit, and energy analysis
radial_azimuthal/      radial/azimuthal observable table and figure generator
data/                  compact derived tables and AIA source manifest
config/                event windows and geometry status
```

Final vector or hybrid figure assets are supplied in `paper/figures/`. Source
programs are included for the data-derived and quantitative figures. The two
purely explanatory vector schematics are distributed as final figure assets.

The PDF in `paper/` is the author's submitted, pre-peer-review manuscript. It
is not a publisher-formatted Version of Record.

The reduced 128-file SDO/AIA subset is distributed in the accompanying Zenodo
archive at [doi:10.5281/zenodo.22288289](https://doi.org/10.5281/zenodo.22288289)
rather than committed to GitHub. Exact source URLs and SHA-256 values are
recorded in `data/manifests/aia_synoptic_manifest.csv`.

## Quick start

Core package and tests:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Regenerate the code-derived manuscript figures from the compact products in
this repository:

```bash
python -m pip install -r requirements-figures.txt
python scripts/rebuild_figures.py
```

Compile the manuscript with a standard LaTeX/BibTeX installation:

```bash
cd paper
latexmk -pdf -bibtex main.tex
```

## Full same-event analysis

Unpack the Zenodo data archive so that its `raw/aia_synoptic` directory is
available, then follow `energy_partition/README.md`. The main stages are:

1. validate the downloaded FITS files against the manifest;
2. build the fixed source, dimming, and front-sector products;
3. infer the regional DEM with the supplied compact AIA response;
4. calculate the explicitly labeled energy brackets;
5. run the AIA 171/193 ridge and exposure-history audit.

The compact response is a calibration/model product, not an observation.
Line-of-sight depth, filling factor, density, mass, and front kinetic energy
remain model-dependent conversions. The front-sector result is not scaled to
the full Sun without a measured angular extent and compression map.

## Data provenance

The AIA observations are public SDO/AIA Level-1.5 synoptic FITS products. The
SWAP material supports a qualitative display of the event analyzed by O'Hara
et al. (2019); it is not used for new photometry in this Review. Provider data
are not relicensed here. See `LICENSES.md` for the scope of each license.

## Citation and license

Code is released under the BSD 3-Clause License. Author-created derived tables
and original schematics are released under CC BY 4.0. Cite the associated
Review and the archived release at
[doi:10.5281/zenodo.22288289](https://doi.org/10.5281/zenodo.22288289), as
described in `CITATION.cff`.

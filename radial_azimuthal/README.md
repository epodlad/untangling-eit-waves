# Radial--azimuthal EUV-front observables

This compact analysis separates radial propagation speed `v_r` from the
azimuthal drift rate `Omega_pattern` of a selected EUV intensity feature.
Neither quantity is assumed to be a Doppler plasma velocity. Their combination
is an observational morphology and method-comparison diagram, not a fast/slow
MHD-mode classification.

## Contents

- `data/radial_azimuthal_phase_measurements.csv`: published and derived inputs,
  observable definitions, and measurement status;
- `data/event_20170403_pattern_rotation_summary.json`: sensitivity result for
  the reconstructed AIA 171 Angstrom sector;
- `scripts/make_radial_azimuthal_phase_figure.py`: figure generator;
- `scripts/measure_front_pattern_rotation.py`: moving-annular-region analysis
  for the 3 April 2017 test.

## Reproduce the diagram

Run from the repository root:

```bash
python radial_azimuthal/scripts/make_radial_azimuthal_phase_figure.py \
  --rotation-summary radial_azimuthal/data/event_20170403_pattern_rotation_summary.json \
  --csv radial_azimuthal/data/radial_azimuthal_phase_measurements.csv \
  --output paper/figures/fig10_radial_azimuthal_phase.png
```

The program also writes the corresponding PDF. The 1997 measurements are
transcribed from Podladchikova and Berghmans (2005), Attrill et al. (2007), and
Thompson et al. (1999). The 3 April 2017 result is a non-detection under the
tested systematics because the allowed interval crosses zero and the inferred
sign changes across the sensitivity ensemble.


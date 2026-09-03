# Figure 1: signed SWAP and AIA differences

Figure 1 compares three observational realizations of the 1 April 2017 event.
It is a display and observation-operator test, not a new cross-instrument speed
measurement.

The AIA panels contain exposure-normalized signed base differences in DN s-1.
Negative, zero, and positive values denote dimming, little change, and
brightening. A signed asinh normalization is used; an ordinary logarithm is
never applied to signed data. Blue denotes the 171/174 Angstrom channel family
and green the 193/195 Angstrom family only, not a temperature scale.

The SWAP panel uses an author-prepared qualitative display raster for the same
event. It is not a calibrated photometric product and therefore has no
quantitative color bar. Dashed front curves and arrows are qualitative guides,
not fitted or threshold-derived masks.

Run from the repository root:

```bash
python figure1/scripts/make_fig1_real_channels.py
```

The program writes the manuscript PDF to `paper/figures/` and a PNG preview to
`figure1/output/`.


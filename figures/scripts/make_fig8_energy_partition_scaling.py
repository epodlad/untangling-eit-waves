from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, LogFormatterMathtext


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper" / "figures" / "fig8_energy_partition_scaling.pdf"

NAVY = "#17365D"
BLUE = "#188BC1"
ORANGE = "#FF7F45"
PURPLE = "#6F4CAF"
RED = "#D94858"
GREEN = "#18A768"
GRID = "#E7ECF1"
GREY = "#6E8196"
TEXT = "#17212B"
INNER_TEXT_SCALE = 1.10

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 10.0,
        "axes.titlesize": 11.4,
        "axes.titleweight": "bold",
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "axes.linewidth": 0.85,
        "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
    }
)


def style_log_axes(ax, *, both=True):
    """Use quiet publication-style log grids without crowding the data."""
    ax.grid(True, which="major", color=GRID, lw=0.8)
    if both:
        ax.grid(True, which="minor", color=GRID, lw=0.55, alpha=0.72)
    ax.set_axisbelow(True)
    ax.tick_params(which="major", width=0.85, length=4)
    ax.tick_params(which="minor", width=0.55, length=2.3)
    for spine in ax.spines.values():
        spine.set_color("#22282E")
        spine.set_linewidth(0.85)


fig, axes = plt.subplots(2, 2, figsize=(10.9133, 8.1133))
fig.patch.set_facecolor("white")
fig.subplots_adjust(
    left=0.117,
    right=0.975,
    bottom=0.085,
    top=0.895,
    wspace=0.22,
    hspace=0.34,
)
fig.suptitle(
    "ENERGY ACROSS SCALES: WHAT IS KNOWN, ESTIMATED, AND TESTABLE",
    x=0.52,
    y=0.972,
    fontsize=16.0,
    fontweight="bold",
    color=NAVY,
)

# ---------------------------------------------------------------------------
# (a) Unlike energy components
# ---------------------------------------------------------------------------
ax = axes[0, 0]
ax.set_title("(a) Unlike energy components", loc="left", color=NAVY, pad=7)
categories = [
    "compact source\nthermal",
    "14 km s$^{-1}$\nmini-front",
    "45 km s$^{-1}$\nmini-front",
    "global wave\nproxy",
    "strong shock\nestimate",
]
y = np.arange(len(categories))[::-1]

# Ranges are bars; literature anchors are diamonds.
for yy, x0, x1, color in [
    (y[0], 1.0e20, 1.0e24, BLUE),
    (y[1], 3.0e21, 1.2e22, ORANGE),
    (y[2], 3.0e23, 1.2e24, ORANGE),
]:
    ax.plot([x0, x1], [yy, yy], color=color, lw=8.0, solid_capstyle="round")
ax.scatter([1.8e29, 2.5e31], [y[3], y[4]], marker="D", s=58, color=PURPLE, zorder=4)

ax.set_xscale("log")
ax.set_xlim(3.0e19, 8.0e31)
ax.set_ylim(-0.22, 4.22)
ax.set_yticks(y)
ax.set_yticklabels(categories)
ax.set_xticks([1e20, 1e22, 1e24, 1e26, 1e28, 1e30])
ax.xaxis.set_major_formatter(LogFormatterMathtext())
ax.set_xlabel("energy (erg)")
style_log_axes(ax, both=False)

# This key occupies unused upper-right space, clear of all marks.
ax.text(
    0.98,
    0.875,
    "bars: ranges   diamonds: literature anchors",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=7.8 * INNER_TEXT_SCALE,
    color=GREY,
)

# ---------------------------------------------------------------------------
# (b) Hudson heating criterion
# ---------------------------------------------------------------------------
ax = axes[0, 1]
ax.set_title("(b) Hudson heating criterion", loc="left", color=NAVY, pad=7)
energy = np.logspace(20, 32, 600)
pivot = 1.0e26
curves = {
    r"$\gamma<2$: large events": (energy / pivot) ** 0.5,
    r"$\gamma=2$: equal per decade": np.ones_like(energy),
    r"$\gamma>2$: small events": (energy / pivot) ** -0.5,
}
ax.axvspan(1e20, 1e22, color="#DDE3E9", alpha=0.88, zorder=0)
ax.axvline(1e22, color=GREY, ls="--", lw=1.0)
ax.plot(energy, curves[r"$\gamma<2$: large events"], color=RED, lw=2.1)
ax.plot(energy, curves[r"$\gamma=2$: equal per decade"], color=GREEN, lw=2.1)
ax.plot(energy, curves[r"$\gamma>2$: small events"], color=BLUE, lw=2.1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e20, 1e32)
ax.set_ylim(1.2e-1, 7.2)
ax.set_xlabel(r"event energy $E$ (erg)")
ax.set_ylabel(r"energy per log interval  $E^2\,dN/dE$")
ax.text(
    1.0e21,
    6.15,
    "incomplete\nrange",
    color=GREY,
    fontsize=8.0 * INNER_TEXT_SCALE,
    ha="center",
    va="top",
)
style_log_axes(ax)

# A compact manual key stays wholly between the green and blue curves.
for yy, color, label in [
    (0.44, RED, r"$\gamma<2$: large events"),
    (0.37, GREEN, r"$\gamma=2$: equal per decade"),
    (0.30, BLUE, r"$\gamma>2$: small events"),
]:
    ax.plot(
        [0.59, 0.645],
        [yy, yy],
        transform=ax.transAxes,
        color=color,
        lw=2.4,
        solid_capstyle="butt",
        zorder=5,
    )
    ax.text(
        0.66,
        yy,
        label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.5 * INNER_TEXT_SCALE,
        color=TEXT,
        zorder=5,
    )

# ---------------------------------------------------------------------------
# (c) Full-Sun-average relevance
# ---------------------------------------------------------------------------
ax = axes[1, 0]
ax.set_title("(c) Full-Sun-average relevance", loc="left", color=NAVY, pad=7)
categories_c = ["compact\nbrightenings", "mini-front\nkinetic", "generous mini\nproxy"]
yc = np.array([2, 1, 0])
for yy, x0, x1, color in [
    (2, 0.8, 1.25, BLUE),
    (1, 8e-6, 1e-4, ORANGE),
    (0, 1e-2, 1e-1, RED),
]:
    ax.plot([x0, x1], [yy, yy], color=color, lw=9.0, solid_capstyle="round")

ax.set_xscale("log")
ax.set_xlim(3e-7, 2e2)
ax.set_ylim(-0.22, 2.32)
ax.set_yticks(yc)
ax.set_yticklabels(categories_c)
ax.set_xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
ax.xaxis.set_major_formatter(LogFormatterMathtext())
ax.set_xlabel("quiet-Sun heating requirement (%)")
ax.axvline(1e2, color=GREY, ls="--", lw=1.0)
ax.text(
    8.2e1,
    1.45,
    "100% requirement",
    color=GREY,
    fontsize=7.8 * INNER_TEXT_SCALE,
    rotation=90,
    ha="right",
    va="center",
)
style_log_axes(ax)

# The note is moved to empty upper-left space and given a quiet white backing.
ax.text(
    0.02,
    0.955,
    "Front values are upper limits to heating;\ncomplete coronal dissipation is assumed.",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=7.5 * INNER_TEXT_SCALE,
    color=GREY,
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.5},
)

# ---------------------------------------------------------------------------
# (d) Universal-scaling hypothesis
# ---------------------------------------------------------------------------
ax = axes[1, 1]
ax.set_title("(d) Universal-scaling hypothesis", loc="left", color=NAVY, pad=7)
fraction_down = 1e-2 * (energy / pivot) ** -0.3
fraction_flat = np.full_like(energy, 1e-2)
fraction_up = 1e-2 * (energy / pivot) ** 0.3
ax.plot(energy, fraction_down, color=BLUE, lw=2.1)
ax.plot(energy, fraction_flat, color=GREEN, lw=2.1)
ax.plot(energy, fraction_up, color=RED, lw=2.1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e20, 1e32)
ax.set_ylim(1e-4, 1e0)
ax.set_xlabel(r"released energy $E_{\mathrm{release}}$ (erg)")
ax.set_ylabel(r"transported fraction  $E_{\mathrm{front}}/E_{\mathrm{release}}$")
style_log_axes(ax)

# Keep the alpha key left of centre and unboxed so no white patch hides a curve.
for yy, color, label in [
    (0.86, BLUE, r"$\alpha<1$: fraction falls"),
    (0.80, GREEN, r"$\alpha=1$: constant fraction"),
    (0.74, RED, r"$\alpha>1$: fraction rises"),
]:
    ax.plot(
        [0.27, 0.325],
        [yy, yy],
        transform=ax.transAxes,
        color=color,
        lw=2.4,
        solid_capstyle="butt",
        zorder=5,
    )
    ax.text(
        0.34,
        yy,
        label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.8 * INNER_TEXT_SCALE,
        color=TEXT,
        zorder=5,
    )
ax.text(
    0.97,
    0.975,
    "hypotheses, not fits",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8.5 * INNER_TEXT_SCALE,
    fontweight="bold",
    color=RED,
)

# Only the requested tool names remain in the PDF information dictionary.
metadata = {
    "Title": None,
    "Author": None,
    "Subject": None,
    "Keywords": None,
    "Creator": "Python",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
    "Trapped": None,
}
fig.savefig(OUTPUT, format="pdf", metadata=metadata, facecolor="white")
plt.close(fig)

print(OUTPUT)

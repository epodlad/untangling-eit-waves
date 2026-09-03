from math import exp
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper" / "figures" / "fig5_nemo_front_dimming.pdf"


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 3,
    }
)


# Reconstruct the original schematic curve exactly from its two Gaussian terms.
n_points = 600
x = [i / (n_points - 1) for i in range(n_points)]
y = [
    -0.55 * exp(-0.5 * ((value - 0.47) / 0.12) ** 2)
    + 0.60 * exp(-0.5 * ((value - 0.73) / 0.055) ** 2)
    for value in x
]

blue = "#1f77b4"
orange = "#ff7f0e"

fig, ax = plt.subplots(figsize=(7.8, 3.5))

ax.fill_between(
    x,
    y,
    0,
    where=[value <= 0 for value in y],
    facecolor=mpl.colors.to_rgba(blue, 0.18),
    edgecolor="black",
    hatch="///",
    linewidth=0.8,
    interpolate=True,
)
ax.fill_between(
    x,
    y,
    0,
    where=[value >= 0 for value in y],
    facecolor=mpl.colors.to_rgba(orange, 0.18),
    edgecolor="black",
    hatch="\\\\\\",
    linewidth=0.8,
    interpolate=True,
)

ax.axhline(0, color=blue, linewidth=0.9)
ax.plot(x, y, color=blue, linewidth=2.0)

ax.set_xlim(0, 1)
ax.set_ylim(-0.68, 0.72)
ax.set_xticks([])
ax.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax.tick_params(axis="y", labelsize=10)

ax.set_xlabel("Great-circle distance from eruption centre (schematic)", fontsize=11)
ax.set_ylabel(r"Base-difference signal $I_{\lambda}$", fontsize=11)

# Labels moved into clear interior regions; all other geometry is unchanged.
ax.text(
    0.47,
    -0.32,
    "dimming / depletion",
    ha="center",
    va="center",
    fontsize=10,
)
ax.text(
    0.74,
    0.20,
    "bright front /\ncompression",
    ha="center",
    va="top",
    multialignment="center",
    linespacing=1.05,
    fontsize=10,
)

ax.annotate(
    "adjacent boundary",
    xy=(0.64, 0.0),
    xytext=(0.55, 0.55),
    ha="center",
    fontsize=9,
    arrowprops={"arrowstyle": "->", "linewidth": 1.0},
)

fig.tight_layout()
fig.savefig(
    OUTPUT,
    bbox_inches="tight",
    metadata={"Creator": "Python", "Producer": "Matplotlib", "CreationDate": None},
)
plt.close(fig)

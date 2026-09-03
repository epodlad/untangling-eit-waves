from math import cos, radians, sin
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 3,
    }
)

NAVY = "#16365c"
TEXT = "#263544"
MUTED = "#657789"
GRID = "#e8eef3"
ORANGE = "#f87c48"
BLUE = "#1686c9"
PURPLE = "#6c4aa1"
RED = "#d84a5b"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper" / "figures" / "fig4_speed_measurements.pdf"


def style_bar_axis(ax, xlim, ticks, xlabel):
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.38, 1.38)
    ax.set_xticks(ticks)
    ax.set_xlabel(xlabel, fontsize=10, color=TEXT, labelpad=7)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", length=0, pad=4, labelsize=9.6, colors=TEXT)
    ax.tick_params(axis="x", labelsize=9.3, colors=TEXT, width=0.8, length=4)


fig = plt.figure(figsize=(11.2, 4.08), facecolor="white")
grid = fig.add_gridspec(
    1,
    3,
    width_ratios=(1.0, 1.0, 1.08),
    left=0.075,
    right=0.985,
    bottom=0.18,
    top=0.84,
    wspace=0.34,
)
ax_a = fig.add_subplot(grid[0, 0])
ax_b = fig.add_subplot(grid[0, 1])
ax_c = fig.add_subplot(grid[0, 2])

fig.suptitle(
    "THREE QUANTIFIED OBSERVATIONAL OFFSETS",
    x=0.52,
    y=0.965,
    fontsize=15,
    fontweight="bold",
    color=NAVY,
)

# (a) Sector and fitting-interval comparison.
ax_a.set_title(
    "(a) Sector + fit interval",
    loc="left",
    fontsize=11.7,
    fontweight="bold",
    color=NAVY,
    pad=8,
)
ax_a.barh([1, 0], [345, 296], height=0.52, color=[ORANGE, BLUE], zorder=3)
ax_a.set_yticks([1, 0], ["early /\nfastest sector", "longer\nprofile"])
style_bar_axis(
    ax_a,
    (0, 410),
    [0, 100, 200, 300, 400],
    r"mean catalogue speed (km s$^{-1}$)",
)
ax_a.text(
    337,
    1,
    r"345 km s$^{-1}$",
    ha="right",
    va="center",
    fontsize=10.2,
    fontweight="bold",
    color="white",
)
ax_a.text(
    288,
    0,
    r"296 km s$^{-1}$",
    ha="right",
    va="center",
    fontsize=10.2,
    fontweight="bold",
    color="white",
)
ax_a.text(
    9,
    0.50,
    "21 common events\nNitta vs Muhr",
    ha="left",
    va="center",
    fontsize=8.5,
    linespacing=1.18,
    color=MUTED,
)

# (b) Same-event passband and cadence comparison.
ax_b.set_title(
    "(b) Passband + cadence",
    loc="left",
    fontsize=11.7,
    fontweight="bold",
    color=NAVY,
    pad=8,
)
ax_b.barh([1, 0], [475, 238], height=0.52, color=[BLUE, PURPLE], zorder=3)
ax_b.errorbar(
    [475, 238],
    [1, 0],
    xerr=[47, 20],
    fmt="none",
    ecolor=NAVY,
    elinewidth=1.2,
    capsize=3.2,
    capthick=1.2,
    zorder=4,
)
ax_b.set_yticks([1, 0], ["171 Å\n2.5 min", "304 Å\n10 min"])
style_bar_axis(
    ax_b,
    (0, 590),
    [0, 100, 200, 300, 400, 500],
    r"reported peak speed (km s$^{-1}$)",
)
ax_b.text(
    531,
    1,
    "475±47",
    ha="left",
    va="center",
    fontsize=10.2,
    fontweight="bold",
    color=NAVY,
)
ax_b.text(
    266,
    0,
    "238±20",
    ha="left",
    va="center",
    fontsize=10.2,
    fontweight="bold",
    color=NAVY,
)
ax_b.text(
    12,
    0.50,
    "same 19 May 2007 event\n195 Å evolved similarly to 304 Å",
    ha="left",
    va="center",
    fontsize=8.25,
    linespacing=1.18,
    color=MUTED,
)

# (c) Radial-bin quantisation schematic.  The equation and interpretation are
# kept below the geometry so no line or arrow crosses any text.
ax_c.set_xlim(0, 1)
ax_c.set_ylim(0, 1)
ax_c.axis("off")
ax_c.set_title(
    "(c) Radial-bin quantisation",
    loc="left",
    fontsize=11.7,
    fontweight="bold",
    color=NAVY,
    pad=8,
)

ax_c.text(
    0.50,
    0.82,
    r"one ring:  $\Delta r = 91.6$ Mm",
    ha="center",
    va="center",
    fontsize=10.0,
    fontweight="bold",
    color=RED,
)
ax_c.text(
    0.50,
    0.71,
    r"one interval:  $\Delta t \simeq 110$ s",
    ha="center",
    va="center",
    fontsize=9.8,
    color=RED,
)

arc_center = (0.17, 0.27)
for radius in (0.20, 0.30, 0.40):
    ax_c.add_patch(
        Arc(
            arc_center,
            2 * radius,
            2 * radius,
            theta1=11,
            theta2=73,
            color=BLUE,
            linewidth=1.7,
            transform=ax_c.transAxes,
        )
    )

arrow_angle = radians(31)
arrow_start_radius = 0.205
arrow_end_radius = 0.405
arrow_start = (
    arc_center[0] + arrow_start_radius * cos(arrow_angle),
    arc_center[1] + arrow_start_radius * sin(arrow_angle),
)
arrow_end = (
    arc_center[0] + arrow_end_radius * cos(arrow_angle),
    arc_center[1] + arrow_end_radius * sin(arrow_angle),
)
ax_c.annotate(
    "",
    xy=arrow_end,
    xytext=arrow_start,
    xycoords=ax_c.transAxes,
    textcoords=ax_c.transAxes,
    arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.8, "mutation_scale": 12},
)

ax_c.text(
    0.50,
    0.275,
    r"$\Delta r/\Delta t = \mathbf{833\ km\ s^{-1}}$",
    ha="center",
    va="center",
    fontsize=11.0,
    color=NAVY,
)
ax_c.text(
    0.50,
    0.20,
    r"published repeated value $\simeq 834$ km s$^{-1}$",
    ha="center",
    va="center",
    fontsize=8.6,
    color=MUTED,
)
ax_c.text(
    0.50,
    0.045,
    "A numerical sampling signature;\nnot proof that the physical front was unaccelerated.",
    ha="center",
    va="bottom",
    fontsize=8.5,
    linespacing=1.22,
    color=TEXT,
)

fig.savefig(
    OUTPUT,
    bbox_inches="tight",
    metadata={
        "Creator": "Python",
        "Producer": "Python",
        "CreationDate": None,
        "ModDate": None,
    },
)
plt.close(fig)

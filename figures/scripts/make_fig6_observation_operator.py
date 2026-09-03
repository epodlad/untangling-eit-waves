from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper" / "figures" / "fig6_observation_operator.pdf"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def add_stage(ax, x, y, width, height, edge, fill, heading, lines):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.025",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=fill,
        zorder=1,
    )
    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height - 0.055,
        heading,
        ha="center",
        va="top",
        color=edge,
        fontsize=11.5,
        fontweight="bold",
        linespacing=1.05,
        zorder=3,
    )
    heading_rows = heading.count("\n") + 1
    body_offset = 0.155 if heading_rows == 2 else 0.112
    ax.text(
        x + 0.032,
        y + height - body_offset,
        "\n".join(lines),
        ha="left",
        va="top",
        color="#28384A",
        fontsize=10.1,
        linespacing=1.30,
        zorder=3,
    )


fig, ax = plt.subplots(figsize=(10.5, 4.5))
fig.patch.set_facecolor("white")
ax.set_position([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(
    0.5,
    0.805,
    "WHY PUBLISHED SPEEDS CAN DISAGREE",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color="#193A63",
    zorder=4,
)

y0, height = 0.285, 0.405
stages = [
    {
        "x": 0.018,
        "w": 0.175,
        "edge": "#F87843",
        "fill": "#FAEEE9",
        "heading": "1  SOURCE /\n   DRIVER",
        "lines": ["magnetic release", "CME expansion", "piston or blast history"],
    },
    {
        "x": 0.225,
        "w": 0.205,
        "edge": "#1588C8",
        "fill": "#EAF2F9",
        "heading": "2  PHYSICAL\n   RESPONSE",
        "lines": [
            "fast / slow branch",
            "outer wave / shock",
            "CME-related front",
            "dimming / rarefaction",
        ],
    },
    {
        "x": 0.462,
        "w": 0.220,
        "edge": "#21A15A",
        "fill": "#EAF6EF",
        "heading": "3  OBSERVATION",
        "lines": [
            "171/174 or 193/195 Å",
            "LOS + height + projection",
            "cadence + difference image",
            "sector + time interval",
        ],
    },
    {
        "x": 0.714,
        "w": 0.145,
        "edge": "#6B4AA5",
        "fill": "#F2EDF8",
        "heading": "4  TRACKER",
        "lines": ["leading edge", "crest / ridge", "profile maximum"],
    },
    {
        "x": 0.891,
        "w": 0.091,
        "edge": "#193A63",
        "fill": "#F1F4F7",
        "heading": "5  SPEED",
        "lines": [r"$v_{\mathrm{fit}}$", "", "reported", "number"],
    },
]

for stage in stages:
    add_stage(
        ax,
        stage["x"],
        y0,
        stage["w"],
        height,
        stage["edge"],
        stage["fill"],
        stage["heading"],
        stage["lines"],
    )

arrow_y = y0 + 0.205
for current, following in zip(stages[:-1], stages[1:]):
    start = current["x"] + current["w"] + 0.006
    end = following["x"] - 0.006
    arrow = FancyArrowPatch(
        (start, arrow_y),
        (end, arrow_y),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=2.0,
        color="#718096",
        shrinkA=0,
        shrinkB=0,
        zorder=5,
        clip_on=False,
    )
    ax.add_patch(arrow)

ax.text(
    0.5,
    0.165,
    "MATCH THESE BEFORE INTERPRETING A SPEED DIFFERENCE AS A DIFFERENT MHD MODE",
    ha="center",
    va="center",
    fontsize=11.8,
    fontweight="bold",
    color="#D84B5A",
)
ax.text(
    0.5,
    0.088,
    "A genuine physical difference is the residual that remains after event, structure, "
    "passband, geometry and tracker are matched.",
    ha="center",
    va="center",
    fontsize=10.5,
    color="#28384A",
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(
    OUTPUT,
    format="pdf",
    bbox_inches=Bbox.from_extents(0.0, 0.16, 10.5, 4.02),
    pad_inches=0.03,
    metadata={
        "Creator": "Python",
        "Producer": "Matplotlib",
        "CreationDate": None,
    },
)
plt.close(fig)

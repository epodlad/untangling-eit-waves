from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import AsinhNorm, LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figure1" / "output"
OUT.mkdir(parents=True, exist_ok=True)

SWAP = ROOT / "figure1" / "data" / "swap174_2204_display_base_difference.png"
AIA171 = ROOT / "figure1" / "data" / "aia_171_signed_base_difference_20170401_2204-2130_dn_s.fits"
AIA193 = ROOT / "figure1" / "data" / "aia_193_signed_base_difference_20170401_2204-2130_dn_s.fits"

PDF = ROOT / "paper" / "figures" / "fig1_real_channels.pdf"
PNG = OUT / "fig1_real_channels.png"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def signed_cmap(family: str) -> LinearSegmentedColormap:
    if family == "blue":
        colors = [
            (0.00, "#a9c1d2"), (0.22, "#496a83"), (0.43, "#102b3f"),
            (0.50, "#000000"), (0.57, "#061a42"), (0.77, "#075eaa"),
            (0.91, "#37bde9"), (1.00, "#e8fbff"),
        ]
    else:
        colors = [
            (0.00, "#b8c8b5"), (0.22, "#526b50"), (0.43, "#142d19"),
            (0.50, "#000000"), (0.57, "#052b13"), (0.77, "#087d32"),
            (0.91, "#51d34e"), (1.00, "#efffd5"),
        ]
    return LinearSegmentedColormap.from_list(f"signed_{family}", colors, N=513)


def first_image(path: Path):
    """Read the simple 2-D primary FITS arrays used for this figure."""
    raw = path.read_bytes()
    header = {}
    end_card = None
    for offset in range(0, len(raw), 80):
        card = raw[offset:offset + 80].decode("ascii", "replace")
        key = card[:8].strip()
        if key == "END":
            end_card = offset + 80
            break
        if card[8:10] != "= ":
            continue
        value = card[10:80].split("/", 1)[0].strip()
        if value.startswith("'") and value.endswith("'"):
            parsed = value[1:-1].strip()
        elif value in ("T", "F"):
            parsed = value == "T"
        else:
            try:
                parsed = float(value.replace("D", "E")) if any(
                    ch in value for ch in ".EDed"
                ) else int(value)
            except ValueError:
                parsed = value
        header[key] = parsed
    if end_card is None:
        raise ValueError(f"No FITS END card in {path}")
    data_offset = ((end_card + 2879) // 2880) * 2880
    nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
    bitpix = int(header["BITPIX"])
    dtypes = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}
    data = np.frombuffer(raw, dtype=np.dtype(dtypes[bitpix]), count=nx * ny,
                         offset=data_offset).reshape(ny, nx).astype(float)
    data = data * float(header.get("BSCALE", 1.0)) + float(header.get("BZERO", 0.0))
    return data, header


def image_extent(header):
    nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
    dx, dy = float(header["CDELT1"]), float(header["CDELT2"])
    x0 = (0.5 - float(header["CRPIX1"])) * dx + float(header.get("CRVAL1", 0.0))
    x1 = (nx + 0.5 - float(header["CRPIX1"])) * dx + float(header.get("CRVAL1", 0.0))
    y0 = (0.5 - float(header["CRPIX2"])) * dy + float(header.get("CRVAL2", 0.0))
    y1 = (ny + 0.5 - float(header["CRPIX2"])) * dy + float(header.get("CRVAL2", 0.0))
    return (x0, x1, y0, y1)


stroke = [pe.withStroke(linewidth=2.2, foreground="black", alpha=0.82)]
arrow = dict(arrowstyle="->", color="white", lw=1.55, shrinkA=3, shrinkB=2)


def panel_box(ax, panel, title, title_y=0.975, title_va="top", title_size=8.8,
              title_x=0.978, title_ha="right"):
    ax.text(0.022, 0.975, panel, transform=ax.transAxes, va="top", ha="left",
            color="white", fontsize=11.5, fontweight="bold",
            path_effects=stroke, zorder=20)
    ax.text(title_x, title_y, title, transform=ax.transAxes,
            va=title_va, ha=title_ha,
            color="white", fontsize=title_size, fontweight="bold", linespacing=1.18,
            path_effects=stroke, zorder=20)


def note(ax, text, xy, xytext, coords="data", ha="center", fontsize=10.8):
    ax.annotate(text, xy=xy, xycoords=coords, xytext=xytext, textcoords=coords,
                color="white", fontsize=fontsize, fontweight="bold", ha=ha,
                arrowprops=arrow, path_effects=stroke, zorder=25)


def qualitative_front_guide(ax, vertices):
    """Draw a morphology-following display guide, not a fitted front mask."""
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * (len(vertices) - 1)
    guide = PathPatch(
        MplPath(vertices, codes),
        fill=False,
        edgecolor="white",
        linewidth=1.25,
        linestyle=(0, (4.0, 3.0)),
        alpha=0.95,
        capstyle="round",
        zorder=22,
    )
    guide.set_path_effects([
        pe.Stroke(linewidth=2.7, foreground="black", alpha=0.52),
        pe.Normal(),
    ])
    ax.add_patch(guide)


fig = plt.figure(figsize=(11.8, 4.75), facecolor="white")
gs = fig.add_gridspec(1, 3, width_ratios=(1.16, 1.0, 1.0),
                      left=0.035, right=0.992, bottom=0.230, top=0.880,
                      wspace=0.13)

# Published SWAP base-difference movie frame, with its embedded header/footer removed.
swap = Image.open(SWAP).convert("RGB").crop((113, 75, 826, 705))
swap = ImageEnhance.Contrast(swap).enhance(1.45)
swap = ImageEnhance.Color(swap).enhance(1.08)
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(np.asarray(swap), origin="upper", interpolation="nearest", rasterized=True)
ax0.set_axis_off()
panel_box(ax0, "(a)", "SWAP 174 Å | base difference: 22:04 UT - 21:31 UT",
          title_y=0.055, title_va="bottom", title_size=7.4,
          title_x=0.022, title_ha="left")
note(ax0, "front", (300, 155), (175, 66), fontsize=9.7)
note(ax0, "dimming", (385, 229), (360, 86), fontsize=9.7)
note(ax0, "source region", (420, 264), (560, 125), fontsize=9.7)
# Short asymmetric guide along the visible on-disk front; it is deliberately
# incomplete and is not intended as a fitted or threshold-derived boundary.
qualitative_front_guide(
    ax0,
    [(245, 210), (275, 175), (305, 150), (345, 135)],
)

# The AIA panels retain a common solar-coordinate window but independent channel scaling.
panels = [
    (AIA171, "blue", 321.0, 24.4, "(b)", "AIA 171 Å | 22:04 UT\nbase: 21:30 UT",
     (875, 600),
     [(735, 755), (795, 735), (865, 655), (888, 575),
      (905, 520), (924, 470), (947, 420)]),
    (AIA193, "green", 455.0, 29.0, "(c)", "AIA 193 Å | 22:04 UT\nbase: 21:30 UT",
     (845, 565),
     [(680, 690), (748, 675), (825, 615), (858, 560),
      (885, 515), (910, 455), (925, 400)]),
]

for row, (path, family, clip, width, panel, title, front_xy, front_vertices) in enumerate(panels):
    data, header = first_image(path)
    ax = fig.add_subplot(gs[0, row + 1])
    im = ax.imshow(data, origin="lower", extent=image_extent(header),
                   cmap=signed_cmap(family),
                   norm=AsinhNorm(linear_width=width, vmin=-clip, vmax=clip),
                   interpolation="nearest", rasterized=True)
    ax.set_xlim(40, 1210)
    ax.set_ylim(-120, 920)
    ax.set_facecolor("black")
    ax.tick_params(labelsize=7.6, colors="#263746", length=3)
    ax.set_ylabel("Solar-Y (arcsec)" if row == 0 else "", fontsize=8.3)
    # Approximate optical limb from the WCS metadata.
    theta = np.linspace(0, 2*np.pi, 600)
    rsun = float(header.get("RSUN_OBS", 959.0))
    ax.plot(rsun*np.cos(theta), rsun*np.sin(theta), color="#d7dee4",
            lw=0.75, ls=(0, (2.5, 2.5)), alpha=0.85, zorder=8)
    panel_box(ax, panel, title)

    # Same named observables, placed on the structures visible in each channel.
    note(ax, "off-limb front", front_xy, (110, 630), ha="left", fontsize=8.8)
    note(ax, "dimming", (655, 430), (260, 95), ha="center", fontsize=8.8)
    note(ax, "source region", (735, 285), (1110, 155), ha="right", fontsize=8.8)
    qualitative_front_guide(ax, front_vertices)

    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.050, pad=0.105)
    cb.set_ticks([-clip, 0, clip])
    cb.set_ticklabels([f"−{clip:.0f}", "0", f"+{clip:.0f}"])
    cb.ax.tick_params(labelsize=7.0, length=2)
    cb.set_label(r"signed $\Delta I$ (DN s$^{-1}$): decrease  ←  0  →  increase",
                 fontsize=7.2, labelpad=1.5)

fig.suptitle("1 APRIL 2017: ONE EVENT, THREE OBSERVATIONAL REALIZATIONS",
             x=0.515, y=0.965, color="#17365d", fontsize=13.0, fontweight="bold")
fig.text(0.50, 0.090,
         "Channel colors encode display families, not temperature or geometrical height.",
         ha="center", va="bottom", color="#4c6172", fontsize=8.2)

# Crop only the unused white field below the final explanatory line.
page_crop = mpl.transforms.Bbox.from_bounds(
    0.0, 0.37, fig.get_figwidth(), fig.get_figheight() - 0.37
)
fig.savefig(PDF, dpi=450, facecolor="white", bbox_inches=page_crop)
fig.savefig(PNG, dpi=300, facecolor="white", bbox_inches=page_crop)
plt.close(fig)
print(PDF)
print(PNG)

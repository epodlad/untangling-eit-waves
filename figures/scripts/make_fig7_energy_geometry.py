#!/usr/bin/env python3
"""Create the hybrid publication PDF for manuscript Figure 7."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm, LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[2]
INPUT_FITS = ROOT / "figure1" / "data" / "aia_193_signed_base_difference_20170401_2204-2130_dn_s.fits"
OUTPUT_PDF = ROOT / "paper" / "figures" / "fig7_energy_geometry.pdf"
OUTPUT_PNG = ROOT / "figure1" / "output" / "fig7_energy_geometry_preview.png"

NAVY = "#17365D"
GREY = "#667A91"
GREEN = "#56B986"
RED = "#DE4655"
ANGLE = "#6C4AA0"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def signed_green() -> LinearSegmentedColormap:
    controls = [
        (0.00, "#DDE5D9"),
        (0.16, "#9DB097"),
        (0.30, "#617A5E"),
        (0.40, "#314B34"),
        (0.475, "#0A160D"),
        (0.50, "#000000"),
        (0.525, "#041509"),
        (0.60, "#075329"),
        (0.72, "#09853A"),
        (0.86, "#3DBD45"),
        (0.96, "#A7E66F"),
        (1.00, "#F1FFD8"),
    ]
    return LinearSegmentedColormap.from_list("signed_eit_green", controls, N=513)


def _fits_value(card: str):
    raw = card[10:].split("/", 1)[0].strip()
    if raw.startswith("'"):
        return raw.strip(" '")
    if raw in {"T", "F"}:
        return raw == "T"
    try:
        return float(raw) if any(ch in raw for ch in ".Ee") else int(raw)
    except ValueError:
        return raw


def load_image() -> tuple[np.ndarray, dict[str, object], tuple[float, float, float, float]]:
    """Read this simple uncompressed primary-image FITS without extra dependencies."""
    header: dict[str, object] = {}
    with INPUT_FITS.open("rb") as stream:
        blocks = 0
        found_end = False
        while not found_end:
            block = stream.read(2880)
            if len(block) != 2880:
                raise ValueError("Incomplete FITS header")
            blocks += 1
            for offset in range(0, 2880, 80):
                card = block[offset : offset + 80].decode("ascii")
                key = card[:8].strip()
                if key == "END":
                    found_end = True
                    break
                if card[8:10] == "= " and key:
                    header[key] = _fits_value(card)
        nx = int(header["NAXIS1"])
        ny = int(header["NAXIS2"])
        bitpix = int(header["BITPIX"])
        if bitpix != -32:
            raise ValueError(f"Expected BITPIX=-32, found {bitpix}")
        data = np.fromfile(stream, dtype=">f4", count=nx * ny).reshape(ny, nx).astype(float)
    bscale = float(header.get("BSCALE", 1.0))
    bzero = float(header.get("BZERO", 0.0))
    data = data * bscale + bzero
    x0 = (1.0 - float(header["CRPIX1"])) * float(header["CDELT1"]) + float(header.get("CRVAL1", 0.0))
    x1 = (nx - float(header["CRPIX1"])) * float(header["CDELT1"]) + float(header.get("CRVAL1", 0.0))
    y0 = (1.0 - float(header["CRPIX2"])) * float(header["CDELT2"]) + float(header.get("CRVAL2", 0.0))
    y1 = (ny - float(header["CRPIX2"])) * float(header["CDELT2"]) + float(header.get("CRVAL2", 0.0))
    return data, header, (x0, x1, y0, y1)


def draw_left_panel(ax: mpl.axes.Axes) -> None:
    data, header, extent = load_image()
    ax.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap=signed_green(),
        norm=AsinhNorm(linear_width=34.0, vmin=-430.0, vmax=430.0),
        interpolation="nearest",
        rasterized=True,
        aspect="equal",
    )
    ax.set_xlim(40, 1210)
    ax.set_ylim(-120, 920)
    ax.set_xlabel("Solar-X (arcsec)")
    ax.set_ylabel("Solar-Y (arcsec)")
    ax.set_title("(a) Real AIA 193 Å signed base difference", color=NAVY, fontweight="bold", pad=8)

    # The explanatory sector is centered on the apparent eruption source at
    # (750, 300) arcsec. Its sides and annular edges are great-circle curves
    # projected onto the image plane, so the overlay follows the solar sphere
    # rather than forming a flat Euclidean triangle. It remains deliberately
    # illustrative rather than a fitted mask.
    cx, cy = 750.0, 300.0
    rsun = float(header.get("RSUN_OBS", 960.3))
    source_vector = np.array([cx / rsun, cy / rsun, np.sqrt(1.0 - (cx / rsun) ** 2 - (cy / rsun) ** 2)])

    central_angle = np.deg2rad(126.0)
    central_tangent = np.array([np.cos(central_angle), np.sin(central_angle), 0.0])
    central_tangent[2] = -(
        source_vector[0] * central_tangent[0] + source_vector[1] * central_tangent[1]
    ) / source_vector[2]
    central_tangent /= np.linalg.norm(central_tangent)
    transverse_tangent = np.cross(source_vector, central_tangent)
    transverse_tangent /= np.linalg.norm(transverse_tangent)

    def project_sector(rho: float, alpha: float) -> np.ndarray:
        tangent = (
            np.cos(alpha) * central_tangent
            + np.sin(alpha) * transverse_tangent
        )
        point = np.cos(rho) * source_vector + np.sin(rho) * tangent
        return rsun * point[:2]

    alpha1, alpha2 = np.deg2rad(-30.0), np.deg2rad(30.0)
    rho_inner, rho_outer = 0.23, 0.34

    for alpha in (alpha1, alpha2):
        boundary = np.array(
            [project_sector(rho, alpha) for rho in np.linspace(0.0, rho_outer, 100)]
        )
        ax.plot(
            boundary[:, 0],
            boundary[:, 1],
            color="white",
            lw=1.35,
            solid_capstyle="round",
            zorder=6,
        )

    alphas = np.linspace(alpha1, alpha2, 180)
    inner_edge = np.array([project_sector(rho_inner, alpha) for alpha in alphas])
    outer_edge = np.array([project_sector(rho_outer, alpha) for alpha in alphas])
    ax.plot(
        inner_edge[:, 0],
        inner_edge[:, 1],
        color="white",
        lw=1.35,
        ls=(0, (4.0, 3.0)),
        zorder=6,
    )
    ax.plot(
        outer_edge[:, 0],
        outer_edge[:, 1],
        color="white",
        lw=2.0,
        zorder=6,
    )
    ax.scatter([cx], [cy], s=18, facecolor="white", edgecolor="black", linewidth=0.45, zorder=7)
    ax.annotate(
        "illustrative shell sector",
        xy=(555, 525),
        xytext=(120, 820),
        color="white",
        fontsize=10.0,
        arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.32, edgecolor="none"),
        zorder=7,
    )
    ax.text(
        64,
        -98,
        "Explanatory overlay; not a fitted front mask.",
        color="white",
        fontsize=8.7,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="black", alpha=0.30, edgecolor="none"),
        zorder=7,
    )


def draw_right_panel(ax: mpl.axes.Axes) -> None:
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    ax.set_title("(b) Curved shell sector on the solar surface", color=NAVY, fontweight="bold", pad=8)

    center = (2.65, 5.85)
    sphere_radius = 1.35
    azimuth1, azimuth2 = 8.0, 68.0
    inner_rho, outer_rho = 0.50, 0.82

    # A projected sphere makes clear that the measured sector lies on the
    # solar surface.  The green patch is constructed from great-circle
    # coordinates around an eruption center, so it follows the projected
    # spherical surface rather than appearing as a flat annular wedge.
    sphere = Circle(center, sphere_radius, facecolor="#F6F9FB", edgecolor=GREY, lw=1.0, zorder=0)
    ax.add_patch(sphere)
    ax.add_patch(
        Ellipse(center, 2.0 * sphere_radius, 0.72, fill=False, edgecolor="#B8C6D1", lw=0.75, zorder=1)
    )
    ax.add_patch(
        Ellipse(center, 0.90, 2.0 * sphere_radius, fill=False, edgecolor="#B8C6D1", lw=0.75, zorder=1)
    )
    source_vector = np.array([-0.42, -0.34, np.sqrt(1.0 - 0.42**2 - 0.34**2)])
    east = np.array([1.0, 0.0, 0.0]) - source_vector[0] * source_vector
    east /= np.linalg.norm(east)
    north = np.cross(source_vector, east)
    north /= np.linalg.norm(north)

    def project_surface(rho: float, azimuth: float) -> np.ndarray:
        azimuth_rad = np.deg2rad(azimuth)
        point = (
            np.cos(rho) * source_vector
            + np.sin(rho) * (np.cos(azimuth_rad) * east + np.sin(azimuth_rad) * north)
        )
        return np.array(center) + sphere_radius * point[:2]

    azimuths = np.linspace(azimuth1, azimuth2, 70)
    outer_edge = np.array([project_surface(outer_rho, angle) for angle in azimuths])
    inner_edge = np.array([project_surface(inner_rho, angle) for angle in azimuths[::-1]])
    shell_vertices = np.vstack([outer_edge, inner_edge])
    shell = Polygon(
        shell_vertices,
        closed=True,
        facecolor=GREEN,
        edgecolor="#364D46",
        lw=1.2,
        alpha=0.92,
        zorder=3,
    )
    shell.set_clip_path(sphere)
    ax.add_patch(shell)

    # Great-circle boundaries connect the curved surface sector to the
    # eruption center. Delta phi is therefore shown as an angle on the sphere.
    for azimuth in (azimuth1, azimuth2):
        boundary = np.array(
            [project_surface(rho, azimuth) for rho in np.linspace(0.02, outer_rho, 45)]
        )
        ax.plot(
            boundary[:, 0],
            boundary[:, 1],
            color=GREY,
            lw=0.95,
            ls=(0, (2.0, 2.0)),
            zorder=2,
        )

    source_xy = project_surface(0.0, 0.0)
    ax.plot(source_xy[0], source_xy[1], marker="o", ms=3.2, mfc="white", mec=NAVY, mew=0.9, zorder=5)

    angle_curve = np.array(
        [project_surface(0.25, angle) for angle in np.linspace(azimuth1 + 3.0, azimuth2 - 3.0, 45)]
    )
    ax.plot(angle_curve[:, 0], angle_curve[:, 1], color=ANGLE, lw=1.85, zorder=4)
    ax.text(1.62, 5.82, "angular extent " + r"$\Delta\phi$", color=ANGLE, fontsize=10.1, ha="right", va="center")
    ax.plot([1.67, angle_curve[-1, 0]], [5.82, angle_curve[-1, 1]], color=ANGLE, lw=0.85, zorder=3)

    mid_azimuth = 0.5 * (azimuth1 + azimuth2)
    mid_rho = 0.5 * (inner_rho + outer_rho)
    radius_curve = np.array(
        [project_surface(rho, mid_azimuth) for rho in np.linspace(0.04, mid_rho, 55)]
    )
    ax.plot(radius_curve[:, 0], radius_curve[:, 1], color=NAVY, lw=1.35, zorder=4)
    ax.add_patch(
        FancyArrowPatch(
            radius_curve[-5],
            radius_curve[-1],
            arrowstyle="-|>",
            mutation_scale=13,
            color=NAVY,
            lw=1.35,
            zorder=5,
        )
    )
    ax.text(radius_curve[-1, 0] - 0.04, radius_curve[-1, 1] + 0.17, r"$R$", color=NAVY, fontsize=11.5, ha="center", va="center")
    ax.text(3.02, 6.15, "surface distance", color=GREY, fontsize=8.6, ha="left", va="center")

    thickness_start = project_surface(inner_rho, azimuth1)
    thickness_end = project_surface(outer_rho, azimuth1)
    ax.add_patch(
        FancyArrowPatch(
            thickness_start,
            thickness_end,
            arrowstyle="<->",
            mutation_scale=8.5,
            color=NAVY,
            lw=1.05,
            zorder=5,
        )
    )
    ax.text(3.28, 5.67, r"$\Delta R$", color=NAVY, fontsize=11.2, ha="left", va="center")
    ax.text(3.32, 5.40, "front thickness", color=GREY, fontsize=8.6, ha="left", va="center")
    ax.text(9.10, 6.20, r"$V\simeq R\,\Delta\phi\,\Delta R\,h$", color=NAVY, fontsize=15.0, ha="center")
    ax.text(9.10, 5.68, "local surface-volume approximation", color=GREY, fontsize=9.0, ha="center")
    ax.text(8.65, 4.78, r"half-circumference example: $\Delta\phi=\pi$", color=GREY, fontsize=9.2, ha="center")

    y_rule = 3.80
    ax.plot([1.35, 12.10], [y_rule, y_rule], color="#D9E1E7", lw=1.0)
    headers = ["front speed", r"$R$ (Mm)", r"$\Delta R$ (Mm)", r"$h$ (Mm)", r"$V$ (cm$^3$)"]
    xs = [1.65, 6.35, 8.05, 9.55, 11.82]
    aligns = ["left", "center", "center", "center", "right"]
    for x, h, align in zip(xs, headers, aligns):
        ax.text(x, 3.98, h, color=NAVY, fontsize=9.8, fontweight="bold", ha=align, va="bottom")
    rows = [
        (r"14 km s$^{-1}$", "40", "5", "5", r"$3\times10^{27}$"),
        (r"45 km s$^{-1}$", "80", "10", "10", r"$2.5\times10^{28}$"),
    ]
    for y, row in zip((3.30, 2.80), rows):
        for x, value, align in zip(xs, row, aligns):
            ax.text(x, y, value, color="#233950", fontsize=9.6, ha=align, va="center")

    rect = Rectangle((2.10, 1.75), 4.85, 0.55, facecolor="#83C9A6", edgecolor="#48695A", lw=1.0)
    ax.add_patch(rect)
    ax.plot([7.22, 7.22], [1.70, 2.35], color=NAVY, lw=1.2)
    ax.plot([7.10, 7.22], [1.70, 1.70], color=NAVY, lw=1.2)
    ax.plot([7.10, 7.22], [2.35, 2.35], color=NAVY, lw=1.2)
    ax.plot([7.22, 8.05], [2.02, 2.02], color=NAVY, lw=1.2)
    ax.text(8.20, 2.02, r"$h$ (emitting depth)", color=NAVY, fontsize=11.2, va="center")
    ax.text(
        6.65,
        0.88,
        "Every energy estimate scales linearly with this assumed volume.",
        color=RED,
        fontsize=9.6,
        fontweight="bold",
        ha="center",
    )


def main() -> None:
    fig = plt.figure(figsize=(12.25, 5.15), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.14), left=0.052, right=0.985, top=0.82, bottom=0.105, wspace=0.035)
    left = fig.add_subplot(grid[0, 0])
    right = fig.add_subplot(grid[0, 1])
    draw_left_panel(left)
    draw_right_panel(right)
    fig.suptitle(
        "FROM AN OBSERVED FRONT TO AN EXPLICIT VOLUME",
        x=0.52,
        y=0.96,
        color=NAVY,
        fontsize=18.0,
        fontweight="bold",
    )
    metadata = {
        "Creator": "Python and Matplotlib",
        "Producer": "Matplotlib PDF backend",
        "CreationDate": None,
    }
    fig.savefig(OUTPUT_PDF, dpi=300, metadata=metadata)
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

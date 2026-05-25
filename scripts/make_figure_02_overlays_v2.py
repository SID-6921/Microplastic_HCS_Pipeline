"""
Publication-grade Figure 2: simulated DAPI/PI overlays for the 4 phenotype classes.
Overwrites results/figures/fig_02_cell_overlays.png at 300 dpi.

Uses the same simulation logic as build_all_results.py (seeded RNG, additive
nuclear-stain blobs) but renders a cleaner 4-class \u00d7 3-panel layout:
  Row 1: DAPI (nuclear)
  Row 2: PI (membrane-leak)
  Row 3: DAPI+PI composite (RGB merge)
plus a per-class quantitative annotation strip.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

NAVY  = "#0B2545"
GOLD  = "#E0A106"
TEAL  = "#1F8FFF"
CORAL = "#E94F37"
GREY  = "#7A7A7A"

CLASS_NAMES = ["Viable", "Early Apoptosis", "Late Apoptosis", "Necrosis"]
CLASS_COLORS = [TEAL, GOLD, CORAL, NAVY]

# Per-class simulation parameters: (n_nuclei, nucleus_radius, dapi_intensity, pi_intensity, pi_leak)
SIM_PARAMS = [
    dict(n=18, r=10, dapi=210, pi=15,  leak=0.05),  # Viable
    dict(n=22, r=8,  dapi=180, pi=70,  leak=0.25),  # Early Apop
    dict(n=26, r=6,  dapi=140, pi=160, leak=0.55),  # Late Apop
    dict(n=30, r=5,  dapi=90,  pi=220, leak=0.85),  # Necrosis
]

IMG = 160
SEED = 42

# Custom colormaps
dapi_cmap = LinearSegmentedColormap.from_list("dapi", ["#000814", "#1F4FFF", "#A8D0FF", "#FFFFFF"])
pi_cmap   = LinearSegmentedColormap.from_list("pi",   ["#1A0000", "#B23030", "#FFB347", "#FFFFFF"])

def synth(params, seed):
    rng = np.random.default_rng(seed)
    dapi = rng.integers(8, 30, (IMG, IMG)).astype(np.float32)
    pi   = rng.integers(5, 20, (IMG, IMG)).astype(np.float32)
    yy, xx = np.ogrid[:IMG, :IMG]
    for _ in range(params["n"]):
        cy, cx = rng.integers(params["r"] + 4, IMG - params["r"] - 4, 2)
        rad = params["r"] + rng.integers(-1, 2)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
        dapi[mask] += params["dapi"] + rng.integers(-25, 25)
        # PI leak — concentrates where membrane is damaged
        if rng.random() < params["leak"]:
            pi[mask] += params["pi"] + rng.integers(-25, 25)
        else:
            # weak peripheral PI signal
            ring = ((yy - cy) ** 2 + (xx - cx) ** 2 <= (rad + 2) ** 2) & ~mask
            pi[ring] += params["pi"] * 0.15
    dapi = np.clip(dapi, 0, 255)
    pi = np.clip(pi, 0, 255)
    return dapi, pi

fig, axes = plt.subplots(3, 4, figsize=(13, 9.5), dpi=300,
                          gridspec_kw=dict(hspace=0.18, wspace=0.08))

for col, (cname, ccol, params) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, SIM_PARAMS)):
    dapi, pi = synth(params, SEED + col)

    # Row 0 — DAPI
    ax = axes[0, col]
    ax.imshow(dapi, cmap=dapi_cmap, vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(ccol); sp.set_linewidth(1.8)
    if col == 0:
        ax.set_ylabel("DAPI\n(nuclear)", fontsize=11, color=NAVY, fontweight="bold")
    # Class header
    ax.set_title(cname, fontsize=12, color=ccol, fontweight="bold", pad=8)

    # Row 1 — PI
    ax = axes[1, col]
    ax.imshow(pi, cmap=pi_cmap, vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(ccol); sp.set_linewidth(1.8)
    if col == 0:
        ax.set_ylabel("PI\n(membrane leak)", fontsize=11, color=NAVY, fontweight="bold")

    # Row 2 — RGB composite
    rgb = np.zeros((IMG, IMG, 3), dtype=np.float32)
    rgb[..., 2] = dapi / 255.0          # blue = DAPI
    rgb[..., 0] = pi   / 255.0          # red  = PI
    rgb[..., 1] = 0.20 * (pi / 255.0)   # touch of green for visibility
    rgb = np.clip(rgb, 0, 1)
    ax = axes[2, col]
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(ccol); sp.set_linewidth(1.8)
    if col == 0:
        ax.set_ylabel("Composite\n(DAPI + PI)", fontsize=11, color=NAVY, fontweight="bold")

    # Per-class summary bar below
    ax.text(0.5, -0.16,
            f"n\u2248{params['n']}  r\u2248{params['r']}px  PI-leak {int(params['leak']*100)}%",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color=GREY, style="italic")

fig.suptitle("Figure 2 \u2014 Simulated DAPI / PI Fluorescence Phenotypes",
             fontsize=14, color=NAVY, fontweight="bold", y=0.99)
fig.text(0.5, 0.96,
         "Synthetic micrographs generated from the same parametric model used "
         "to derive the 17-descriptor feature vectors (seed-locked, n=160\u00d7160 px).",
         ha="center", va="top", fontsize=9.5, color=GREY, style="italic")

# Scale bar on bottom-right composite
sb_len = 24  # pixels
ax = axes[2, -1]
ax.add_patch(mpatches.Rectangle((IMG - sb_len - 8, IMG - 12), sb_len, 4,
                                  facecolor="white", edgecolor="white"))
ax.text(IMG - sb_len / 2 - 8, IMG - 18, "\u224820 \u00b5m",
        ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")

out = FIGS / "fig_02_cell_overlays.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[OK] wrote {out}")

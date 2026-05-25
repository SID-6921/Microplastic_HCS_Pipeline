"""
Publication-grade Figure 1: end-to-end HCS pipeline schematic.
Overwrites results/figures/fig_01_pipeline_workflow.png at 300 dpi.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

NAVY  = "#0B2545"
GOLD  = "#E0A106"
TEAL  = "#1F8FFF"
CORAL = "#E94F37"
GREY  = "#7A7A7A"
LIGHT = "#F2F4F7"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": NAVY,
    "axes.linewidth": 1.0,
})

stages = [
    ("Image\nAcquisition",     "DAPI + PI\nfluorescence",                NAVY),
    ("Preprocessing",          "Resize \u00b7 denoise\nz-score normalise", TEAL),
    ("Cell Detection",         "Adaptive\nthresholding",                 TEAL),
    ("Feature Extraction",     "17 morphological\ndescriptors",           GOLD),
    ("Classifier Stack",       "LR \u00b7 RF \u00b7 CNN\nR18s \u00b7 R18p",  CORAL),
    ("Statistical Validation", "Calibration \u00b7 BH-FDR\npermutation tests", NAVY),
]

fig, ax = plt.subplots(figsize=(15, 4.2), dpi=300)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

n = len(stages)
box_w, box_h = 0.125, 0.55
gap = (1.0 - n * box_w) / (n + 1)
xs = [gap + i * (box_w + gap) for i in range(n)]
y0 = 0.22

for i, ((title, sub, col), x) in enumerate(zip(stages, xs)):
    # Drop shadow
    ax.add_patch(mpatches.FancyBboxPatch(
        (x + 0.004, y0 - 0.012), box_w, box_h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor="#00000018", edgecolor="none", zorder=1))
    # Main box
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y0), box_w, box_h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=col, edgecolor="white", linewidth=1.6, zorder=2))
    # Stage number badge
    ax.add_patch(mpatches.Circle(
        (x + 0.018, y0 + box_h - 0.05), 0.022,
        facecolor="white", edgecolor=col, linewidth=1.4, zorder=3))
    ax.text(x + 0.018, y0 + box_h - 0.05, f"{i+1}",
            ha="center", va="center", fontsize=8.5,
            color=col, fontweight="bold", zorder=4)
    # Title
    ax.text(x + box_w / 2, y0 + box_h - 0.14, title,
            ha="center", va="center", fontsize=10.5,
            color="white", fontweight="bold", zorder=4)
    # Subtitle
    ax.text(x + box_w / 2, y0 + 0.14, sub,
            ha="center", va="center", fontsize=8.5,
            color="white", zorder=4)
    # Arrow to next
    if i < n - 1:
        arrow = FancyArrowPatch(
            (x + box_w + 0.002, y0 + box_h / 2),
            (xs[i + 1] - 0.002, y0 + box_h / 2),
            arrowstyle="-|>", mutation_scale=14,
            color=NAVY, linewidth=1.8, zorder=2)
        ax.add_patch(arrow)

# Sub-track: data products beneath each stage
products = [
    "raw .tif", "norm. arrays", "cell masks",
    "feature.csv", "predictions",
    "Tables 1\u201311"
]
for x, prod in zip(xs, products):
    ax.text(x + box_w / 2, y0 - 0.06, prod,
            ha="center", va="center", fontsize=7.8,
            color=GREY, style="italic")

# Top banner
ax.text(0.5, 0.94, "End-to-End Simulation-Conditioned HCS Benchmarking Pipeline",
        ha="center", va="center", fontsize=13, color=NAVY, fontweight="bold")
ax.text(0.5, 0.88, "Simulation \u2192 Features \u2192 Classification \u2192 Calibrated, FDR-controlled inference",
        ha="center", va="center", fontsize=9.5, color=GREY, style="italic")

# Bottom legend strip
ax.add_patch(mpatches.Rectangle((0, 0.02), 1, 0.07, facecolor=LIGHT, edgecolor="none"))
ax.text(0.5, 0.055,
        "Open-source implementation: github.com/SID-6921/Microplastic_HCS_Pipeline   "
        "\u00b7   Reproducible seed-locked workflow   \u00b7   Python 3.12 / scikit-learn 1.8 / PyTorch 2.x",
        ha="center", va="center", fontsize=8.2, color=NAVY)

out = FIGS / "fig_01_pipeline_workflow.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[OK] wrote {out}")

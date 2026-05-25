"""
Generates Figure 10: conceptual schematic of the degenerate vs non-degenerate
feature regime. Three panels:
  (A) Dominant-feature classification: histogram of one feature cleanly
      separating classes -> high AUC.
  (B) AUC collapse after removal of the dominant feature (LR / RF bar pair).
  (C) Distributed morphology (non-degenerate): no single feature dominates,
      AUC stays moderate but is more biologically faithful.

Produces: results/figures/fig_10_degenerate_regime_schematic.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 220,
})

NAVY   = "#0B2545"
GOLD   = "#E0A106"
TEAL   = "#1F8FFF"
CORAL  = "#E94F37"
GREY   = "#7A7A7A"

rng = np.random.default_rng(42)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

# ── Panel A: Dominant-feature class separation ────────────────────────────
axA = axes[0]
v = rng.normal(-1.4, 0.45, 600)
ea = rng.normal(-0.5, 0.45, 600)
la = rng.normal(0.5, 0.45, 600)
nc = rng.normal(1.6, 0.45, 600)
for arr, lbl, c in [(v, "Viable", TEAL), (ea, "Early Apop.", GOLD),
                    (la, "Late Apop.", CORAL), (nc, "Necrosis", NAVY)]:
    axA.hist(arr, bins=28, alpha=0.55, color=c, label=lbl, density=True)
axA.set_title("A. Degenerate regime\n(one feature dominates)")
axA.set_xlabel("membrane_permeability_proxy  (z-score)")
axA.set_ylabel("Density")
axA.legend(fontsize=8, frameon=False, loc="upper right")
axA.text(0.02, 0.95, "Macro-AUC = 0.98",
         transform=axA.transAxes, fontsize=10, fontweight="bold",
         color=NAVY, va="top")

# ── Panel B: AUC collapse after feature removal ───────────────────────────
axB = axes[1]
models = ["LR", "RF"]
auc_all  = [0.981, 0.972]
auc_drop = [0.861, 0.872]
x = np.arange(len(models))
w = 0.35
b1 = axB.bar(x - w/2, auc_all,  w, color=TEAL,  label="All 17 features")
b2 = axB.bar(x + w/2, auc_drop, w, color=CORAL,
             label="Dominant feature removed")
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        axB.text(rect.get_x() + rect.get_width()/2, h + 0.008,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=9)
# Delta arrows
for xi, a, b in zip(x, auc_all, auc_drop):
    axB.annotate("", xy=(xi + w/2, b + 0.005), xytext=(xi - w/2, a - 0.005),
                 arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    axB.text(xi, (a + b) / 2 + 0.01, f"\u0394 = {a-b:+.3f}",
             ha="center", color=GREY, fontsize=9, fontweight="bold")
axB.set_xticks(x); axB.set_xticklabels(models)
axB.set_ylabel("Macro-AUC (held-out)")
axB.set_ylim(0.80, 1.02)
axB.set_title("B. AUC collapse after\ndominant-feature ablation")
axB.legend(fontsize=8, frameon=False, loc="lower left")

# ── Panel C: Distributed morphology (non-degenerate) ──────────────────────
axC = axes[2]
# Each class is moderately separated across multiple weak features
n_per = 220
classes = ["Viable", "Early Apop.", "Late Apop.", "Necrosis"]
colors = [TEAL, GOLD, CORAL, NAVY]
centers = np.array([[-1.0, -0.4], [-0.3, 0.6], [0.6, -0.7], [1.1, 0.9]])
for (cx, cy), lbl, c in zip(centers, classes, colors):
    pts = rng.normal(0.0, 0.55, (n_per, 2)) + np.array([cx, cy])
    axC.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.55, color=c, label=lbl,
                edgecolors="none")
axC.set_title("C. Non-degenerate regime\n(distributed morphology)")
axC.set_xlabel("Feature axis 1 (weak)")
axC.set_ylabel("Feature axis 2 (weak)")
axC.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
axC.text(0.02, 0.05, "Macro-AUC \u2248 0.86  (biological faithful)",
         transform=axC.transAxes, fontsize=10, fontweight="bold",
         color=NAVY, va="bottom",
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))

fig.suptitle(
    "Figure 10. Degenerate vs non-degenerate feature regimes in HCS benchmarks",
    fontsize=12, fontweight="bold", y=1.04)

out_png = FIGS / "fig_10_degenerate_regime_schematic.png"
fig.savefig(out_png, dpi=220, bbox_inches="tight")
print(f"[OK] wrote {out_png}")

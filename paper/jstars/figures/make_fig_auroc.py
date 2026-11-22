"""Fig. 2: building-level AUROC per held-out development event, frozen encoders and controls, with 95% tile-cluster
bootstrap intervals.

Values are the recorded p02b (20 Nov 2026), p03 and p04 (22 Nov 2026) results, identical to
paper/expected_values.yaml (checked by notebooks/reproduce/r01_reproduce_paper.ipynb).
Run: python make_fig_auroc.py  ->  fig_auroc.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVENTS = ["Beirut\n(explosion)", "Congo\n(volcano)", "Hawaii\n(wildfire)", "La Palma\n(volcano)", "Libya\n(flood)"]
SETTINGS = [  # label, AUROC, 95% tile-cluster bootstrap (lower, upper), colour, marker
    ("DINOv3 ViT-B, outline", [0.646, 0.849, 0.504, 0.604, 0.498],
     [(0.381, 0.873), (0.792, 0.897), (0.403, 0.607), (0.548, 0.658), (0.387, 0.605)], "#1f4e9c", "o"),
    ("DINOv3 ViT-B, 10 m ring", [0.654, 0.853, 0.549, 0.533, 0.503],
     [(0.383, 0.892), (0.791, 0.901), (0.434, 0.654), (0.472, 0.597), (0.382, 0.619)], "#7aa6e0", "s"),
    ("ImageNet ViT-B, outline", [0.465, 0.781, 0.716, 0.755, 0.600],
     [(0.262, 0.684), (0.650, 0.881), (0.637, 0.770), (0.704, 0.804), (0.498, 0.703)], "#c2571a", "D"),
    ("DINOv3 ViT-L sat., outline", [0.742, 0.827, 0.755, 0.856, 0.646],
     [(0.650, 0.838), (0.681, 0.880), (0.704, 0.794), (0.828, 0.881), (0.549, 0.734)], "#2a8a4a", "v"),
    ("Image-free control", [0.661, 0.269, 0.405, 0.544, 0.272],
     [(0.584, 0.761), (0.129, 0.637), (0.279, 0.526), (0.498, 0.588), (0.168, 0.384)], "#555555", "^"),
]

plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.45, 2.55))
ax.axhline(0.5, color="#999999", lw=0.8, ls="--", zorder=0)
offsets = [-0.28, -0.14, 0.0, 0.14, 0.28]
for (label, vals, ci, colour, marker), dx in zip(SETTINGS, offsets):
    xs = [i + dx for i in range(len(EVENTS))]
    err = [[v - lo for v, (lo, _) in zip(vals, ci)], [hi - v for v, (_, hi) in zip(vals, ci)]]
    ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=colour, elinewidth=0.7, capsize=1.5, alpha=0.8, zorder=2)
    ax.scatter(xs, vals, s=20, color=colour, marker=marker, edgecolor="white", linewidth=0.5, label=label, zorder=3)
ax.set_xticks(range(len(EVENTS)))
ax.set_xticklabels(EVENTS, fontsize=7)
ax.set_ylabel("Building-level AUROC")
ax.set_ylim(0.1, 0.95)
ax.set_xlim(-0.5, 4.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6.5,
          handletextpad=0.3, columnspacing=0.8)
fig.tight_layout(pad=0.3)
fig.savefig(Path(__file__).with_name("fig_auroc.pdf"))
print("wrote fig_auroc.pdf")

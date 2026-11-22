"""Fig. 2: building-level AUROC per held-out development event, frozen encoders and controls.

Values are the recorded p02b (20 Nov 2026) and p03 (22 Nov 2026) results, identical to
paper/expected_values.yaml (checked by notebooks/reproduce/r01_reproduce_paper.ipynb).
Run: python make_fig_auroc.py  ->  fig_auroc.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVENTS = ["Beirut\n(explosion)", "Congo\n(volcano)", "Hawaii\n(wildfire)", "La Palma\n(volcano)", "Libya\n(flood)"]
SETTINGS = [  # label, values, colour, marker
    ("DINOv3, outline", [0.646, 0.849, 0.504, 0.604, 0.498], "#1f4e9c", "o"),
    ("DINOv3, 10 m ring (surroundings)", [0.654, 0.853, 0.549, 0.533, 0.503], "#7aa6e0", "s"),
    ("ImageNet ViT-B, outline", [0.465, 0.781, 0.716, 0.755, 0.600], "#c2571a", "D"),
    ("Image-free control", [0.661, 0.269, 0.405, 0.544, 0.272], "#555555", "^"),
]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.45, 2.55))
ax.axhline(0.5, color="#999999", lw=0.8, ls="--", zorder=0)
ax.text(4.45, 0.505, "chance", ha="right", va="bottom", fontsize=7, color="#666666")
offsets = [-0.24, -0.08, 0.08, 0.24]
for (label, vals, colour, marker), dx in zip(SETTINGS, offsets):
    ax.scatter([i + dx for i in range(len(EVENTS))], vals, s=22, color=colour, marker=marker,
               edgecolor="white", linewidth=0.5, label=label, zorder=3)
ax.set_xticks(range(len(EVENTS)))
ax.set_xticklabels(EVENTS, fontsize=7)
ax.set_ylabel("Building-level AUROC")
ax.set_ylim(0.2, 0.92)
ax.set_xlim(-0.5, 4.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6.5,
          handletextpad=0.3, columnspacing=0.8)
fig.tight_layout(pad=0.3)
fig.savefig(Path(__file__).with_name("fig_auroc.pdf"))
print("wrote fig_auroc.pdf")

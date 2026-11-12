"""Figures from numbers.json. Palette: validated categorical slots (light mode); identity also carried by marker shape."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import CORPORA

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures"); os.makedirs(F, exist_ok=True)
N = json.load(open(os.path.join(R, "numbers.json")))
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.linewidth": 0.6, "xtick.color": "#333333", "ytick.color": "#333333"})
SER = [("presentation", "Presentation flags", "#2a78d6", "o"), ("structure", "Structure only", "#eb6834", "s"),
       ("char_host", "Char n-gram, host view", "#1baf7a", "D"), ("char_canon", "Char n-gram, canonical", "#eda100", "^"),
       ("char_raw", "Char n-gram, raw", "#e87ba4", "v")]

# Figure 1: dot plot of within-corpus MCC (random split) per detector
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ys = np.arange(len(CORPORA))[::-1]
for k, (m, lab, col, mk) in enumerate(SER):
    v = np.array([N["within"][c]["random"][m]["mcc"] for c in CORPORA])
    off = (k - 2) * 0.12
    ax.errorbar(v[:, 0], ys + off, xerr=v[:, 1], fmt=mk, ms=6, color=col, mec="white", mew=0.8, elinewidth=1, capsize=0, label=lab, zorder=3)
ax.set_yticks(ys); ax.set_yticklabels(CORPORA)
ax.set_xlim(-0.02, 1.04); ax.set_xlabel("MCC on held-out 20% (random split, mean and 95% CI over 10 seeds)")
ax.grid(axis="x", color="#dddddd", lw=0.5, zorder=0)
for y in ys[:-1]:
    ax.axhline(y - 0.5, color="#eeeeee", lw=0.5, zorder=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig_within.pdf")); fig.savefig(os.path.join(F, "fig_within.png"), dpi=200)

# Figure 2: cross-corpus AUC heatmaps for raw, canonical and host-view character models
X = N.get("cross", {}).get("pair", {}).get("url", {})
if X and all(tr in X.get("char_host", {}) for tr in CORPORA):
    mods = [("char_raw", "Raw string"), ("char_canon", "Canonical"), ("char_host", "Host view")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
    for ax, (m, title) in zip(axs, mods):
        M = np.full((len(CORPORA), len(CORPORA)), np.nan)
        for i, tr in enumerate(CORPORA):
            for j, te in enumerate(CORPORA):
                if tr == te:
                    M[i, j] = N["within"][tr]["random"][m]["auc"][0]
                else:
                    M[i, j] = X[m][tr][te]["auc"][0] if te in X[m][tr] else np.nan
        im = ax.imshow(M, vmin=0.3, vmax=1.0, cmap=cmap)
        for i in range(len(CORPORA)):
            for j in range(len(CORPORA)):
                if np.isnan(M[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.5, color="#555555"); continue
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5.4,
                        color="white" if M[i, j] > 0.75 else "#222222", fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(CORPORA))); ax.set_xticklabels(CORPORA, rotation=60, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_yticks(range(len(CORPORA))); axs[0].set_yticklabels(CORPORA, fontsize=7); axs[0].set_ylabel("Training corpus")
    fig.text(0.5, 0.005, "Test corpus", ha="center")
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02); cb.set_label("ROC-AUC", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(F, "fig_cross.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(F, "fig_cross.png"), dpi=200, bbox_inches="tight")
print("figures written")

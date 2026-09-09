"""One-off plot: KHOP 200-epoch model (MLP/RF on frozen embeddings) vs ECFP
Morgan fingerprints, across LIPO/Tox21/BACE. Reads eval_many_models_mlp_rf_results.json.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

MODEL = "KHOP_B1024_H1024_200EP_GAT_9M_ZINC"

# Categorical slots 1 (blue) and 2 (orange) from the validated palette --
# fixed order, colorblind-safe adjacent pair (CVD dE 9.1 light).
COLOR_EMBEDDINGS = "#2a78d6"
COLOR_ECFP = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

with open("eval_many_models_mlp_rf_results.json") as f:
    results = json.load(f)

summary = results[MODEL]["summary"]


def get(key):
    entry = summary[key]
    return entry["mean"], entry["std"]


panels = [
    (
        "LIPO",
        "Test R² (higher is better)",
        [
            ("MLP", get("lipo_embeddings_mlp_r2"), get("lipo_fingerprints_mlp_r2")),
            ("RF", get("lipo_embeddings_rf_r2"), get("lipo_fingerprints_rf_r2")),
        ],
        (0, 0.85),
    ),
    (
        "Tox21",
        "Mean test ROC-AUC over 12 tasks",
        [
            ("MLP", get("tox21_embeddings_mlp_roc_auc"), get("tox21_fingerprints_mlp_roc_auc")),
            ("RF", get("tox21_embeddings_rf_roc_auc"), get("tox21_fingerprints_rf_roc_auc")),
        ],
        (0.6, 0.9),
    ),
    (
        "BACE",
        "Test ROC-AUC (scaffold split, single run)",
        [
            ("MLP", get("bace_embeddings_mlp_roc_auc"), get("bace_fingerprints_mlp_roc_auc")),
            ("RF", get("bace_embeddings_rf_roc_auc"), get("bace_fingerprints_rf_roc_auc")),
        ],
        (0.6, 0.95),
    ),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 5), facecolor=SURFACE)
fig.suptitle(
    "KHOP GAT encoder (200 epochs, ZINC-9M) vs ECFP fingerprints",
    fontsize=14, fontweight="bold", color=INK_PRIMARY, x=0.02, ha="left"
)
fig.text(
    0.02, 0.93,
    "MLP head and Random Forest trained on frozen SSL embeddings vs. Morgan/ECFP fingerprints — final checkpoint (epoch 200)",
    fontsize=9.5, color=INK_SECONDARY, ha="left"
)

bar_width = 0.32

for ax, (dataset_name, ylabel, groups, ylim) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    x = np.arange(len(groups))

    emb_means = [g[1][0] for g in groups]
    emb_stds = [g[1][1] for g in groups]
    ecfp_means = [g[2][0] for g in groups]
    ecfp_stds = [g[2][1] for g in groups]
    labels = [g[0] for g in groups]

    bars_emb = ax.bar(
        x - bar_width / 2, emb_means, bar_width, yerr=emb_stds, capsize=4,
        color=COLOR_EMBEDDINGS, label="SSL embeddings", zorder=3,
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.2, "capthick": 1.2},
    )
    bars_ecfp = ax.bar(
        x + bar_width / 2, ecfp_means, bar_width, yerr=ecfp_stds, capsize=4,
        color=COLOR_ECFP, label="ECFP fingerprints", zorder=3,
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.2, "capthick": 1.2},
    )

    for bars, means, stds in ((bars_emb, emb_means, emb_stds), (bars_ecfp, ecfp_means, ecfp_stds)):
        for rect, mean, std in zip(bars, means, stds):
            ax.text(
                rect.get_x() + rect.get_width() / 2, mean + std + ylim[1] * 0.015,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK_PRIMARY,
            )

    ax.set_title(dataset_name, fontsize=12, fontweight="bold", color=INK_PRIMARY, pad=10)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=INK_PRIMARY)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", colors=INK_MUTED, labelsize=9)

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, legend_labels, loc="upper right", bbox_to_anchor=(0.99, 0.97),
    frameon=False, fontsize=10, labelcolor=INK_PRIMARY,
)

fig.tight_layout(rect=[0, 0, 1, 0.88])
output_path = f"models/{MODEL}/mlp_rf_vs_ecfp.png"
fig.savefig(output_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
print(f"Saved {output_path}")

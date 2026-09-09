"""Epoch-sweep trend plot for KHOP_B1024_H1024_200EP_GAT_9M_ZINC (10, 50, 100,
150, 190, 200 checkpoints) -- shows downstream MLP/RF performance as a
function of training length, not a model-vs-model comparison, so a line
chart (epoch on x, metric on y) communicates the trend far more clearly than
generate_comparison_plot's grouped-bar-chart form. Same palette/ink tokens
as eval_many_models_mlp_rf.py for visual consistency with the rest of the
pipeline's saved plots.

Two rows: top row is the original primary metric per dataset (R² / ROC-AUC),
bottom row is the metric added later per dataset (RMSE / MCC) -- the same
metrics used to answer whether the "epoch-200 jump" shows up in classification
QUALITY, not just ranking (ROC-AUC) or R².

Shaded bands show +/- 1 std across the 5 random split seeds for LIPO/Tox21.
BACE uses a single fixed scaffold split (no seed variation) for THESE
already-computed historical results -- no variance estimate at all, flagged
explicitly rather than left to look misleadingly precise. (BACE's default
splitter changed to "random" going forward -- see evaluation/knn_bace.py --
but these six checkpoints were evaluated before that change, under scaffold,
and re-evaluating them isn't free, so this plot stays consistent with the
numbers already reported rather than silently mixing splitters.)

RMSE and BACE's MCC aren't in eval_mlp_rf_results.json's "summary" dict (RMSE
was never added there; BACE was never rerun with the MCC-computing code since
that rerun only touched Tox21) -- both are recomputed here directly from
already-cached raw data (test_metrics.rmse for LIPO, test_probabilities for
BACE), no new job needed.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef

from evaluation.knn_bace import load_bace_splits_from_deepchem

_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SURFACE = "#fcfcfb"
_MLP_COLOR = "#2a78d6"   # categorical slot 1
_RF_COLOR = "#eb6834"    # categorical slot 2

EPOCHS = [10, 50, 100, 150, 190, 200]

# --- Load summary + raw for every epoch ---
summary = {}
raw = {}
for ep in EPOCHS:
    with open(f"models/KHOP_200EP_EPOCH{ep}/eval_mlp_rf_results.json") as f:
        d = json.load(f)
        summary[ep] = d["summary"]
        raw[ep] = d["raw"]

# --- Recompute LIPO RMSE mean/std across the 5 seeds directly from raw ---
for ep in EPOCHS:
    for method in ("mlp", "rf"):
        rmses = [raw[ep]["lipo"][i]["embeddings"][method]["test_metrics"]["rmse"] for i in range(len(raw[ep]["lipo"]))]
        summary[ep][f"lipo_embeddings_{method}_rmse"] = {"mean": float(np.mean(rmses)), "std": float(np.std(rmses))}

# --- Recompute BACE MCC (single scaffold split, seed=None -- matches how these were originally evaluated) ---
bace_rows, _ = load_bace_splits_from_deepchem("data/MoleculeNet_BACE_custom", "scaffold", split_seed=None)
bace_ytest = np.array([r[1] for r in bace_rows["test"]])
for ep in EPOCHS:
    for method in ("mlp", "rf"):
        proba = np.array(raw[ep]["bace"][0]["embeddings"][method]["test_probabilities"])
        pred = (proba >= 0.5).astype(int)
        summary[ep][f"bace_embeddings_{method}_mcc"] = {"mean": float(matthews_corrcoef(bace_ytest, pred)), "std": 0.0}

data = summary

fig, axes = plt.subplots(2, 3, figsize=(14, 10), facecolor=_SURFACE)
fig.suptitle(
    "KHOP_B1024_H1024_200EP_GAT_9M_ZINC: downstream performance vs. training epoch",
    fontsize=13, fontweight="bold", color=_INK_PRIMARY, x=0.02, y=0.975, ha="left",
)
fig.text(0.02, 0.945, "Checkpoints at epochs 10, 50, 100, 150, 190, 200 -- same model, same eval protocol. Top row: original metric. Bottom row: RMSE / MCC.",
          fontsize=9, color=_INK_SECONDARY, ha="left")
fig.text(0.02, 0.925, "Bands = ±1 std over 5 random splits (LIPO, Tox21). BACE: single fixed scaffold split, no repeats -- no variance to show.",
          fontsize=9, color=_INK_SECONDARY, ha="left")

x_positions = list(range(len(EPOCHS)))  # evenly-spaced categorical positions -- the real epoch
                                          # gaps are uneven (10,50,100,150,190,200) and crowd/overlap
                                          # both the tick labels and the end-point labels on a true
                                          # linear scale, especially the 190/200 pair.

row1 = [
    ("LIPO", "Test R² (embeddings, higher is better)", "lipo_embeddings_mlp_r2", "lipo_embeddings_rf_r2", True),
    ("BACE", "Test ROC-AUC (embeddings, higher is better)", "bace_embeddings_mlp_roc_auc", "bace_embeddings_rf_roc_auc", False),
    ("Tox21", "Mean test ROC-AUC, 12 tasks (embeddings, higher is better)", "tox21_embeddings_mlp_roc_auc", "tox21_embeddings_rf_roc_auc", True),
]
row2 = [
    ("LIPO", "Test RMSE (embeddings, lower is better)", "lipo_embeddings_mlp_rmse", "lipo_embeddings_rf_rmse", True),
    ("BACE", "Test MCC (embeddings, higher is better)", "bace_embeddings_mlp_mcc", "bace_embeddings_rf_mcc", False),
    ("Tox21", "Mean test MCC, 12 tasks (embeddings, higher is better)", "tox21_embeddings_mlp_mcc", "tox21_embeddings_rf_mcc", True),
]

for row_axes, panels in zip(axes, (row1, row2)):
    for ax, (title, ylabel, mlp_key, rf_key, has_std) in zip(row_axes, panels):
        ax.set_facecolor(_SURFACE)
        mlp_vals = [data[ep][mlp_key]["mean"] for ep in EPOCHS]
        rf_vals = [data[ep][rf_key]["mean"] for ep in EPOCHS]

        if has_std:
            mlp_stds = [data[ep][mlp_key]["std"] for ep in EPOCHS]
            rf_stds = [data[ep][rf_key]["std"] for ep in EPOCHS]
            ax.fill_between(x_positions, [m - s for m, s in zip(mlp_vals, mlp_stds)], [m + s for m, s in zip(mlp_vals, mlp_stds)],
                             color=_MLP_COLOR, alpha=0.15, linewidth=0, zorder=2)
            ax.fill_between(x_positions, [m - s for m, s in zip(rf_vals, rf_stds)], [m + s for m, s in zip(rf_vals, rf_stds)],
                             color=_RF_COLOR, alpha=0.15, linewidth=0, zorder=2)

        ax.plot(x_positions, mlp_vals, color=_MLP_COLOR, linewidth=2, marker="o", markersize=8, label="MLP", zorder=3, clip_on=False)
        ax.plot(x_positions, rf_vals, color=_RF_COLOR, linewidth=2, marker="o", markersize=8, label="RF", zorder=3, clip_on=False)

        # Direct label the final point only, not every point.
        ax.annotate(f"{mlp_vals[-1]:.3f}", (x_positions[-1], mlp_vals[-1]), xytext=(10, 4),
                    textcoords="offset points", fontsize=9, color=_MLP_COLOR, fontweight="bold", annotation_clip=False)
        ax.annotate(f"{rf_vals[-1]:.3f}", (x_positions[-1], rf_vals[-1]), xytext=(10, -4),
                    textcoords="offset points", fontsize=9, color=_RF_COLOR, fontweight="bold", annotation_clip=False)

        ax.set_title(title, fontsize=12, fontweight="bold", color=_INK_PRIMARY, pad=10)
        if not has_std:
            ax.text(0.5, 1.0, "single split -- no error band", transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=7.5, color=_INK_MUTED, style="italic")
        ax.set_ylabel(ylabel, fontsize=9, color=_INK_SECONDARY)
        ax.set_xlabel("Epoch", fontsize=9, color=_INK_SECONDARY)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(ep) for ep in EPOCHS])
        ax.set_xlim(-0.3, len(EPOCHS) - 1 + 0.55)  # extra right margin for the end-point labels
        ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(_BASELINE)
        ax.tick_params(axis="both", colors=_INK_MUTED, labelsize=8.5)

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.975), frameon=False, fontsize=9.5, labelcolor=_INK_PRIMARY)
fig.tight_layout(rect=[0, 0, 1, 0.905], h_pad=3.5)

out_path = "models/KHOP_B1024_H1024_200EP_GAT_9M_ZINC/epoch_sweep_mlp_rf.png"
fig.savefig(out_path, dpi=200, facecolor=_SURFACE, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

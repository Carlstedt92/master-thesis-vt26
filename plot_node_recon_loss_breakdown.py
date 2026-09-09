"""Loss breakdown for NODE_RECON_TEST_60EP: total train loss split into its
two components (graph_dino_loss, node_recon_loss), plus the same run's
val_loss, against the exact baseline (MIXED_GAT4HEADS_TEST_60EP -- identical
architecture/augmentation, node loss disabled) it was designed to isolate one
variable against.
"""

import json
import matplotlib.pyplot as plt

_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SURFACE = "#fcfcfb"
_TOTAL_COLOR = "#1a211d"
_GRAPH_COLOR = "#2a78d6"
_NODE_COLOR = "#eb6834"
_BASELINE_COLOR = "#8a8f89"

with open("models/NODE_RECON_TEST_60EP/loss_history.json") as f:
    dino = json.load(f)["DINO_Loss"]
with open("models/MIXED_GAT4HEADS_TEST_60EP/loss_history.json") as f:
    baseline_dino = json.load(f)["DINO_Loss"]

epochs = [r["epoch"] for r in dino]
total_loss = [r["train_loss"] for r in dino]
graph_loss = [r["graph_dino_loss"] for r in dino]
node_loss = [r["node_recon_loss"] for r in dino]
baseline_epochs = [r["epoch"] for r in baseline_dino]
baseline_loss = [r["train_loss"] for r in baseline_dino]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), facecolor=_SURFACE)
fig.suptitle("NODE_RECON_TEST_60EP: training loss breakdown", fontsize=13, fontweight="bold", color=_INK_PRIMARY, x=0.02, y=0.98, ha="left")
fig.text(0.02, 0.93, "Left: full range (epoch 1's spike forces a 0-14 axis, so the flat region is hard to read). Middle: same data, epochs 5-60 only, readable scale. Right: component breakdown.",
          fontsize=9, color=_INK_SECONDARY, ha="left")

ax1.set_facecolor(_SURFACE)
ax1.plot(epochs, total_loss, color=_TOTAL_COLOR, linewidth=2, label="NODE_RECON total (graph+node)", zorder=3)
ax1.plot(baseline_epochs, baseline_loss, color=_BASELINE_COLOR, linewidth=2, linestyle="--", label="MIXED_GAT4HEADS baseline (no node loss)", zorder=3)
ax1.set_title("Total loss vs. baseline", fontsize=12, fontweight="bold", color=_INK_PRIMARY, pad=10)
ax1.set_ylabel("Train loss", fontsize=9, color=_INK_SECONDARY)
ax1.set_xlabel("Epoch", fontsize=9, color=_INK_SECONDARY)
ax1.legend(frameon=False, fontsize=8.5, labelcolor=_INK_PRIMARY, loc="upper right")

ax2.set_facecolor(_SURFACE)
ax2.plot(epochs, total_loss, color=_TOTAL_COLOR, linewidth=2, label="NODE_RECON total", zorder=3)
ax2.plot(baseline_epochs, baseline_loss, color=_BASELINE_COLOR, linewidth=2, linestyle="--", label="baseline total (no node loss)", zorder=3)
ax2.set_title("Zoomed: epochs 5-60", fontsize=12, fontweight="bold", color=_INK_PRIMARY, pad=10)
ax2.set_ylabel("Train loss", fontsize=9, color=_INK_SECONDARY)
ax2.set_xlabel("Epoch", fontsize=9, color=_INK_SECONDARY)
ax2.set_xlim(5, 60)
ax2.set_ylim(0, 3.1)
ax2.legend(frameon=False, fontsize=8.5, labelcolor=_INK_PRIMARY, loc="upper right")

ax3.set_facecolor(_SURFACE)
ax3.plot(epochs, graph_loss, color=_GRAPH_COLOR, linewidth=2, marker="o", markersize=3, label="graph_dino_loss", zorder=3)
ax3.plot(epochs, node_loss, color=_NODE_COLOR, linewidth=2, marker="o", markersize=3, label="node_recon_loss", zorder=3)
ax3.axhline(0.463, color=_BASELINE_COLOR, linewidth=1.3, linestyle="--", zorder=2)
ax3.text(0.02, 0.04, "baseline graph loss floor (ep50, no node loss): 0.463", transform=ax3.transAxes,
          fontsize=8, color=_BASELINE_COLOR, ha="left", va="bottom", style="italic")
ax3.set_title("Component breakdown", fontsize=12, fontweight="bold", color=_INK_PRIMARY, pad=10)
ax3.set_ylabel("Loss", fontsize=9, color=_INK_SECONDARY)
ax3.set_xlabel("Epoch", fontsize=9, color=_INK_SECONDARY)
ax3.legend(frameon=False, fontsize=8.5, labelcolor=_INK_PRIMARY, loc="upper right")

for ax in (ax1, ax2, ax3):
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_BASELINE)
    ax.tick_params(axis="both", colors=_INK_MUTED, labelsize=8.5)

fig.tight_layout(rect=[0, 0, 1, 0.88])
out_path = "models/NODE_RECON_TEST_60EP/loss_breakdown.png"
fig.savefig(out_path, dpi=200, facecolor=_SURFACE, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

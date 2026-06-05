"""Plot embedding-space diagnostics from a DINO loss history JSON file.

The plot shows teacher/student entropy on the left y-axis and embedding
standard deviations on the right y-axis, all against epoch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


SERIES_STYLES = {
    "teacher_entropy": {"label": "Teacher Entropy", "color": "#1f77b4", "linestyle": "-"},
    "student_entropy": {"label": "Student Entropy", "color": "#2ca02c", "linestyle": "--"},
    "embedding_std": {"label": "Embedding Std. Dev.", "color": "#d62728", "linestyle": "-"},
    "encoder_embedding_std": {"label": "Encoder Embedding Std. Dev.", "color": "#ff7f0e", "linestyle": "--"},
}


def _set_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")


def load_loss_history(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_series(records: list[dict], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        if x_key not in row or y_key not in row:
            continue
        value = row.get(y_key)
        if value is None:
            continue
        xs.append(row[x_key])
        ys.append(value)
    return xs, ys


def plot_embedding_space_dynamics(loss_history_path: str | Path, output_path: str | Path, title: str = "Embedding Space Dynamics") -> None:
    loss_history = load_loss_history(loss_history_path)
    records = loss_history.get("DINO_Loss", [])
    if not records:
        raise ValueError("No DINO_Loss entries found in the loss history JSON")

    epochs, teacher_entropy = _extract_series(records, "epoch", "teacher_entropy")
    _, student_entropy = _extract_series(records, "epoch", "student_entropy")
    right_epochs, embedding_std = _extract_series(records, "epoch", "embedding_std")
    _, encoder_embedding_std = _extract_series(records, "epoch", "encoder_embedding_std")

    if not epochs:
        raise ValueError("No epoch data found in DINO_Loss")

    _set_plot_style()
    fig, ax_left = plt.subplots(figsize=(11, 6))

    left_lines = []
    left_labels = []
    for key, values in (("teacher_entropy", teacher_entropy), ("student_entropy", student_entropy)):
        style = SERIES_STYLES[key]
        (line,) = ax_left.plot(epochs, values, color=style["color"], linestyle=style["linestyle"], linewidth=2, label=style["label"])
        left_lines.append(line)
        left_labels.append(style["label"])

    ax_left.set_xlabel("Epoch")
    ax_left.set_ylabel("Entropy")
    ax_left.tick_params(axis="y")

    ax_right = ax_left.twinx()
    right_lines = []
    right_labels = []
    for key, values in (("embedding_std", embedding_std), ("encoder_embedding_std", encoder_embedding_std)):
        style = SERIES_STYLES[key]
        (line,) = ax_right.plot(right_epochs, values, color=style["color"], linestyle=style["linestyle"], linewidth=2, label=style["label"])
        right_lines.append(line)
        right_labels.append(style["label"])

    ax_right.set_ylabel("Embedding Std. Dev.")
    ax_right.tick_params(axis="y")

    lines = left_lines + right_lines
    labels = left_labels + right_labels
    ax_left.set_title(title)
    fig.legend(lines, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot teacher/student entropy and embedding std curves from DINO loss history.")
    parser.add_argument("--input", required=True, help="Path to loss_history.json")
    parser.add_argument("--output", default=None, help="Path to save the plot PNG")
    parser.add_argument("--title", default="Embedding Space Dynamics", help="Plot title")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name("embedding_space_dynamics.png")

    plot_embedding_space_dynamics(input_path, output_path, title=args.title)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _extract_online_knn_validation_metric(online_eval_entry: dict, dataset: str):
    datasets = online_eval_entry.get("evaluation", {}).get("datasets", {})
    dataset_payload = datasets.get(dataset)
    if dataset_payload is None:
        return None, None

    primary_value = dataset_payload.get("primary_metric_value")
    primary_name = dataset_payload.get("primary_metric_name")
    if primary_value is not None and primary_name is not None:
        return primary_value, str(primary_name)

    knn = dataset_payload.get("embeddings", {}).get("knn", {})
    validation_metrics = knn.get("validation_metrics", {})
    if not validation_metrics:
        return None, None

    for metric_name in ("r2", "rmse", "mae", "roc_auc", "f1", "balanced_accuracy"):
        metric_value = validation_metrics.get(metric_name)
        if metric_value is not None:
            return metric_value, f"knn_val_{metric_name}"

    return None, None


def plot_ssl_and_online_knn(loss_history_path: Path, output_path: Path, dataset: str):
    with open(loss_history_path, "r", encoding="utf-8") as f:
        loss_history = json.load(f)

    dino_loss = pd.DataFrame(loss_history.get("DINO_Loss", []))
    online_eval = loss_history.get("Evaluation_Loss", {}).get("online_eval", [])

    if dino_loss.empty:
        raise ValueError("No DINO_Loss entries found in loss_history.json")

    eval_rows = []
    metric_label = "online_val_metric"
    for entry in online_eval:
        value, detected_label = _extract_online_knn_validation_metric(entry, dataset)
        if value is None:
            continue
        if detected_label is not None:
            metric_label = detected_label
        eval_rows.append({"epoch": int(entry["epoch"]), "online_metric": float(value)})

    eval_df = pd.DataFrame(eval_rows)

    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(
        dino_loss["epoch"],
        dino_loss["train_loss"],
        color="#1f77b4",
        label="ssl_train_loss",
        linewidth=2,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("SSL Train Loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    lines = []
    labels = []
    for line in ax1.get_lines():
        lines.append(line)
        labels.append(line.get_label())

    if not eval_df.empty:
        ax2 = ax1.twinx()
        ax2.plot(
            eval_df["epoch"],
            eval_df["online_metric"],
            color="#d62728",
            label=metric_label,
            linewidth=2,
        )
        ax2.set_ylabel(metric_label, color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        for line in ax2.get_lines():
            lines.append(line)
            labels.append(line.get_label())

    plt.title(f"SSL and Online kNN Curves ({dataset})")
    fig.legend(lines, labels, loc="upper right", frameon=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
def main():
    parser = argparse.ArgumentParser(description="Plot SSL loss and online kNN metric per epoch.")
    parser.add_argument("--model", required=True, help="Model directory name under models/.")
    parser.add_argument("--dataset", default="lipo", help="Dataset key in online eval history (e.g., lipo or hiv).")
    parser.add_argument(
        "--loss-history",
        default=None,
        help="Optional explicit path to loss_history.json. Defaults to models/<model>/loss_history.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output image path. Defaults to models/<model>/loss_curves_ssl_knn_<dataset>.png",
    )
    args = parser.parse_args()

    loss_history_path = (
        Path(args.loss_history)
        if args.loss_history is not None
        else Path(f"models/{args.model}/loss_history.json")
    )
    output_path = (
        Path(args.output)
        if args.output is not None
        else Path(f"models/{args.model}/loss_curves_ssl_knn_{args.dataset}.png")
    )

    plot_ssl_and_online_knn(loss_history_path, output_path, args.dataset)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
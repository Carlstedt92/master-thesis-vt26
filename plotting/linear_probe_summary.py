"""Linear probe summary aggregation and visualization."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def update_linear_probe_summary(summary, result):
    """Aggregate linear-probe result into per-dataset summary."""
    dataset_name = result["dataset"]
    primary_metric = result["primary_metric"]
    seed = result.get("split_seed")

    emb_value = float(result["embeddings"]["test_metrics"][primary_metric])
    fp_value = float(result["fingerprints"]["test_metrics"][primary_metric])

    if dataset_name not in summary:
        summary[dataset_name] = {
            "primary_metric": primary_metric,
            "seeds": [],
            "embeddings_test_primary": [],
            "fingerprints_test_primary": [],
        }

    summary[dataset_name]["seeds"].append(seed)
    summary[dataset_name]["embeddings_test_primary"].append(emb_value)
    summary[dataset_name]["fingerprints_test_primary"].append(fp_value)


def finalize_linear_probe_summary(summary):
    """Finalize linear-probe summary with mean/std across seeds."""
    finalized = {}
    for dataset_name, payload in summary.items():
        emb_values = np.asarray(payload["embeddings_test_primary"], dtype=float)
        fp_values = np.asarray(payload["fingerprints_test_primary"], dtype=float)

        if len(emb_values) == 0 or len(fp_values) == 0:
            continue

        ddof = 1 if len(emb_values) > 1 else 0
        finalized[dataset_name] = {
            "primary_metric": payload["primary_metric"],
            "n_runs": int(len(emb_values)),
            "seeds": payload["seeds"],
            "embeddings": {
                "values": emb_values.tolist(),
                "mean": float(np.mean(emb_values)),
                "std": float(np.std(emb_values, ddof=ddof)),
            },
            "fingerprints": {
                "values": fp_values.tolist(),
                "mean": float(np.mean(fp_values)),
                "std": float(np.std(fp_values, ddof=ddof)),
            },
        }

    return finalized


def save_linear_probe_artifacts(model_name: str, summary):
    """Save linear-probe summary JSON and plot."""
    if not summary:
        return

    model_dir = Path(f"models/{model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)

    json_path = model_dir / "linear_probe_random_seed_summary.json"
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    datasets = list(summary.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset_name in enumerate(datasets):
        payload = summary[dataset_name]
        axis = axes[idx]
        metric_name = str(payload["primary_metric"]).lower()
        metric_label = "Mean ROC-AUC" if "roc_auc" in metric_name else "Mean RMSE"

        means = [payload["embeddings"]["mean"], payload["fingerprints"]["mean"]]
        stds = [payload["embeddings"]["std"], payload["fingerprints"]["std"]]

        axis.bar([0, 1], means, yerr=stds, capsize=4, color=["tab:blue", "tab:orange"])
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Embeddings", "Fingerprints"])
        axis.set_title(f"{dataset_name} ({payload['primary_metric']})")
        axis.set_ylabel(metric_label)
        axis.grid(axis="y", alpha=0.3)

        if "roc_auc" in metric_name:
            axis.set_ylim(0.0, 1.0)

    title_model_name = model_name.split("/")[0]
    title_suffix = " (finetuned result)" if "finetune_eval_logs" in model_name else ""
    fig.suptitle(f"Linear Probe Performance Across Runs for model: {title_model_name}{title_suffix}")
    fig.tight_layout()
    plot_path = model_dir / "linear_probe_random_seed_summary.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"Saved linear-probe summary JSON: {json_path}")
    print(f"Saved linear-probe summary plot: {plot_path}")

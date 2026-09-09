"""Comparison figure for finetuned vs non-finetuned downstream results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_ORDER = ["LIPO", "BACE", "Tox21"]
FAMILY_ORDER = ["knn", "linear_probe", "random_forest"]
FAMILY_TITLES = {
    "knn": "KNN",
    "linear_probe": "Linear Probe",
    "random_forest": "Random Forest",
}
FAMILY_LABEL = "R2 (LIPO) / ROC AUC (BACE, Tox21)"


def _read_json(path: Path):
    with open(path, "r") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        return payload["result"]
    return payload


def _metric_key(dataset: str) -> str:
    if dataset == "LIPO":
        return "r2"
    if dataset == "BACE":
        return "roc_auc"
    if dataset == "Tox21":
        return "roc_auc_mean_tasks"
    raise ValueError(f"Unsupported dataset: {dataset}")


def _non_finetuned_paths(model_dir: Path, family: str, dataset: str):
    if family == "knn":
        if dataset == "Tox21":
            pattern = model_dir / "eval_logs" / "knn" / "TOX21" / "knn_tox21*.json"
        else:
            pattern = model_dir / "eval_logs" / "knn" / dataset / f"knn_{dataset.lower()}*.json"
    elif family == "linear_probe":
        pattern = model_dir / "eval_logs" / "linear_probe" / dataset.lower() / f"linear_probe_{dataset.lower()}*.json"
    elif family == "random_forest":
        if dataset == "Tox21":
            pattern = model_dir / "eval_logs" / "random_forest" / "TOX21" / "rf_tox21_seed*.json"
        else:
            pattern = model_dir / "eval_logs" / "random_forest" / dataset / f"rf_{dataset.lower()}_seed*.json"
    else:
        raise ValueError(f"Unsupported family: {family}")

    return sorted(pattern.parent.glob(pattern.name))


def _finetuned_path(model_dir: Path, dataset: str):
    return sorted((model_dir / "finetune_eval_logs").glob(f"finetune_eval_{dataset.lower()}_seed*.json"))


def _extract_non_finetuned_values(model_dir: Path, family: str, dataset: str):
    metric_key = _metric_key(dataset)
    emb_values = []
    fp_values = []

    for path in _non_finetuned_paths(model_dir, family, dataset):
        payload = _read_json(path)
        emb_values.append(float(payload["embeddings"]["test_metrics"][metric_key]))
        fp_values.append(float(payload["fingerprints"]["test_metrics"][metric_key]))

    if not emb_values or not fp_values:
        return None

    ddof = 1 if len(emb_values) > 1 else 0
    return {
        "n_runs": int(len(emb_values)),
        "embeddings": {
            "values": emb_values,
            "mean": float(np.mean(emb_values)),
            "std": float(np.std(emb_values, ddof=ddof)),
        },
        "fingerprints": {
            "values": fp_values,
            "mean": float(np.mean(fp_values)),
            "std": float(np.std(fp_values, ddof=ddof)),
        },
    }


def _extract_finetuned_values(model_dir: Path, family: str, dataset: str):
    paths = _finetuned_path(model_dir, dataset)
    if not paths:
        return None

    metric_key = _metric_key(dataset)
    values = []

    for path in paths:
        payload = _read_json(path)
        if dataset == "Tox21":
            value = float(payload["embeddings"]["mean_test_roc_auc"][family])
        else:
            value = float(payload["embeddings"][family]["test_metrics"][metric_key])
        values.append(value)

    ddof = 1 if len(values) > 1 else 0
    return {
        "n_runs": int(len(values)),
        "values": values,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=ddof)),
    }


def build_comparison_summary(model_name: str):
    model_dir = Path(f"models/{model_name}")
    summary = {}

    for family in FAMILY_ORDER:
        family_summary = {}
        for dataset in DATASET_ORDER:
            non_finetuned = _extract_non_finetuned_values(model_dir, family, dataset)
            finetuned = _extract_finetuned_values(model_dir, family, dataset)
            if non_finetuned is None or finetuned is None:
                continue

            family_summary[dataset] = {
                "metric": _metric_key(dataset),
                "non_finetuned": non_finetuned,
                "finetuned": finetuned,
                "fingerprints": {
                    "values": list(non_finetuned["fingerprints"]["values"]),
                    "mean": float(non_finetuned["fingerprints"]["mean"]),
                    "std": float(non_finetuned["fingerprints"]["std"]),
                },
            }

        if family_summary:
            summary[family] = family_summary

    return summary


def render_comparison_plot(summary, output_path: Path):
    if not summary:
        raise ValueError("No comparison data available.")

    fig, axes = plt.subplots(len(FAMILY_ORDER), 1, figsize=(12, 13), sharex=True)
    if len(FAMILY_ORDER) == 1:
        axes = [axes]

    x_positions = np.arange(len(DATASET_ORDER))
    bar_width = 0.24
    offsets = [-bar_width, 0.0, bar_width]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["Non-finetuned avg (5 seeds)", "Finetuned avg (seeds)", "Fingerprints avg (5 seeds)"]

    for axis, family in zip(axes, FAMILY_ORDER):
        family_data = summary.get(family, {})

        for offset, color, label_idx, label in zip(offsets, colors, range(3), labels):
            heights = []
            for dataset in DATASET_ORDER:
                payload = family_data.get(dataset)
                if payload is None:
                    heights.append(np.nan)
                elif label_idx == 0:
                    heights.append(float(payload["non_finetuned"]["embeddings"]["mean"]))
                elif label_idx == 1:
                    heights.append(float(payload["finetuned"]["mean"]))
                else:
                    heights.append(float(payload["fingerprints"]["mean"]))

            axis.bar(x_positions + offset, heights, width=bar_width, color=color, label=label)

        axis.set_title(FAMILY_TITLES[family])
        axis.set_ylabel(FAMILY_LABEL)
        axis.grid(axis="y", alpha=0.3)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(DATASET_ORDER)
        axis.set_xlim(-0.5, len(DATASET_ORDER) - 0.5)

        ymin, ymax = axis.get_ylim()
        axis.set_ylim(bottom=min(0.0, ymin), top=max(ymax, 1.0) if family != "knn" else ymax)

    axes[-1].set_xlabel("Dataset")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Finetuned vs non-finetuned downstream performance")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_comparison_artifacts(model_name: str):
    summary = build_comparison_summary(model_name)
    if not summary:
        return None

    model_dir = Path(f"models/{model_name}/finetune_eval_logs")
    model_dir.mkdir(parents=True, exist_ok=True)

    json_path = model_dir / "finetune_eval_comparison_3x1.json"
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    plot_path = model_dir / "finetune_eval_comparison_3x1.png"
    render_comparison_plot(summary, plot_path)

    print(f"Saved comparison summary: {json_path}")
    print(f"Saved comparison plot: {plot_path}")
    return summary, json_path, plot_path
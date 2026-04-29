"""Build architecture-level KNN comparison plots from per-model random-seed summaries.

Expected input per model:
- models/<model_name>/knn_random_seed_summary.json

This script aggregates:
1) Within each model seed: already averaged over split seeds by eval_many_models.py.
2) Across model seeds per architecture (e.g., 3 GINE models, 3 GAT models).
3) Morgan baseline across all provided model seeds.

Example usage: 
    python architecture_knn_summary_plot.py --gine-models gine1,gine2,gine3 --gat-models gat1,gat2,gat3 --datasets dataset1,dataset2

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_summary(model_name: str) -> dict:
    path = Path("models") / model_name / "knn_random_seed_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary for model '{model_name}': {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_values(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "values": []}
    ddof = 1 if arr.size > 1 else 0
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=ddof)),
        "values": arr.tolist(),
    }


def collect_dataset_payload(models: list[str], dataset: str) -> tuple[list[float], list[float], str]:
    emb_values = []
    fp_values = []
    primary_metric = None

    for model_name in models:
        summary = load_summary(model_name)
        if dataset not in summary:
            raise KeyError(f"Dataset '{dataset}' not found in summary for model '{model_name}'.")

        payload = summary[dataset]
        metric = payload["primary_metric"]
        if primary_metric is None:
            primary_metric = metric
        elif primary_metric != metric:
            raise ValueError(
                f"Metric mismatch for dataset '{dataset}': got both '{primary_metric}' and '{metric}'."
            )

        emb_values.append(float(payload["embeddings"]["mean"]))
        fp_values.append(float(payload["fingerprints"]["mean"]))

    return emb_values, fp_values, primary_metric


def discover_common_datasets(model_names: list[str]) -> list[str]:
    if not model_names:
        return []

    dataset_sets = []
    for model_name in model_names:
        summary = load_summary(model_name)
        dataset_sets.append(set(summary.keys()))

    common = set.intersection(*dataset_sets) if dataset_sets else set()
    return sorted(common)


def metric_slug(metric_name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in metric_name).strip("_").lower()


def plot_dataset_group(datasets: list[str], results: dict, output_plot: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        payload = results[dataset]
        primary_metric = payload["primary_metric"]

        axis = axes[idx]
        labels = ["GINE", "GAT", "Morgan"]
        means = [
            payload["gine_embeddings"]["mean"],
            payload["gat_embeddings"]["mean"],
            payload["morgan_fingerprints"]["mean"],
        ]
        stds = [
            payload["gine_embeddings"]["std"],
            payload["gat_embeddings"]["std"],
            payload["morgan_fingerprints"]["std"],
        ]

        axis.bar(
            [0, 1, 2],
            means,
            yerr=stds,
            capsize=4,
            color=["#1b9e77", "#d95f02", "#7570b3"],
        )
        axis.set_xticks([0, 1, 2])
        axis.set_xticklabels(labels)
        axis.set_title(f"{dataset} ({primary_metric})")

        if primary_metric in {"rmse", "mae", "mse"}:
            axis.set_ylabel("Mean score (lower is better)")
        else:
            axis.set_ylabel("Mean score (higher is better)")

        if "roc_auc" in primary_metric:
            axis.set_ylim(0.0, 1.0)

        axis.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=300)
    plt.close(fig)


def build_plot_and_report(
    gine_models: list[str],
    gat_models: list[str],
    datasets: list[str],
    output_plot: Path,
    output_json: Path,
) -> None:
    all_models = gine_models + gat_models
    if not all_models:
        raise ValueError("No model names provided.")

    if not datasets:
        datasets = discover_common_datasets(all_models)

    if not datasets:
        raise ValueError("No common datasets found across the provided models.")

    results = {}

    for dataset in datasets:
        gine_emb, gine_fp, primary_metric = collect_dataset_payload(gine_models, dataset)
        gat_emb, gat_fp, primary_metric_gat = collect_dataset_payload(gat_models, dataset)

        if primary_metric != primary_metric_gat:
            raise ValueError(
                f"Metric mismatch between architectures for dataset '{dataset}': "
                f"{primary_metric} vs {primary_metric_gat}."
            )

        gine_stats = aggregate_values(gine_emb)
        gat_stats = aggregate_values(gat_emb)

        # Morgan baseline is architecture-agnostic; average across all provided model seeds.
        morgan_stats = aggregate_values(gine_fp + gat_fp)

        results[dataset] = {
            "primary_metric": primary_metric,
            "gine_embeddings": gine_stats,
            "gat_embeddings": gat_stats,
            "morgan_fingerprints": morgan_stats,
            "source_models": {
                "gine": gine_models,
                "gat": gat_models,
            },
        }

    plot_dataset_group(
        datasets=datasets,
        results=results,
        output_plot=output_plot,
        title="Architecture-level KNN Summary (model seeds x split seeds)",
    )

    datasets_by_metric = {}
    for dataset_name, payload in results.items():
        metric = payload["primary_metric"]
        datasets_by_metric.setdefault(metric, []).append(dataset_name)

    if len(datasets_by_metric) > 1:
        for metric_name, metric_datasets in datasets_by_metric.items():
            metric_output_plot = output_plot.with_name(
                f"{output_plot.stem}_{metric_slug(metric_name)}{output_plot.suffix}"
            )
            plot_dataset_group(
                datasets=metric_datasets,
                results=results,
                output_plot=metric_output_plot,
                title=(
                    "Architecture-level KNN Summary "
                    f"({metric_name}; model seeds x split seeds)"
                ),
            )
            print(f"Saved metric-specific summary plot: {metric_output_plot}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved architecture summary plot: {output_plot}")
    print(f"Saved architecture summary JSON: {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a final comparison plot for GINE vs GAT vs Morgan from saved KNN summaries."
    )
    parser.add_argument(
        "--gine-models",
        type=str,
        required=True,
        help="Comma-separated model folders for GINE seeds.",
    )
    parser.add_argument(
        "--gat-models",
        type=str,
        required=True,
        help="Comma-separated model folders for GAT seeds.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma-separated datasets to include (default: common datasets across all models).",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="models/architecture_knn_summary.png",
        help="Output path for the summary figure.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="models/architecture_knn_summary.json",
        help="Output path for the aggregated summary JSON.",
    )

    args = parser.parse_args()

    gine_models = parse_csv(args.gine_models)
    gat_models = parse_csv(args.gat_models)
    datasets = parse_csv(args.datasets) if args.datasets else []

    if not gine_models:
        raise ValueError("--gine-models must contain at least one model.")
    if not gat_models:
        raise ValueError("--gat-models must contain at least one model.")

    build_plot_and_report(
        gine_models=gine_models,
        gat_models=gat_models,
        datasets=datasets,
        output_plot=Path(args.output_plot),
        output_json=Path(args.output_json),
    )


if __name__ == "__main__":
    main()

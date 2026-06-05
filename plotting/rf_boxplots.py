"""Create RF summary plots from a single model folder.

Loads files named rf_eval_<dataset>.json from models/<model>/ and plots
the primary metric values for embeddings vs fingerprints.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def update_random_seed_summary(summary, result):
    """Aggregate one RF evaluation run into a per-dataset summary."""
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


def finalize_random_seed_summary(summary):
    """Finalize RF summary with mean/std across runs."""
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


def save_random_seed_artifacts(model_name: str, summary):
    """Save RF summary JSON and barplot figure."""
    if not summary:
        return

    model_dir = Path(f"models/{model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)

    json_path = model_dir / "rf_random_seed_summary.json"
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
    fig.suptitle(f"Random Forest Performance Across Runs for model: {title_model_name}{title_suffix}")
    fig.tight_layout()

    plot_path = model_dir / "rf_random_seed_summary.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"Saved RF summary JSON: {json_path}")
    print(f"Saved RF summary plot: {plot_path}")
def parse_csv(values: str):
    return [item.strip() for item in values.split(",") if item.strip()]


def find_rf_eval_files(model_dir: Path):
    candidates = sorted(model_dir.glob("rf_eval_*.json"))
    eval_files = []
    for path in candidates:
        # Accept only rf_eval_<dataset>.json and skip generated summaries.
        stem = path.stem
        if not stem.startswith("rf_eval_"):
            continue
        suffix = stem[len("rf_eval_") :]
        if suffix in {"barplots_summary", "boxplots_summary"}:
            continue
        eval_files.append(path)
    return eval_files


def load_metric_from_file(path: Path, metric_split: str):
    with open(path, "r") as handle:
        payload = json.load(handle)

    dataset = str(payload["dataset"])
    primary_metric = str(payload["primary_metric"])

    emb_metrics = payload["embeddings"][f"{metric_split}_metrics"]
    fp_metrics = payload["fingerprints"][f"{metric_split}_metrics"]
    emb_candidates = payload["embeddings"].get(f"candidate_{metric_split}_metrics", [])
    fp_candidates = payload["fingerprints"].get(f"candidate_{metric_split}_metrics", [])

    emb_value = emb_metrics.get(primary_metric)
    fp_value = fp_metrics.get(primary_metric)

    if emb_value is None or fp_value is None:
        raise ValueError(
            f"Missing primary metric '{primary_metric}' in {metric_split}_metrics for {path}"
        )

    return dataset, primary_metric, float(emb_value), float(fp_value), emb_candidates, fp_candidates


def collect_results(model_dir: Path, datasets_filter, metric_split: str):
    results = {}
    for json_path in find_rf_eval_files(model_dir):
        dataset, primary_metric, emb_value, fp_value, emb_candidates, fp_candidates = load_metric_from_file(
            json_path, metric_split
        )

        if datasets_filter and dataset.lower() not in datasets_filter:
            continue

        if dataset not in results:
            results[dataset] = {
                "primary_metric": primary_metric,
                "embeddings": emb_value,
                "fingerprints": fp_value,
                "embeddings_candidates": emb_candidates,
                "fingerprints_candidates": fp_candidates,
            }

    return results


def summarize_results(results, model_name: str, metric_split: str):
    summary = {}
    for dataset, payload in results.items():
        summary[dataset] = {
            "model": model_name,
            "metric_split": metric_split,
            "primary_metric": payload["primary_metric"],
            "embeddings": float(payload["embeddings"]),
            "fingerprints": float(payload["fingerprints"]),
        }
    return summary


def _candidate_values(payload, key: str, primary_metric: str):
    values = []
    for item in payload.get(key, []):
        metric_value = item.get(primary_metric)
        if metric_value is not None:
            values.append(float(metric_value))
    return values


def render_boxplots(results, output_path: Path, model_name: str, metric_split: str):
    preferred_order = ["LIPO", "Tox21", "BACE"]
    present = set(results.keys())
    ordered = [name for name in preferred_order if name in present]
    remaining = sorted([name for name in results.keys() if name not in ordered])
    datasets = ordered + remaining
    if not datasets:
        raise ValueError("No RF evaluation JSON files matched the selection.")

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        payload = results[dataset]
        axis = axes[idx]
        metric_name = str(payload["primary_metric"])

        emb_values = _candidate_values(payload, "embeddings_candidates", metric_name)
        fp_values = _candidate_values(payload, "fingerprints_candidates", metric_name)

        if not emb_values:
            emb_values = [float(payload["embeddings"])]
        if not fp_values:
            fp_values = [float(payload["fingerprints"])]

        bp = axis.boxplot(
            [emb_values, fp_values],
            labels=["Embeddings", "Fingerprints"],
            patch_artist=True,
            showmeans=True,
        )
        colors = ["tab:blue", "tab:orange"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)

        for position, values, color in zip([1, 2], [emb_values, fp_values], colors):
            jitter = np.random.default_rng(42 + idx).normal(position, 0.04, size=len(values))
            axis.scatter(jitter, values, s=16, color=color, alpha=0.55, edgecolors="none")

        axis.set_title(f"{dataset} ({metric_name})")
        axis.set_ylabel(f"{metric_split.title()} {metric_name}")
        axis.grid(axis="y", alpha=0.3)

        if "roc_auc" in metric_name.lower():
            axis.set_ylim(0.0, 1.0)

    fig.suptitle(
        f"Random Forest: Hyperparameter search distributions for {model_name} ({metric_split} metrics)"
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def render_barplots(results, output_path: Path, model_name: str, metric_split: str):
    preferred_order = ["LIPO", "Tox21", "BACE"]
    present = set(results.keys())
    ordered = [name for name in preferred_order if name in present]
    remaining = sorted([name for name in results.keys() if name not in ordered])
    datasets = ordered + remaining
    if not datasets:
        raise ValueError("No RF evaluation JSON files matched the selection.")

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        payload = results[dataset]
        axis = axes[idx]

        emb_value = float(payload["embeddings"])
        fp_value = float(payload["fingerprints"])
        metric_name = str(payload["primary_metric"]).lower()

        axis.bar([0, 1], [emb_value, fp_value], color=["tab:blue", "tab:orange"])
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Embeddings", "Fingerprints"])

        axis.set_title(f"{dataset} ({payload['primary_metric']})")
        axis.set_ylabel(f"{metric_split.title()} {payload['primary_metric']}")
        axis.grid(axis="y", alpha=0.3)

        if "roc_auc" in metric_name:
            axis.set_ylim(0.0, 1.0)

    fig.suptitle(
        f"Random Forest: Embeddings vs Fingerprints for {model_name} ({metric_split} metrics)"
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name under models/<name>/")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma-separated dataset filter, e.g. lipo,bace,tox21",
    )
    parser.add_argument(
        "--metric-split",
        choices=["test", "validation"],
        default="test",
        help="Which metric split to plot",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=None,
        help="Optional custom output path for the barplot figure",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=None,
        help="Output path for aggregated summary JSON",
    )

    args = parser.parse_args()

    datasets_filter = {item.lower() for item in parse_csv(args.datasets)}
    models_dir = Path(args.models_dir)
    model_dir = models_dir / args.model

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    output_plot = (
        Path(args.output_plot)
        if args.output_plot is not None
        else model_dir / "rf_eval_barplots.png"
    )
    output_summary = (
        Path(args.output_summary)
        if args.output_summary is not None
        else model_dir / "rf_eval_barplots_summary.json"
    )

    results = collect_results(model_dir, datasets_filter, args.metric_split)
    summary = summarize_results(results, args.model, args.metric_split)

    render_barplots(results, output_plot, args.model, args.metric_split)

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved RF barplot figure: {output_plot}")
    print(f"Saved RF barplot summary: {output_summary}")


if __name__ == "__main__":
    main()

"""Run multiple KNN downstream evaluations for multiple SSL model folders.

This runs LIPO, Tox21, and BACE evaluators for each model.

Example Usage: 
    python eval_many_models.py --models model1,model2,model3 --k_values 5,10,15

"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from knn_eval_bace import run_knn_eval_bace
from knn_eval_lip import run_knn_eval_lipo
from knn_eval_tox21 import run_knn_eval_tox21


def parse_models(models_csv: str):
    return [name.strip() for name in models_csv.split(",") if name.strip()]


def parse_int_list(values_csv: str):
    return [int(value.strip()) for value in values_csv.split(",") if value.strip()]


def update_random_seed_summary(summary, result):
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
    if not summary:
        return

    model_dir = Path(f"models/{model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)

    json_path = model_dir / "knn_random_seed_summary.json"
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

    fig.suptitle(f"Average Performance Across Runs for model: {model_name}")
    fig.tight_layout()
    plot_path = model_dir / "knn_random_seed_summary.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"Saved run summary JSON: {json_path}")
    print(f"Saved run summary plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Run KNN evals (LIPO, Tox21, BACE) for multiple SSL models.")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated model names, e.g. GDZ_5000Epochs,GDZ_GAT_KHOP",
    )

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--lipo-data-dir", type=str, default="data/MoleculeNet_LIPO_custom")
    parser.add_argument("--lipo-splitter", type=str, default="random")

    parser.add_argument("--tox21-data-dir", type=str, default="data/MoleculeNet_Tox21_custom")
    parser.add_argument("--tox21-splitter", type=str, default="random")

    parser.add_argument("--bace-data-dir", type=str, default="data/MoleculeNet_BACE_custom")
    parser.add_argument("--bace-splitter", type=str, default="scaffold")
    parser.add_argument(
        "--random-split-seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds used only when a dataset splitter is random.",
    )

    args = parser.parse_args()

    model_names = parse_models(args.models)
    if not model_names:
        raise ValueError("No model names provided via --models")

    random_split_seeds = parse_int_list(args.random_split_seeds)
    if (args.lipo_splitter == "random" or args.tox21_splitter == "random") and not random_split_seeds:
        raise ValueError("At least one --random-split-seeds value is required for random split datasets.")

    for model_name in model_names:
        print(f"\n=== Evaluating {model_name} ===")
        random_seed_summary = {}

        if args.lipo_splitter == "random":
            print(f"\n--- LIPO (random seeds: {random_split_seeds}) ---")
            for seed in random_split_seeds:
                try:
                    print(f"\n[LIPO] Running seed={seed}")
                    result = run_knn_eval_lipo(
                        ssl_model_name=model_name,
                        checkpoint_path=None,
                        lipo_data_dir=args.lipo_data_dir,
                        lipo_splitter=args.lipo_splitter,
                        split_seed=seed,
                        device_preference=args.device,
                    )
                    update_random_seed_summary(random_seed_summary, result)
                except Exception as exc:
                    print(f"[FAILED][LIPO][seed={seed}] {model_name}: {exc}")
        else:
            try:
                print("\n--- LIPO ---")
                result = run_knn_eval_lipo(
                    ssl_model_name=model_name,
                    checkpoint_path=None,
                    lipo_data_dir=args.lipo_data_dir,
                    lipo_splitter=args.lipo_splitter,
                    split_seed=None,
                    device_preference=args.device,
                )
                update_random_seed_summary(random_seed_summary, result)
            except Exception as exc:
                print(f"[FAILED][LIPO] {model_name}: {exc}")

        if args.tox21_splitter == "random":
            print(f"\n--- Tox21 (random seeds: {random_split_seeds}) ---")
            for seed in random_split_seeds:
                try:
                    print(f"\n[Tox21] Running seed={seed}")
                    result = run_knn_eval_tox21(
                        ssl_model_name=model_name,
                        checkpoint_path=None,
                        tox21_data_dir=args.tox21_data_dir,
                        tox21_splitter=args.tox21_splitter,
                        split_seed=seed,
                        device_preference=args.device,
                    )
                    update_random_seed_summary(random_seed_summary, result)
                except Exception as exc:
                    print(f"[FAILED][Tox21][seed={seed}] {model_name}: {exc}")
        else:
            try:
                print("\n--- Tox21 ---")
                result = run_knn_eval_tox21(
                    ssl_model_name=model_name,
                    checkpoint_path=None,
                    tox21_data_dir=args.tox21_data_dir,
                    tox21_splitter=args.tox21_splitter,
                    split_seed=None,
                    device_preference=args.device,
                )
                update_random_seed_summary(random_seed_summary, result)
            except Exception as exc:
                print(f"[FAILED][Tox21] {model_name}: {exc}")

        try:
            print("\n--- BACE ---")
            result = run_knn_eval_bace(
                ssl_model_name=model_name,
                checkpoint_path=None,
                bace_data_dir=args.bace_data_dir,
                bace_splitter=args.bace_splitter,
                device_preference=args.device,
            )
            update_random_seed_summary(random_seed_summary, result)
        except Exception as exc:
            print(f"[FAILED][BACE] {model_name}: {exc}")

        finalized_summary = finalize_random_seed_summary(random_seed_summary)
        if finalized_summary:
            save_random_seed_artifacts(model_name, finalized_summary)


if __name__ == "__main__":
    main()

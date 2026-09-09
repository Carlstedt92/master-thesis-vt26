"""Phase 2 of the two-phase MLP/RF eval: fit MLP/RF probes on the embeddings
extracted by eval_phase1_extract_embeddings.py, plus Morgan fingerprints.

Fingerprints don't depend on the model at all, so they're built once per
(dataset, seed) and reused across every model in this run, instead of being
recomputed per model like the original combined script did. MLP training
here runs on tiny, already-extracted feature arrays (no per-molecule RDKit
or big-encoder GPU calls), so it stays fast even without a GPU -- this
phase is designed to run CPU-only.

Results are saved in the exact same shape/convention as
eval_many_models_mlp_rf.py (same per-model JSON, same combined output file,
same comparison plot function) so downstream tooling doesn't need to care
which script produced them.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.knn_bace import (
    build_fingerprint_features as build_bace_fingerprint_features,
    load_bace_admet_benchmark_splits,
    load_bace_splits_from_deepchem,
)
from evaluation.knn_lipo import (
    build_fingerprint_features as build_lipo_fingerprint_features,
    load_lipo_splits_from_deepchem,
    resolve_torch_device,
)
from evaluation.knn_tox21 import build_fingerprint_features as build_tox21_fingerprint_features, load_tox21_splits_from_deepchem
from evaluation.tdc_datasets import (
    build_fingerprint_features as build_tdc_fingerprint_features,
    load_tdc_admet_benchmark_splits,
    load_tdc_splits_from_tdc,
)
from evaluation.mlp_rf import evaluate_mlp_classification, evaluate_mlp_regression, evaluate_rf_classification, evaluate_rf_regression
from eval_many_models_mlp_rf import generate_dataset_comparison_plots, _aggregate_seeds

TDC_DATASETS = ("bbb_martins", "herg", "ames")

_fp_cache = {}


def _get_fingerprints(dataset, seed, lipo_data_dir, lipo_splitter, bace_data_dir, bace_splitter, tox21_data_dir, tox21_splitter,
                       tdc_data_dir=None, tdc_splitter=None):
    key = (dataset, seed)
    if key in _fp_cache:
        return _fp_cache[key]

    if dataset in TDC_DATASETS:
        if tdc_splitter == "admet_benchmark":
            rows_by_split, _ = load_tdc_admet_benchmark_splits(dataset, tdc_data_dir, split_seed=seed)
        else:
            rows_by_split, _ = load_tdc_splits_from_tdc(dataset, tdc_data_dir, tdc_splitter, split_seed=seed)
        fp = {}
        for split in ("train", "val", "test"):
            X, y, _ = build_tdc_fingerprint_features(rows_by_split[split])
            fp[split] = (X, y)
    elif dataset == "lipo":
        rows_by_split, _ = load_lipo_splits_from_deepchem(lipo_data_dir, lipo_splitter, split_seed=seed)
        fp = {}
        for split in ("train", "val", "test"):
            X, y, _ = build_lipo_fingerprint_features(rows_by_split[split])
            fp[split] = (X, y)
    elif dataset == "bace":
        if bace_splitter == "admet_benchmark":
            rows_by_split, _ = load_bace_admet_benchmark_splits(bace_data_dir, split_seed=seed)
        else:
            rows_by_split, _ = load_bace_splits_from_deepchem(bace_data_dir, bace_splitter, split_seed=seed)
        fp = {}
        for split in ("train", "val", "test"):
            X, y, _ = build_bace_fingerprint_features(rows_by_split[split])
            fp[split] = (X, y)
    elif dataset == "tox21":
        data_by_split, split_stats = load_tox21_splits_from_deepchem(tox21_data_dir, tox21_splitter, split_seed=seed)
        fp = {"_task_names": split_stats["task_names"]}
        for split in ("train", "val", "test"):
            smiles = data_by_split[split]["smiles"]
            labels = data_by_split[split]["labels"]
            X, kept, _ = build_tox21_fingerprint_features(smiles)
            fp[split] = (X, labels[kept])
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    _fp_cache[key] = fp
    return fp


def _load_embeddings(cache_dir, model_name, dataset, seed):
    seed_suffix = f"seed{seed}" if seed is not None else "seedNone"
    path = Path(cache_dir) / model_name / f"{dataset}_{seed_suffix}.npz"
    data = np.load(path)
    return {"train": (data["train_X"], data["train_y"]), "val": (data["val_X"], data["val_y"]), "test": (data["test_X"], data["test_y"])}


def fit_lipo(cache_dir, model_name, seed, lipo_data_dir, lipo_splitter, device):
    emb = _load_embeddings(cache_dir, model_name, "lipo", seed)
    fp = _get_fingerprints("lipo", seed, lipo_data_dir, lipo_splitter, None, None, None, None)

    seed_for_rf = seed if seed is not None else 0
    mlp_emb = evaluate_mlp_regression(*emb["train"], *emb["val"], *emb["test"], device)
    rf_emb = evaluate_rf_regression(*emb["train"], *emb["val"], *emb["test"], seed=seed_for_rf)
    mlp_fp = evaluate_mlp_regression(*fp["train"], *fp["val"], *fp["test"], device)
    rf_fp = evaluate_rf_regression(*fp["train"], *fp["val"], *fp["test"], seed=seed_for_rf)

    print(f"[LIPO seed={seed}] test R2  MLP(emb)={mlp_emb['test_metrics']['r2']:.4f}  RF(emb)={rf_emb['test_metrics']['r2']:.4f}  "
          f"MLP(fp)={mlp_fp['test_metrics']['r2']:.4f}  RF(fp)={rf_fp['test_metrics']['r2']:.4f}")

    return {"dataset": "LIPO", "split_seed": seed, "primary_metric": "rmse",
            "embeddings": {"mlp": mlp_emb, "rf": rf_emb}, "fingerprints": {"mlp": mlp_fp, "rf": rf_fp}}


def fit_bace(cache_dir, model_name, seed, bace_data_dir, bace_splitter, device):
    emb = _load_embeddings(cache_dir, model_name, "bace", seed)
    fp = _get_fingerprints("bace", seed, None, None, bace_data_dir, bace_splitter, None, None)

    seed_for_rf = seed if seed is not None else 0
    mlp_emb = evaluate_mlp_classification(*emb["train"], *emb["val"], *emb["test"], device)
    rf_emb = evaluate_rf_classification(*emb["train"], *emb["val"], *emb["test"], seed=seed_for_rf)
    mlp_fp = evaluate_mlp_classification(*fp["train"], *fp["val"], *fp["test"], device)
    rf_fp = evaluate_rf_classification(*fp["train"], *fp["val"], *fp["test"], seed=seed_for_rf)

    print(f"[BACE seed={seed}] test ROC-AUC  MLP(emb)={mlp_emb['test_metrics']['roc_auc']:.4f}  RF(emb)={rf_emb['test_metrics']['roc_auc']:.4f}  "
          f"MLP(fp)={mlp_fp['test_metrics']['roc_auc']:.4f}  RF(fp)={rf_fp['test_metrics']['roc_auc']:.4f}")

    return {"dataset": "BACE", "split_seed": seed, "primary_metric": "roc_auc",
            "embeddings": {"mlp": mlp_emb, "rf": rf_emb}, "fingerprints": {"mlp": mlp_fp, "rf": rf_fp}}


# Display name for each TDC dataset -- used only for print()/result labeling below.
_TDC_LABELS = {"bbb_martins": "BBB_Martins", "herg": "hERG", "ames": "AMES"}


def fit_tdc_classification(dataset_key, cache_dir, model_name, seed, tdc_data_dir, tdc_splitter, device):
    """Single-task binary classification fit, structurally identical to
    fit_bace -- shared across bbb_martins/herg/ames instead of copy-pasting
    fit_bace three times, since (unlike LIPO/BACE/Tox21, which each have a
    different task shape) all three TDC datasets here are literally the
    same shape as BACE."""
    label = _TDC_LABELS[dataset_key]
    emb = _load_embeddings(cache_dir, model_name, dataset_key, seed)
    fp = _get_fingerprints(dataset_key, seed, None, None, None, None, None, None, tdc_data_dir, tdc_splitter)

    seed_for_rf = seed if seed is not None else 0
    mlp_emb = evaluate_mlp_classification(*emb["train"], *emb["val"], *emb["test"], device)
    rf_emb = evaluate_rf_classification(*emb["train"], *emb["val"], *emb["test"], seed=seed_for_rf)
    mlp_fp = evaluate_mlp_classification(*fp["train"], *fp["val"], *fp["test"], device)
    rf_fp = evaluate_rf_classification(*fp["train"], *fp["val"], *fp["test"], seed=seed_for_rf)

    print(f"[{label} seed={seed}] test ROC-AUC  MLP(emb)={mlp_emb['test_metrics']['roc_auc']:.4f}  RF(emb)={rf_emb['test_metrics']['roc_auc']:.4f}  "
          f"MLP(fp)={mlp_fp['test_metrics']['roc_auc']:.4f}  RF(fp)={rf_fp['test_metrics']['roc_auc']:.4f}")

    return {"dataset": label, "split_seed": seed, "primary_metric": "roc_auc",
            "embeddings": {"mlp": mlp_emb, "rf": rf_emb}, "fingerprints": {"mlp": mlp_fp, "rf": rf_fp}}


def fit_tox21(cache_dir, model_name, seed, tox21_data_dir, tox21_splitter, device):
    emb = _load_embeddings(cache_dir, model_name, "tox21", seed)
    fp = _get_fingerprints("tox21", seed, None, None, None, None, tox21_data_dir, tox21_splitter)
    task_names = fp["_task_names"]

    def _prepare(X, labels, task_index):
        lab = labels[:, task_index]
        finite = np.isfinite(lab)
        X_f, lab_f = X[finite], lab[finite]
        y = lab_f.astype(int)
        binary_mask = np.isin(y, [0, 1])
        return X_f[binary_mask], y[binary_mask]

    seed_for_rf = seed if seed is not None else 0
    per_task = []
    for task_index, task_name in enumerate(task_names):
        emb_train_X, emb_train_y = _prepare(*emb["train"], task_index)
        emb_val_X, emb_val_y = _prepare(*emb["val"], task_index)
        emb_test_X, emb_test_y = _prepare(*emb["test"], task_index)
        fp_train_X, fp_train_y = _prepare(*fp["train"], task_index)
        fp_val_X, fp_val_y = _prepare(*fp["val"], task_index)
        fp_test_X, fp_test_y = _prepare(*fp["test"], task_index)

        if min(len(emb_train_y), len(emb_val_y), len(emb_test_y), len(fp_train_y), len(fp_val_y), len(fp_test_y)) < 10:
            continue
        if any(len(np.unique(y)) < 2 for y in (emb_train_y, emb_val_y, emb_test_y, fp_train_y, fp_val_y, fp_test_y)):
            continue

        mlp_emb = evaluate_mlp_classification(emb_train_X, emb_train_y, emb_val_X, emb_val_y, emb_test_X, emb_test_y, device)
        rf_emb = evaluate_rf_classification(emb_train_X, emb_train_y, emb_val_X, emb_val_y, emb_test_X, emb_test_y, seed=seed_for_rf)
        mlp_fp = evaluate_mlp_classification(fp_train_X, fp_train_y, fp_val_X, fp_val_y, fp_test_X, fp_test_y, device)
        rf_fp = evaluate_rf_classification(fp_train_X, fp_train_y, fp_val_X, fp_val_y, fp_test_X, fp_test_y, seed=seed_for_rf)

        # Keep the full test_metrics dict (roc_auc, f1, mcc, balanced_accuracy) for each
        # variant, not just roc_auc -- matches how LIPO/BACE retain their full metric set
        # rather than a single scalar, so F1/MCC don't have to be thrown away here only
        # to require a rerun later to get them back.
        per_task.append({
            "task": task_name,
            "mlp_emb": mlp_emb["test_metrics"], "rf_emb": rf_emb["test_metrics"],
            "mlp_fp": mlp_fp["test_metrics"], "rf_fp": rf_fp["test_metrics"],
        })

    if not per_task:
        raise RuntimeError("No Tox21 tasks could be evaluated after filtering.")

    def _agg(variant, metric):
        vals = [t[variant][metric] for t in per_task]
        return float(np.mean(vals)), float(np.std(vals))

    def _summary_for(variant):
        summary = {}
        for metric in ("roc_auc", "f1", "mcc"):
            mean, std = _agg(variant, metric)
            summary[f"{metric}_mean_tasks"] = mean
            summary[f"{metric}_std_tasks"] = std
        return summary

    mlp_emb_summary = _summary_for("mlp_emb")
    rf_emb_summary = _summary_for("rf_emb")
    mlp_fp_summary = _summary_for("mlp_fp")
    rf_fp_summary = _summary_for("rf_fp")

    print(f"[Tox21 seed={seed}] mean test over {len(per_task)} tasks  "
          f"MLP(emb) ROC-AUC={mlp_emb_summary['roc_auc_mean_tasks']:.4f} F1={mlp_emb_summary['f1_mean_tasks']:.4f} MCC={mlp_emb_summary['mcc_mean_tasks']:.4f}  "
          f"RF(emb) ROC-AUC={rf_emb_summary['roc_auc_mean_tasks']:.4f} F1={rf_emb_summary['f1_mean_tasks']:.4f} MCC={rf_emb_summary['mcc_mean_tasks']:.4f}")

    return {"dataset": "Tox21", "split_seed": seed, "primary_metric": "roc_auc_mean_tasks", "n_tasks_evaluated": len(per_task),
            "embeddings": {"mlp": {"test_metrics": mlp_emb_summary}, "rf": {"test_metrics": rf_emb_summary}},
            "fingerprints": {"mlp": {"test_metrics": mlp_fp_summary}, "rf": {"test_metrics": rf_fp_summary}}}


def main():
    parser = argparse.ArgumentParser(description="Phase 2: fit MLP/RF probes on cached embeddings + fingerprints.")
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--random-split-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--datasets", type=str, default="lipo,bace,tox21")
    parser.add_argument("--lipo-data-dir", type=str, default="data/MoleculeNet_LIPO_custom")
    parser.add_argument("--lipo-splitter", type=str, default="random")
    parser.add_argument("--bace-data-dir", type=str, default="data/MoleculeNet_BACE_custom")
    parser.add_argument("--bace-splitter", type=str, default="admet_benchmark")
    parser.add_argument("--tox21-data-dir", type=str, default="data/MoleculeNet_Tox21_custom")
    parser.add_argument("--tox21-splitter", type=str, default="random")
    parser.add_argument("--bbb-martins-data-dir", type=str, default="data/TDC_BBB_Martins_custom")
    parser.add_argument("--bbb-martins-splitter", type=str, default="admet_benchmark")
    parser.add_argument("--herg-data-dir", type=str, default="data/TDC_hERG_custom")
    parser.add_argument("--herg-splitter", type=str, default="admet_benchmark")
    parser.add_argument("--ames-data-dir", type=str, default="data/TDC_AMES_custom")
    parser.add_argument("--ames-splitter", type=str, default="admet_benchmark")
    parser.add_argument("--cache-dir", type=str, default="embedding_cache")
    parser.add_argument("--allow-partial-results", action="store_true")
    parser.add_argument("--output", type=str, default="eval_many_models_mlp_rf_results.json")
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
    print(f"Device: {device}")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.random_split_seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    splitters = {
        "lipo": args.lipo_splitter, "bace": args.bace_splitter, "tox21": args.tox21_splitter,
        "bbb_martins": args.bbb_martins_splitter, "herg": args.herg_splitter, "ames": args.ames_splitter,
    }
    tdc_data_dirs = {"bbb_martins": args.bbb_martins_data_dir, "herg": args.herg_data_dir, "ames": args.ames_data_dir}

    all_results = {}
    for model_name in model_names:
        print(f"\n=== Fitting probes for {model_name} ===")
        model_results = {"lipo": [], "tox21": [], "bace": [], "bbb_martins": [], "herg": [], "ames": []}
        failures = []

        for dataset in datasets:
            splitter = splitters[dataset]
            # admet_benchmark ALWAYS uses TDC's official 5 leaderboard seeds (1-5) -- see the
            # matching comment in eval_phase1_extract_embeddings.py's main() for why.
            if splitter == "admet_benchmark":
                dataset_seeds = [1, 2, 3, 4, 5]
            elif splitter == "random":
                dataset_seeds = seeds
            else:
                dataset_seeds = [None]
            for seed in dataset_seeds:
                try:
                    if dataset == "lipo":
                        model_results["lipo"].append(fit_lipo(args.cache_dir, model_name, seed, args.lipo_data_dir, args.lipo_splitter, device))
                    elif dataset == "bace":
                        model_results["bace"].append(fit_bace(args.cache_dir, model_name, seed, args.bace_data_dir, args.bace_splitter, device))
                    elif dataset == "tox21":
                        model_results["tox21"].append(fit_tox21(args.cache_dir, model_name, seed, args.tox21_data_dir, args.tox21_splitter, device))
                    elif dataset in TDC_DATASETS:
                        model_results[dataset].append(
                            fit_tdc_classification(dataset, args.cache_dir, model_name, seed, tdc_data_dirs[dataset], splitter, device)
                        )
                except Exception as exc:
                    msg = f"[{dataset.upper()}][seed={seed}] {model_name}: {exc}"
                    print(f"[FAILED] {msg}")
                    failures.append(msg)

        summary = {}
        if model_results["lipo"]:
            for method in ("mlp", "rf"):
                for feat in ("embeddings", "fingerprints"):
                    for metric in ("r2", "rmse", "mae"):
                        mean, std = _aggregate_seeds(model_results["lipo"], (feat, method, "test_metrics", metric))
                        summary[f"lipo_{feat}_{method}_{metric}"] = {"mean": mean, "std": std}
        if model_results["bace"]:
            for method in ("mlp", "rf"):
                for feat in ("embeddings", "fingerprints"):
                    for metric in ("roc_auc", "f1", "mcc"):
                        mean, std = _aggregate_seeds(model_results["bace"], (feat, method, "test_metrics", metric))
                        summary[f"bace_{feat}_{method}_{metric}"] = {"mean": mean, "std": std}
        if model_results["tox21"]:
            for method in ("mlp", "rf"):
                for feat in ("embeddings", "fingerprints"):
                    for metric in ("roc_auc", "f1", "mcc"):
                        mean, std = _aggregate_seeds(model_results["tox21"], (feat, method, "test_metrics", f"{metric}_mean_tasks"))
                        summary[f"tox21_{feat}_{method}_{metric}"] = {"mean": mean, "std": std}
        for tdc_ds in TDC_DATASETS:
            if model_results[tdc_ds]:
                for method in ("mlp", "rf"):
                    for feat in ("embeddings", "fingerprints"):
                        for metric in ("roc_auc", "f1", "mcc"):
                            mean, std = _aggregate_seeds(model_results[tdc_ds], (feat, method, "test_metrics", metric))
                            summary[f"{tdc_ds}_{feat}_{method}_{metric}"] = {"mean": mean, "std": std}

        model_output_dir = Path(f"models/{model_name}")
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_output_path = model_output_dir / "eval_mlp_rf_results.json"

        # Merge with whatever's already on disk for datasets this run didn't touch --
        # otherwise a partial rerun (e.g. `--datasets tox21` to pick up newly-added
        # metrics) would silently wipe out already-good LIPO/BACE results for the
        # same model, since each dataset's fit is only ever redone deliberately.
        if model_output_path.exists():
            with open(model_output_path) as f:
                existing = json.load(f)
        else:
            existing = {}
        merged_raw = {"lipo": [], "bace": [], "tox21": [], "bbb_martins": [], "herg": [], "ames": [], **existing.get("raw", {})}
        merged_summary = dict(existing.get("summary", {}))
        merged_failures = [msg for msg in existing.get("failures", []) if not any(msg.startswith(f"[{ds.upper()}]") for ds in datasets)]

        for ds in datasets:
            merged_raw[ds] = model_results[ds]
            merged_summary = {k: v for k, v in merged_summary.items() if not k.startswith(f"{ds}_")}
            merged_summary.update({k: v for k, v in summary.items() if k.startswith(f"{ds}_")})
        merged_failures += failures

        model_result = {"summary": merged_summary, "raw": merged_raw, "failures": merged_failures}
        all_results[model_name] = model_result

        print(f"\n--- {model_name} summary ---")
        for key, v in merged_summary.items():
            print(f"  {key}: {v['mean']:.4f} ± {v['std']:.4f}")

        with open(model_output_path, "w") as f:
            json.dump(model_result, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"✓ Saved {model_name}'s results to {model_output_path}")

        if failures and not args.allow_partial_results:
            raise RuntimeError(f"Evaluation completed with {len(failures)} failures for {model_name}. Re-run with --allow-partial-results to keep partial outputs.")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nSaved combined results to {args.output}")

    plotted_models = [m for m in model_names if all_results[m]["summary"]]
    if plotted_models:
        saved_plot_paths = generate_dataset_comparison_plots(all_results, plotted_models)
        for dataset_key, paths in saved_plot_paths.items():
            for path in paths:
                print(f"✓ Saved {dataset_key} comparison plot to {path}")
    else:
        print("⚠ Skipping plot generation -- no model produced a non-empty summary.")


if __name__ == "__main__":
    main()

"""Run MLP-head and Random-Forest downstream evaluations for multiple SSL models.

Same protocol as eval_many_models.py (LIPO, Tox21, BACE; SSL embeddings vs
Morgan fingerprints as the benchmark; scaffold split for BACE, random splits
with multi-seed averaging for LIPO/Tox21) but swaps kNN -> Random Forest and
the linear probe -> a small MLP head (RegressionHead / ClassificationHead
from model/gnn_model.py, trained via gradient descent), on both the frozen
SSL embeddings and the Morgan fingerprints.

Example usage:
    uv run python eval_many_models_mlp_rf.py --models MODEL1,MODEL2 --device cpu
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.knn_bace import (
    build_embedding_features as build_bace_embedding_features,
    build_fingerprint_features as build_bace_fingerprint_features,
    load_bace_admet_benchmark_splits,
    load_bace_splits_from_deepchem,
)
from evaluation.knn_lipo import (
    build_embedding_features as build_lipo_embedding_features,
    build_fingerprint_features as build_lipo_fingerprint_features,
    load_lipo_splits_from_deepchem,
    resolve_checkpoint_path,
    resolve_torch_device,
    infer_graph_featurization,
)
from evaluation.knn_tox21 import (
    build_embedding_features as build_tox21_embedding_features,
    build_fingerprint_features as build_tox21_fingerprint_features,
    load_tox21_splits_from_deepchem,
)
from evaluation.mlp_rf import (
    evaluate_mlp_classification,
    evaluate_mlp_regression,
    evaluate_rf_classification,
    evaluate_rf_regression,
)
from evaluation.tdc_datasets import (
    build_embedding_features as build_tdc_embedding_features,
    build_fingerprint_features as build_tdc_fingerprint_features,
    load_tdc_admet_benchmark_splits,
)
from model.config import ModelConfig
from model.gnn_model import GNNModel

import torch


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint["config"])
    explicit_h, encode_h = infer_graph_featurization(config)
    use_extended_features = bool(getattr(config, "use_extended_features", False))
    scale_eccentricity = bool(getattr(config, "scale_eccentricity", False))
    model = GNNModel.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, explicit_h, encode_h, use_extended_features, scale_eccentricity


def eval_lipo(model_name, checkpoint_path, device, seed, lipo_data_dir, lipo_splitter):
    rows_by_split, split_stats = load_lipo_splits_from_deepchem(lipo_data_dir, lipo_splitter, split_seed=seed)
    resolved_checkpoint = resolve_checkpoint_path(model_name, checkpoint_path)
    model, config, explicit_h, encode_h, use_extended, scale_ecc = load_model(resolved_checkpoint, device)

    emb, fp = {}, {}
    for split in ("train", "val", "test"):
        X, y, _ = build_lipo_embedding_features(rows_by_split[split], model, device, explicit_h, encode_h, use_extended, scale_ecc)
        emb[split] = (X, y)
        Xf, yf, _ = build_lipo_fingerprint_features(rows_by_split[split])
        fp[split] = (Xf, yf)

    seed_for_rf = seed if seed is not None else 0
    mlp_emb = evaluate_mlp_regression(*emb["train"], *emb["val"], *emb["test"], device)
    rf_emb = evaluate_rf_regression(*emb["train"], *emb["val"], *emb["test"], seed=seed_for_rf)
    mlp_fp = evaluate_mlp_regression(*fp["train"], *fp["val"], *fp["test"], device)
    rf_fp = evaluate_rf_regression(*fp["train"], *fp["val"], *fp["test"], seed=seed_for_rf)

    print(
        f"[LIPO seed={seed}] test R2  MLP(emb)={mlp_emb['test_metrics']['r2']:.4f}  "
        f"RF(emb)={rf_emb['test_metrics']['r2']:.4f}  MLP(fp)={mlp_fp['test_metrics']['r2']:.4f}  "
        f"RF(fp)={rf_fp['test_metrics']['r2']:.4f}"
    )

    return {
        "dataset": "LIPO",
        "split_seed": seed,
        "primary_metric": "rmse",
        "embeddings": {"mlp": mlp_emb, "rf": rf_emb},
        "fingerprints": {"mlp": mlp_fp, "rf": rf_fp},
    }


def eval_bace(model_name, checkpoint_path, device, seed, bace_data_dir, bace_splitter):
    if bace_splitter == "admet_benchmark":
        rows_by_split, split_stats = load_bace_admet_benchmark_splits(bace_data_dir, split_seed=seed)
    else:
        rows_by_split, split_stats = load_bace_splits_from_deepchem(bace_data_dir, bace_splitter, split_seed=seed)
    resolved_checkpoint = resolve_checkpoint_path(model_name, checkpoint_path)
    model, config, explicit_h, encode_h, use_extended, scale_ecc = load_model(resolved_checkpoint, device)

    emb, fp = {}, {}
    for split in ("train", "val", "test"):
        X, y, _ = build_bace_embedding_features(rows_by_split[split], model, device, explicit_h, encode_h, use_extended, scale_ecc)
        emb[split] = (X, y)
        Xf, yf, _ = build_bace_fingerprint_features(rows_by_split[split])
        fp[split] = (Xf, yf)

    seed_for_rf = seed if seed is not None else 0
    mlp_emb = evaluate_mlp_classification(*emb["train"], *emb["val"], *emb["test"], device)
    rf_emb = evaluate_rf_classification(*emb["train"], *emb["val"], *emb["test"], seed=seed_for_rf)
    mlp_fp = evaluate_mlp_classification(*fp["train"], *fp["val"], *fp["test"], device)
    rf_fp = evaluate_rf_classification(*fp["train"], *fp["val"], *fp["test"], seed=seed_for_rf)

    print(
        f"[BACE seed={seed}] test ROC-AUC  MLP(emb)={mlp_emb['test_metrics']['roc_auc']:.4f}  "
        f"RF(emb)={rf_emb['test_metrics']['roc_auc']:.4f}  MLP(fp)={mlp_fp['test_metrics']['roc_auc']:.4f}  "
        f"RF(fp)={rf_fp['test_metrics']['roc_auc']:.4f}"
    )

    return {
        "dataset": "BACE",
        "split_seed": seed,
        "primary_metric": "roc_auc",
        "embeddings": {"mlp": mlp_emb, "rf": rf_emb},
        "fingerprints": {"mlp": mlp_fp, "rf": rf_fp},
    }


# TDC ADMET datasets (bbb_martins/herg/ames) -- structurally identical to eval_bace (single-task
# binary classification), so this is one generic function instead of three copy-pastes, unlike
# eval_lipo/eval_bace/eval_tox21 above which each have a genuinely different task shape. Always
# uses the official admet_benchmark protocol (fixed leaderboard test set, seed in 1..5) -- see
# evaluation/tdc_datasets.py -- so results here are leaderboard-comparable, matching the frozen-
# embedding eval's own default.
_TDC_LABELS = {"bbb_martins": "BBB_Martins", "herg": "hERG", "ames": "AMES"}


def eval_tdc_classification(dataset_key, model_name, checkpoint_path, device, seed, tdc_data_dir):
    label = _TDC_LABELS[dataset_key]
    rows_by_split, split_stats = load_tdc_admet_benchmark_splits(dataset_key, tdc_data_dir, split_seed=seed)
    resolved_checkpoint = resolve_checkpoint_path(model_name, checkpoint_path)
    model, config, explicit_h, encode_h, use_extended, scale_ecc = load_model(resolved_checkpoint, device)

    emb, fp = {}, {}
    for split in ("train", "val", "test"):
        X, y, _ = build_tdc_embedding_features(rows_by_split[split], model, device, explicit_h, encode_h, use_extended, scale_ecc)
        emb[split] = (X, y)
        Xf, yf, _ = build_tdc_fingerprint_features(rows_by_split[split])
        fp[split] = (Xf, yf)

    seed_for_rf = seed if seed is not None else 0
    mlp_emb = evaluate_mlp_classification(*emb["train"], *emb["val"], *emb["test"], device)
    rf_emb = evaluate_rf_classification(*emb["train"], *emb["val"], *emb["test"], seed=seed_for_rf)
    mlp_fp = evaluate_mlp_classification(*fp["train"], *fp["val"], *fp["test"], device)
    rf_fp = evaluate_rf_classification(*fp["train"], *fp["val"], *fp["test"], seed=seed_for_rf)

    print(
        f"[{label} seed={seed}] test ROC-AUC  MLP(emb)={mlp_emb['test_metrics']['roc_auc']:.4f}  "
        f"RF(emb)={rf_emb['test_metrics']['roc_auc']:.4f}  MLP(fp)={mlp_fp['test_metrics']['roc_auc']:.4f}  "
        f"RF(fp)={rf_fp['test_metrics']['roc_auc']:.4f}"
    )

    return {
        "dataset": label,
        "split_seed": seed,
        "primary_metric": "roc_auc",
        "embeddings": {"mlp": mlp_emb, "rf": rf_emb},
        "fingerprints": {"mlp": mlp_fp, "rf": rf_fp},
    }


def eval_tox21(model_name, checkpoint_path, device, seed, tox21_data_dir, tox21_splitter):
    data_by_split, split_stats = load_tox21_splits_from_deepchem(tox21_data_dir, tox21_splitter, split_seed=seed)
    resolved_checkpoint = resolve_checkpoint_path(model_name, checkpoint_path)
    model, config, explicit_h, encode_h, use_extended, scale_ecc = load_model(resolved_checkpoint, device)

    emb_split, fp_split = {}, {}
    for split in ("train", "val", "test"):
        smiles = data_by_split[split]["smiles"]
        labels = data_by_split[split]["labels"]
        emb_X, emb_kept, _ = build_tox21_embedding_features(smiles, model, device, explicit_h, encode_h, use_extended, scale_ecc)
        fp_X, fp_kept, _ = build_tox21_fingerprint_features(smiles)
        emb_split[split] = {"X": emb_X, "labels": labels[emb_kept]}
        fp_split[split] = {"X": fp_X, "labels": labels[fp_kept]}

    def _prepare(split_data, task_index):
        labels = split_data["labels"][:, task_index]
        finite_mask = np.isfinite(labels)
        labels = labels[finite_mask]
        X = split_data["X"][finite_mask]
        y = labels.astype(int)
        binary_mask = np.isin(y, [0, 1])
        return X[binary_mask], y[binary_mask]

    seed_for_rf = seed if seed is not None else 0
    per_task = []
    for task_index, task_name in enumerate(split_stats["task_names"]):
        emb_train_X, emb_train_y = _prepare(emb_split["train"], task_index)
        emb_val_X, emb_val_y = _prepare(emb_split["val"], task_index)
        emb_test_X, emb_test_y = _prepare(emb_split["test"], task_index)
        fp_train_X, fp_train_y = _prepare(fp_split["train"], task_index)
        fp_val_X, fp_val_y = _prepare(fp_split["val"], task_index)
        fp_test_X, fp_test_y = _prepare(fp_split["test"], task_index)

        if min(len(emb_train_y), len(emb_val_y), len(emb_test_y), len(fp_train_y), len(fp_val_y), len(fp_test_y)) < 10:
            continue
        if any(len(np.unique(y)) < 2 for y in (emb_train_y, emb_val_y, emb_test_y, fp_train_y, fp_val_y, fp_test_y)):
            continue

        mlp_emb = evaluate_mlp_classification(emb_train_X, emb_train_y, emb_val_X, emb_val_y, emb_test_X, emb_test_y, device)
        rf_emb = evaluate_rf_classification(emb_train_X, emb_train_y, emb_val_X, emb_val_y, emb_test_X, emb_test_y, seed=seed_for_rf)
        mlp_fp = evaluate_mlp_classification(fp_train_X, fp_train_y, fp_val_X, fp_val_y, fp_test_X, fp_test_y, device)
        rf_fp = evaluate_rf_classification(fp_train_X, fp_train_y, fp_val_X, fp_val_y, fp_test_X, fp_test_y, seed=seed_for_rf)

        per_task.append({
            "task": task_name,
            "mlp_emb": mlp_emb["test_metrics"]["roc_auc"],
            "rf_emb": rf_emb["test_metrics"]["roc_auc"],
            "mlp_fp": mlp_fp["test_metrics"]["roc_auc"],
            "rf_fp": rf_fp["test_metrics"]["roc_auc"],
        })

    if not per_task:
        raise RuntimeError("No Tox21 tasks could be evaluated after filtering.")

    def _agg(key):
        vals = [t[key] for t in per_task]
        return float(np.mean(vals)), float(np.std(vals))

    mlp_emb_mean, mlp_emb_std = _agg("mlp_emb")
    rf_emb_mean, rf_emb_std = _agg("rf_emb")
    mlp_fp_mean, mlp_fp_std = _agg("mlp_fp")
    rf_fp_mean, rf_fp_std = _agg("rf_fp")

    print(
        f"[Tox21 seed={seed}] mean test ROC-AUC over {len(per_task)} tasks  "
        f"MLP(emb)={mlp_emb_mean:.4f}±{mlp_emb_std:.4f}  RF(emb)={rf_emb_mean:.4f}±{rf_emb_std:.4f}  "
        f"MLP(fp)={mlp_fp_mean:.4f}±{mlp_fp_std:.4f}  RF(fp)={rf_fp_mean:.4f}±{rf_fp_std:.4f}"
    )

    return {
        "dataset": "Tox21",
        "split_seed": seed,
        "primary_metric": "roc_auc_mean_tasks",
        "n_tasks_evaluated": len(per_task),
        "embeddings": {
            "mlp": {"test_metrics": {"roc_auc_mean_tasks": mlp_emb_mean, "roc_auc_std_tasks": mlp_emb_std}},
            "rf": {"test_metrics": {"roc_auc_mean_tasks": rf_emb_mean, "roc_auc_std_tasks": rf_emb_std}},
        },
        "fingerprints": {
            "mlp": {"test_metrics": {"roc_auc_mean_tasks": mlp_fp_mean, "roc_auc_std_tasks": mlp_fp_std}},
            "rf": {"test_metrics": {"roc_auc_mean_tasks": rf_fp_mean, "roc_auc_std_tasks": rf_fp_std}},
        },
        "per_task": per_task,
    }


def _aggregate_seeds(results, metric_path):
    """metric_path e.g. ('embeddings', 'mlp', 'test_metrics', 'r2')."""
    vals = []
    for r in results:
        node = r
        for key in metric_path:
            node = node[key]
        vals.append(node)
    return float(np.mean(vals)), float(np.std(vals))


# Categorical palette (validated colorblind-safe order -- see the dataviz
# skill). Model bars take slots 1/2/3/4 in the order models were passed on
# the command line; the shared ECFP fingerprint baseline always takes the
# LAST slot used, so it reads as "the constant reference" regardless of how
# many models are being compared.
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SURFACE = "#fcfcfb"
_ECFP_COLOR = "#e34948"  # fixed slot for the fingerprint baseline, distinct from any model slot


def generate_comparison_plot(all_results: dict, model_names: list, output_dir_names: list = None, checkpoint_epochs: dict = None):
    """Grouped bar chart: each model's frozen-embedding MLP/RF score vs. the
    shared ECFP fingerprint baseline, faceted by dataset (LIPO/Tox21/BACE).
    Saved into every model's own directory (models/<name>/) -- called
    automatically at the end of every run, not a separate manual step.
    """
    panels = [
        ("LIPO", "Test R² (higher is better)", "lipo", "r2"),
        ("Tox21", "Mean test ROC-AUC over 12 tasks", "tox21", "roc_auc"),
        ("BACE", "Test ROC-AUC (scaffold split)", "bace", "roc_auc"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), facecolor=_SURFACE)
    fig.suptitle(
        "Frozen-embedding downstream eval vs. ECFP fingerprints",
        fontsize=14, fontweight="bold", color=_INK_PRIMARY, x=0.02, ha="left",
    )
    model_colors = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(model_names)}
    model_labels = {}
    for name in model_names:
        epoch = (checkpoint_epochs or {}).get(name)
        model_labels[name] = f"{name} (epoch {epoch})" if epoch is not None else name

    fig.text(
        0.02, 0.93, f"Models: {', '.join(model_labels[m] for m in model_names)}",
        fontsize=9.5, color=_INK_SECONDARY, ha="left",
    )

    n_bars = len(model_names) + 1  # + ECFP
    bar_width = min(0.32, 0.8 / n_bars)

    for ax, (dataset_label, ylabel, dataset_key, metric_key) in zip(axes, panels):
        ax.set_facecolor(_SURFACE)
        methods = ["mlp", "rf"]
        x = np.arange(len(methods))

        # ECFP baseline is model-independent -- pull it from whichever model has it.
        fp_means, fp_stds = [], []
        for method in methods:
            summary_key = f"{dataset_key}_fingerprints_{method}_{metric_key}"
            entry = next(
                (all_results[m]["summary"][summary_key] for m in model_names if summary_key in all_results[m]["summary"]),
                None,
            )
            fp_means.append(entry["mean"] if entry else np.nan)
            fp_stds.append(entry["std"] if entry else 0.0)

        offset_start = -bar_width * n_bars / 2 + bar_width / 2
        for i, model_name in enumerate(model_names):
            means, stds = [], []
            for method in methods:
                summary_key = f"{dataset_key}_embeddings_{method}_{metric_key}"
                entry = all_results[model_name]["summary"].get(summary_key)
                means.append(entry["mean"] if entry else np.nan)
                stds.append(entry["std"] if entry else 0.0)
            offset = offset_start + i * bar_width
            ax.bar(
                x + offset, means, bar_width, yerr=stds, capsize=3,
                color=model_colors[model_name], label=model_labels[model_name], zorder=3,
                error_kw={"ecolor": _INK_SECONDARY, "elinewidth": 1.0, "capthick": 1.0},
            )

        offset = offset_start + len(model_names) * bar_width
        ax.bar(
            x + offset, fp_means, bar_width, yerr=fp_stds, capsize=3,
            color=_ECFP_COLOR, label="ECFP fingerprints", zorder=3,
            error_kw={"ecolor": _INK_SECONDARY, "elinewidth": 1.0, "capthick": 1.0},
        )

        ax.set_title(dataset_label, fontsize=12, fontweight="bold", color=_INK_PRIMARY, pad=10)
        ax.set_ylabel(ylabel, fontsize=9, color=_INK_SECONDARY)
        ax.set_xticks(x)
        ax.set_xticklabels(["MLP", "RF"], fontsize=10, color=_INK_PRIMARY)
        ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(_BASELINE)
        ax.tick_params(axis="both", colors=_INK_MUTED, labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.97),
        frameon=False, fontsize=9.5, labelcolor=_INK_PRIMARY,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    # Saved into every involved model's own directory, not a shared/general
    # location -- each model's folder keeps a full record of what it was
    # last compared against.
    saved_paths = []
    for name in (output_dir_names or model_names):
        out_dir = Path(f"models/{name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "eval_mlp_rf_vs_ecfp.png"
        fig.savefig(out_path, dpi=200, facecolor=_SURFACE, bbox_inches="tight")
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


# Per-dataset metric sets -- (summary-key metric name, display label, higher-is-better).
_DATASET_METRICS = {
    "lipo": [
        ("r2", "Test R²", True),
        ("rmse", "Test RMSE", False),
        ("mae", "Test MAE", False),
    ],
    "bace": [
        ("roc_auc", "Test ROC-AUC", True),
        ("mcc", "Test MCC", True),
        ("f1", "Test F1 (class of interest)", True),
    ],
    "tox21": [
        ("roc_auc", "Mean test ROC-AUC, 12 tasks", True),
        ("mcc", "Mean test MCC, 12 tasks", True),
        ("f1", "Mean test F1, 12 tasks (class of interest)", True),
    ],
    # bbb_martins/herg/ames: TDC ADMET datasets, added alongside MoleculeNet
    # (moleculenet.org down; TDC's actively-benchmarked leaderboard makes
    # these comparable to more recent 2022-2025 GNN/SSL papers). All three
    # are single-task binary classification, same metric set as BACE.
    "bbb_martins": [
        ("roc_auc", "Test ROC-AUC", True),
        ("mcc", "Test MCC", True),
        ("f1", "Test F1 (class of interest)", True),
    ],
    "herg": [
        ("roc_auc", "Test ROC-AUC", True),
        ("mcc", "Test MCC", True),
        ("f1", "Test F1 (class of interest)", True),
    ],
    "ames": [
        ("roc_auc", "Test ROC-AUC", True),
        ("mcc", "Test MCC", True),
        ("f1", "Test F1 (class of interest)", True),
    ],
}


def generate_dataset_comparison_plots(all_results: dict, model_names: list, output_dir_names: list = None, checkpoint_epochs: dict = None, filename_prefix: str = "eval"):
    """One figure per dataset (LIPO / BACE / Tox21), each with one panel per
    metric in _DATASET_METRICS -- replaces generate_comparison_plot's single
    combined 3-panel (one metric per dataset) figure now that each dataset
    reports 3 metrics instead of 1, which would make a single combined
    figure either cramped (9 panels) or misleading (only showing one metric
    per dataset). Same grouped-bar-chart mechanics and palette as
    generate_comparison_plot, just one dataset's worth of panels per figure
    instead of one panel per dataset.

    filename_prefix defaults to "eval" (-> eval_{dataset}_vs_ecfp.png, the
    frozen-embedding convention). Callers evaluating a DIFFERENT condition in
    the same models/<name>/ directory (e.g. finetune_phase2_eval.py passing
    "finetune_eval") MUST override this -- writing to the frozen filename and
    renaming afterward would briefly overwrite (and, since rename doesn't
    restore it, permanently lose) whatever frozen-eval plot was already
    there.

    Returns {dataset_key: [saved_path, ...]}.
    """
    model_colors = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(model_names)}
    model_labels = {}
    for name in model_names:
        epoch = (checkpoint_epochs or {}).get(name)
        model_labels[name] = f"{name} (epoch {epoch})" if epoch is not None else name

    n_bars = len(model_names) + 1  # + ECFP
    bar_width = min(0.32, 0.8 / n_bars)
    dataset_titles = {
        "lipo": "LIPO", "bace": "BACE", "tox21": "Tox21",
        "bbb_martins": "BBB (TDC, Martins)", "herg": "hERG (TDC)", "ames": "AMES (TDC)",
    }

    all_saved = {}
    for dataset_key, metrics in _DATASET_METRICS.items():
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics) + 1, 5.5), facecolor=_SURFACE)
        fig.suptitle(
            f"{dataset_titles[dataset_key]}: frozen-embedding downstream eval vs. ECFP fingerprints",
            fontsize=14, fontweight="bold", color=_INK_PRIMARY, x=0.02, ha="left",
        )
        fig.text(
            0.02, 0.93, f"Models: {', '.join(model_labels[m] for m in model_names)}",
            fontsize=9.5, color=_INK_SECONDARY, ha="left",
        )

        for ax, (metric_key, metric_label, higher_is_better) in zip(axes, metrics):
            ax.set_facecolor(_SURFACE)
            methods = ["mlp", "rf"]
            x = np.arange(len(methods))

            fp_means, fp_stds = [], []
            for method in methods:
                summary_key = f"{dataset_key}_fingerprints_{method}_{metric_key}"
                entry = next(
                    (all_results[m]["summary"][summary_key] for m in model_names if summary_key in all_results[m]["summary"]),
                    None,
                )
                fp_means.append(entry["mean"] if entry else np.nan)
                fp_stds.append(entry["std"] if entry else 0.0)

            offset_start = -bar_width * n_bars / 2 + bar_width / 2
            for i, model_name in enumerate(model_names):
                means, stds = [], []
                for method in methods:
                    summary_key = f"{dataset_key}_embeddings_{method}_{metric_key}"
                    entry = all_results[model_name]["summary"].get(summary_key)
                    means.append(entry["mean"] if entry else np.nan)
                    stds.append(entry["std"] if entry else 0.0)
                offset = offset_start + i * bar_width
                ax.bar(
                    x + offset, means, bar_width, yerr=stds, capsize=3,
                    color=model_colors[model_name], label=model_labels[model_name], zorder=3,
                    error_kw={"ecolor": _INK_SECONDARY, "elinewidth": 1.0, "capthick": 1.0},
                )

            offset = offset_start + len(model_names) * bar_width
            ax.bar(
                x + offset, fp_means, bar_width, yerr=fp_stds, capsize=3,
                color=_ECFP_COLOR, label="ECFP fingerprints", zorder=3,
                error_kw={"ecolor": _INK_SECONDARY, "elinewidth": 1.0, "capthick": 1.0},
            )

            arrow = "↑ higher is better" if higher_is_better else "↓ lower is better"
            ax.set_ylabel(f"{metric_label} ({arrow})", fontsize=9.5, color=_INK_PRIMARY)
            ax.set_xticks(x)
            ax.set_xticklabels(["MLP", "RF"], fontsize=10, color=_INK_PRIMARY)
            ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color(_BASELINE)
            ax.tick_params(axis="both", colors=_INK_MUTED, labelsize=9)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.97),
            frameon=False, fontsize=9.5, labelcolor=_INK_PRIMARY,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.88])

        saved_paths = []
        for name in (output_dir_names or model_names):
            out_dir = Path(f"models/{name}")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{filename_prefix}_{dataset_key}_vs_ecfp.png"
            fig.savefig(out_path, dpi=200, facecolor=_SURFACE, bbox_inches="tight")
            saved_paths.append(out_path)
        plt.close(fig)
        all_saved[dataset_key] = saved_paths

    return all_saved


def main():
    parser = argparse.ArgumentParser(description="Run MLP-head and RF evals (LIPO, Tox21, BACE) for multiple SSL models.")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--lipo-data-dir", type=str, default="data/MoleculeNet_LIPO_custom")
    parser.add_argument("--lipo-splitter", type=str, default="random")
    parser.add_argument("--tox21-data-dir", type=str, default="data/MoleculeNet_Tox21_custom")
    parser.add_argument("--tox21-splitter", type=str, default="random")
    parser.add_argument("--bace-data-dir", type=str, default="data/MoleculeNet_BACE_custom")
    parser.add_argument("--bace-splitter", type=str, default="scaffold")
    parser.add_argument("--random-split-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--allow-partial-results", action="store_true")
    parser.add_argument("--output", type=str, default="eval_many_models_mlp_rf_results.json")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.random_split_seeds.split(",") if s.strip()]
    device = resolve_torch_device(args.device)
    print(f"Device: {device}")

    all_results = {}
    checkpoint_epochs = {}
    for model_name in model_names:
        print(f"\n=== Evaluating {model_name} ===")
        checkpoint_path = args.checkpoint_path
        if not checkpoint_path and args.checkpoint_name:
            checkpoint_path = str(Path(f"models/{model_name}/checkpoints") / args.checkpoint_name)

        # Which checkpoint actually gets used, and which epoch it's from --
        # checkpoint selection (best SSL val loss vs. best online eval vs. an
        # arbitrary epoch snapshot) materially changes results, so record it
        # rather than leaving it implicit in the checkpoint filename alone.
        resolved_checkpoint = checkpoint_path or resolve_checkpoint_path(model_name, None)
        try:
            ckpt_meta = torch.load(resolved_checkpoint, map_location="cpu", weights_only=False)
            checkpoint_epochs[model_name] = ckpt_meta.get("epoch")
            del ckpt_meta
        except Exception as exc:
            print(f"  (could not read checkpoint epoch from {resolved_checkpoint}: {exc})")
            checkpoint_epochs[model_name] = None
        print(f"  Checkpoint: {resolved_checkpoint} (epoch {checkpoint_epochs[model_name]})")

        model_results = {"lipo": [], "tox21": [], "bace": []}
        failures = []

        lipo_seeds = seeds if args.lipo_splitter == "random" else [None]
        for seed in lipo_seeds:
            try:
                model_results["lipo"].append(eval_lipo(model_name, checkpoint_path, device, seed, args.lipo_data_dir, args.lipo_splitter))
            except Exception as exc:
                print(f"[FAILED][LIPO][seed={seed}] {model_name}: {exc}")
                failures.append(f"LIPO seed={seed}: {exc}")

        tox21_seeds = seeds if args.tox21_splitter == "random" else [None]
        for seed in tox21_seeds:
            try:
                model_results["tox21"].append(eval_tox21(model_name, checkpoint_path, device, seed, args.tox21_data_dir, args.tox21_splitter))
            except Exception as exc:
                print(f"[FAILED][Tox21][seed={seed}] {model_name}: {exc}")
                failures.append(f"Tox21 seed={seed}: {exc}")

        bace_seeds = seeds if args.bace_splitter == "random" else [None]
        for seed in bace_seeds:
            try:
                model_results["bace"].append(eval_bace(model_name, checkpoint_path, device, seed, args.bace_data_dir, args.bace_splitter))
            except Exception as exc:
                print(f"[FAILED][BACE][seed={seed}] {model_name}: {exc}")
                failures.append(f"BACE seed={seed}: {exc}")

        summary = {}
        if model_results["lipo"]:
            for method in ("mlp", "rf"):
                for feat in ("embeddings", "fingerprints"):
                    mean, std = _aggregate_seeds(model_results["lipo"], (feat, method, "test_metrics", "r2"))
                    summary[f"lipo_{feat}_{method}_r2"] = {"mean": mean, "std": std}
        if model_results["bace"]:
            for method in ("mlp", "rf"):
                for feat in ("embeddings", "fingerprints"):
                    mean, std = _aggregate_seeds(model_results["bace"], (feat, method, "test_metrics", "roc_auc"))
                    summary[f"bace_{feat}_{method}_roc_auc"] = {"mean": mean, "std": std}
        if model_results["tox21"]:
            for method in ("mlp", "rf"):
                for feat in ("embeddings", "fingerprints"):
                    mean, std = _aggregate_seeds(model_results["tox21"], (feat, method, "test_metrics", "roc_auc_mean_tasks"))
                    summary[f"tox21_{feat}_{method}_roc_auc"] = {"mean": mean, "std": std}

        model_result = {
            "summary": summary,
            "raw": model_results,
            "failures": failures,
            "checkpoint_path": resolved_checkpoint,
            "checkpoint_epoch": checkpoint_epochs[model_name],
        }
        all_results[model_name] = model_result

        print(f"\n--- {model_name} summary ---")
        for key, v in summary.items():
            print(f"  {key}: {v['mean']:.4f} ± {v['std']:.4f}")

        # Always saved into the model's own directory -- not just the shared
        # --output file below, which is easy to overwrite/lose track of and
        # isn't tied to any particular model. This is the durable record.
        model_output_dir = Path(f"models/{model_name}")
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_output_path = model_output_dir / "eval_mlp_rf_results.json"
        with open(model_output_path, "w") as f:
            json.dump(model_result, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"✓ Saved {model_name}'s results to {model_output_path}")

        if failures and not args.allow_partial_results:
            raise RuntimeError(
                f"Evaluation completed with {len(failures)} failures for {model_name}. "
                "Re-run with --allow-partial-results to keep partial outputs without failing."
            )

    # Combined multi-model file, kept only as a convenience for cross-model
    # comparison scripts/plots -- the per-model files above are the source
    # of truth and are what should be relied on for any single model.
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nSaved combined results to {args.output}")

    # Plot generation is automatic, not a separate manual step -- saved into
    # every involved model's own directory (see generate_comparison_plot).
    plotted_models = [m for m in model_names if all_results[m]["summary"]]
    if plotted_models:
        saved_plot_paths = generate_comparison_plot(all_results, plotted_models, checkpoint_epochs=checkpoint_epochs)
        for path in saved_plot_paths:
            print(f"✓ Saved comparison plot to {path}")
    else:
        print("⚠ Skipping plot generation -- no model produced a non-empty summary.")


if __name__ == "__main__":
    main()

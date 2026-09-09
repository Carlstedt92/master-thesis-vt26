"""Finetune GNN encoders on labeled tasks, then evaluate learned embeddings.

This follows the cleaner task-specific setup used in `knn_eval_lip_finetune.py`:
- load SSL checkpoint encoder weights
- replace the SSL projection head with a task head
- finetune encoder + task head on labeled train data with validation early stopping
- freeze the finetuned encoder and evaluate embeddings with kNN, linear probe, and random forest

It runs on all three datasets:
- LIPO: regression
- BACE: binary classification
- Tox21: multi-task classification

Example:
    python finetune_eval_many_models.py --models model1,model2 --finetune-epochs 30 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from evaluation.linear_probe import evaluate_linear_probe_classification, evaluate_linear_probe_regression
from rf_eval import evaluate_rf_classification, evaluate_rf_regression
from plotting.knn_summary import (
    update_random_seed_summary,
    finalize_random_seed_summary,
    save_random_seed_artifacts,
)
from plotting.linear_probe_summary import (
    update_linear_probe_summary,
    finalize_linear_probe_summary,
    save_linear_probe_artifacts,
)
from plotting.rf_boxplots import (
    update_random_seed_summary as update_rf_random_seed_summary,
    finalize_random_seed_summary as finalize_rf_random_seed_summary,
    save_random_seed_artifacts as save_rf_random_seed_artifacts,
)
from plotting.finetune_comparison_summary import save_comparison_artifacts
from datahandling.graph_creation import smiles_to_pygdata
from model.config import ModelConfig
from model.gnn_model import ClassificationHead, GNNModel, RegressionHead


FP_RADIUS = 2
FP_NBITS = 2048
DEFAULT_K_VALUES = [3, 5, 11, 21, 31, 41, 51]
DEFAULT_ALPHAS = [0.01, 0.1, 1.0, 10.0]
DEFAULT_CS = [0.01, 0.1, 1.0, 10.0]
DEFAULT_N_ESTIMATORS = [50, 100, 200, 300]
DEFAULT_MAX_DEPTHS = [10, 20, 30, None]

SUMMARY_METRIC_BY_DATASET = {
    "LIPO": "rmse",
    "BACE": "roc_auc",
    "Tox21": "roc_auc_mean_tasks",
}


def parse_csv(values: str, cast):
    return [cast(value.strip()) for value in values.split(",") if value.strip()]


def parse_max_depths(values: str):
    parsed = []
    for value in values.split(","):
        value = value.strip()
        if not value:
            continue
        if value.lower() == "none":
            parsed.append(None)
        else:
            parsed.append(int(value))
    return parsed


def resolve_checkpoint_path(model_name: str, checkpoint_path: str | None, checkpoint_name: str | None) -> str:
    if checkpoint_path:
        return checkpoint_path
    if checkpoint_name:
        return str(Path(f"models/{model_name}/checkpoints") / checkpoint_name)

    # best_model.pth (SSL-val-loss-selected) is the default everywhere else
    # this session (evaluation/knn_bace.py's own resolve_checkpoint_path uses
    # the same priority) -- this used to try best_online_eval_model.pth
    # first, the opposite order, which was a real footgun.
    best_model = Path(f"models/{model_name}/checkpoints/best_model.pth")
    if best_model.exists():
        return str(best_model)
    return str(Path(f"models/{model_name}/checkpoints/best_online_eval_model.pth"))


def load_checkpoint_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint["config"])
    model = GNNModel.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def load_model_for_finetuning(checkpoint_path: str, device: torch.device, dataset: str):
    model, config = load_checkpoint_model(checkpoint_path, device)
    if dataset == "lipo":
        model.head = RegressionHead(input_dim=config.hidden_dim, hidden_dim=max(64, config.hidden_dim // 2), output_dim=1).to(device)
    elif dataset == "bace":
        model.head = ClassificationHead(input_dim=config.hidden_dim, hidden_dim=max(64, config.hidden_dim // 2), output_dim=1).to(device)
    elif dataset == "tox21":
        # Replaced later with a multi-task linear head once task count is known.
        model.head = nn.Identity()
    elif dataset in ("bbb_martins", "herg", "ames"):
        # TDC single-task binary classification -- structurally identical to bace.
        model.head = ClassificationHead(input_dim=config.hidden_dim, hidden_dim=max(64, config.hidden_dim // 2), output_dim=1).to(device)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for param in model.parameters():
        param.requires_grad = True

    return model, config


def rows_to_graphs(rows, explicit_hydrogens, encode_hydrogen_count, target_is_vector: bool = False):
    graphs = []
    labels = []
    invalid = 0

    for item in rows:
        if target_is_vector:
            smiles, target = item
            target_value = np.asarray(target)
        else:
            smiles, target = item
            target_value = float(target)

        data = smiles_to_pygdata(
            smiles,
            explicit_hydrogens=explicit_hydrogens,
            encode_hydrogen_count=encode_hydrogen_count,
        )
        if data is None or data.num_nodes == 0:
            invalid += 1
            continue

        data.y = torch.tensor(target_value, dtype=torch.float32)
        graphs.append(data)
        labels.append(target_value)

    return graphs, np.asarray(labels), invalid


def build_embedding_features(rows, model, device, explicit_hydrogens, encode_hydrogen_count, target_is_vector: bool = False):
    features = []
    labels = []
    invalid = 0

    model.eval()
    with torch.no_grad():
        for item in rows:
            if target_is_vector:
                smiles, target = item
                target_value = np.asarray(target)
            else:
                smiles, target = item
                target_value = float(target)

            data = smiles_to_pygdata(
                smiles,
                explicit_hydrogens=explicit_hydrogens,
                encode_hydrogen_count=encode_hydrogen_count,
            )
            if data is None or data.num_nodes == 0:
                invalid += 1
                continue

            data = data.to(device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            emb = model.get_embeddings(data.x, data.edge_index, data.edge_attr, batch).squeeze(0)
            features.append(emb.cpu().numpy())
            labels.append(target_value)

    return np.asarray(features), np.asarray(labels), invalid


def build_fingerprint_features(rows, target_is_vector: bool = False, radius: int = FP_RADIUS, nbits: int = FP_NBITS):
    features = []
    labels = []
    invalid = 0
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    for item in rows:
        if target_is_vector:
            smiles, target = item
            target_value = np.asarray(target)
        else:
            smiles, target = item
            target_value = float(target)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid += 1
            continue

        bitvect = morgan_generator.GetFingerprint(mol)
        arr = np.zeros((nbits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bitvect, arr)
        features.append(arr)
        labels.append(target_value)

    return np.asarray(features), np.asarray(labels), invalid


def make_graph_loader(rows, explicit_hydrogens, encode_hydrogen_count, batch_size, target_is_vector: bool = False):
    graphs, labels, invalid = rows_to_graphs(rows, explicit_hydrogens, encode_hydrogen_count, target_is_vector=target_is_vector)
    if len(graphs) == 0:
        raise RuntimeError("No valid graphs found for finetuning.")
    for data, label in zip(graphs, labels):
        data.y = torch.tensor(label, dtype=torch.float32)
    return DataLoader(graphs, batch_size=batch_size, shuffle=True), labels, invalid, graphs


def evaluate_regression_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_classification_metrics(y_true, y_proba):
    y_true_int = np.asarray(y_true).astype(int)
    pred_labels = (np.asarray(y_proba) >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true_int, y_proba)),
        # pos_label=1 = class of interest (active/toxic) -- see evaluation/mlp_rf.py's comment
        # for the full justification; same convention applied here for consistency.
        "f1": float(f1_score(y_true_int, pred_labels)),
        "mcc": float(matthews_corrcoef(y_true_int, pred_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_int, pred_labels)),
    }


def _collect_test_predictions(model, loader, dataset: str, device):
    """Run the (already best-epoch-loaded) model over a test loader and
    collect predictions -- same per-dataset branching as finetune_model's own
    validation loop, factored out so it can be reused for a genuine
    end-to-end test evaluation of the ACTUAL finetuned head, not a re-probed
    embedding. For tox21 (multi-task), returns per-task lists; otherwise
    returns flat (y_true, y_score) arrays."""
    model.eval()
    if dataset == "tox21":
        scores_per_task, labels_per_task = None, None
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                labels = batch.y.float()
                if out.dim() >= 2 and labels.dim() == 1 and labels.numel() == out.numel():
                    labels = labels.view_as(out)
                mask = ~torch.isnan(labels)
                probs = torch.sigmoid(out)
                n_tasks = probs.shape[1] if probs.dim() > 1 else 1
                if scores_per_task is None:
                    scores_per_task = [[] for _ in range(n_tasks)]
                    labels_per_task = [[] for _ in range(n_tasks)]
                for t in range(n_tasks):
                    cond = mask[:, t]
                    if cond.sum() > 0:
                        scores_per_task[t].extend(probs[cond, t].cpu().numpy().tolist())
                        labels_per_task[t].extend(labels[cond, t].cpu().numpy().tolist())
        return scores_per_task, labels_per_task

    y_true, y_score = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
            labels = batch.y.view(-1).float()
            if dataset != "lipo":
                out = torch.sigmoid(out)
            y_score.extend(out.cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_score)


def evaluate_finetuned_head_on_test(model, test_loader, dataset: str, device):
    """Genuine end-to-end test evaluation of the ACTUAL finetuned task head --
    the model passed in must already have its best-val-epoch state loaded
    (finetune_model does this before returning). This is deliberately kept
    separate from -- and reported alongside, not instead of -- the
    re-probed-embeddings numbers phase 2 computes: the two answer different
    questions ("how good is the finetuned model itself" vs. "did finetuning
    change the encoder's general representation quality"), and conflating
    them was the actual bug -- phase 1 was discarding the trained head before
    this was ever measured.
    """
    if dataset == "tox21":
        scores_per_task, labels_per_task = _collect_test_predictions(model, test_loader, dataset, device)
        per_task_metrics = []
        for scores, labs in zip(scores_per_task, labels_per_task):
            if len(np.unique(labs)) < 2:
                continue
            per_task_metrics.append(evaluate_classification_metrics(labs, scores))
        if not per_task_metrics:
            raise RuntimeError("No Tox21 test tasks had both classes present -- cannot evaluate finetuned head.")
        agg = {}
        for metric in ("roc_auc", "f1", "mcc", "balanced_accuracy"):
            vals = [m[metric] for m in per_task_metrics]
            agg[f"{metric}_mean_tasks"] = float(np.mean(vals))
            agg[f"{metric}_std_tasks"] = float(np.std(vals))
        return agg

    y_true, y_score = _collect_test_predictions(model, test_loader, dataset, device)
    if dataset == "lipo":
        return evaluate_regression_metrics(y_true, y_score)
    return evaluate_classification_metrics(y_true, y_score)


def make_summary_result(result, metric_type: str):
    """Convert a finetune result into the shape expected by the plotting helpers."""
    dataset_name = result["dataset"]
    primary_metric = SUMMARY_METRIC_BY_DATASET[dataset_name]

    if metric_type == "knn":
        if dataset_name == "Tox21":
            emb_value = float(result["embeddings"]["mean_test_roc_auc"]["knn"])
            fp_value = float(result["fingerprints"]["mean_test_roc_auc"]["knn"])
        else:
            emb_value = float(result["embeddings"]["knn"]["test_metrics"][primary_metric])
            fp_value = float(result["fingerprints"]["knn"]["test_metrics"][primary_metric])
    elif metric_type == "linear_probe":
        if dataset_name == "Tox21":
            emb_value = float(result["embeddings"]["mean_test_roc_auc"]["linear_probe"])
            fp_value = float(result["fingerprints"]["mean_test_roc_auc"]["linear_probe"])
        else:
            emb_value = float(result["embeddings"]["linear_probe"]["test_metrics"][primary_metric])
            fp_value = float(result["fingerprints"]["linear_probe"]["test_metrics"][primary_metric])
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")

    return {
        "dataset": dataset_name,
        "primary_metric": primary_metric,
        "split_seed": result.get("seed"),
        "embeddings": {"test_metrics": {primary_metric: emb_value}},
        "fingerprints": {"test_metrics": {primary_metric: fp_value}},
    }


def make_rf_summary_result(result):
    """Convert a finetune result into the shape expected by rf_boxplots helpers."""
    dataset_name = result["dataset"]
    primary_metric = SUMMARY_METRIC_BY_DATASET[dataset_name]

    if dataset_name == "Tox21":
        emb_value = float(result["embeddings"]["mean_test_roc_auc"]["random_forest"])
        fp_value = float(result["fingerprints"]["mean_test_roc_auc"]["random_forest"])
    else:
        emb_value = float(result["embeddings"]["random_forest"]["test_metrics"][primary_metric])
        fp_value = float(result["fingerprints"]["random_forest"]["test_metrics"][primary_metric])

    return {
        "dataset": dataset_name,
        "primary_metric": primary_metric,
        "split_seed": result.get("seed"),
        "embeddings": {"test_metrics": {primary_metric: emb_value}},
        "fingerprints": {"test_metrics": {primary_metric: fp_value}},
    }


def tune_knn_regression(X_train, y_train, X_val, y_val, X_test, y_test, k_values):
    best = None
    best_val_rmse = float("inf")
    best_val_r2 = -float("inf")

    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
        knn.fit(X_train, y_train)
        val_pred = knn.predict(X_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        val_r2 = float(r2_score(y_val, val_pred))
        if val_r2 > best_val_r2 or (np.isclose(val_r2, best_val_r2) and val_rmse < best_val_rmse):
            best = k
            best_val_r2 = val_r2
            best_val_rmse = val_rmse

    knn = KNeighborsRegressor(n_neighbors=best, weights="distance")
    knn.fit(X_train, y_train)
    test_pred = knn.predict(X_test)
    return {
        "best_k": int(best),
        "validation_metrics": evaluate_regression_metrics(y_val, KNeighborsRegressor(n_neighbors=best, weights="distance").fit(X_train, y_train).predict(X_val)),
        "test_metrics": evaluate_regression_metrics(y_test, test_pred),
    }


def tune_knn_classification(X_train, y_train, X_val, y_val, X_test, y_test, k_values):
    best = None
    best_val_roc = -float("inf")
    best_val_f1 = -float("inf")
    eps = 1e-3

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn.fit(X_train, y_train)
        val_proba = knn.predict_proba(X_val)[:, 1]
        val_pred = knn.predict(X_val)
        val_roc = float(roc_auc_score(y_val, val_proba))
        val_f1 = float(f1_score(y_val, val_pred))

        if val_roc > best_val_roc + eps:
            best = k
            best_val_roc = val_roc
            best_val_f1 = val_f1
        elif abs(val_roc - best_val_roc) <= eps and val_f1 > best_val_f1 + eps:
            best = k
            best_val_f1 = val_f1

    knn = KNeighborsClassifier(n_neighbors=best, weights="distance")
    knn.fit(X_train, y_train)
    test_proba = knn.predict_proba(X_test)[:, 1]
    return {
        "best_k": int(best),
        "validation_metrics": evaluate_classification_metrics(y_val, KNeighborsClassifier(n_neighbors=best, weights="distance").fit(X_train, y_train).predict_proba(X_val)[:, 1]),
        "test_metrics": evaluate_classification_metrics(y_test, test_proba),
    }


def finetune_model(model, train_loader, val_loader, device, dataset: str, epochs: int, learning_rate: float, weight_decay: float, test_loader=None, scheduler_type: str = "constant"):
    # Use loss minimization for regression (LIPO) and ROC AUC maximization for
    # classification tasks (BACE, Tox21). Training still uses BCEWithLogitsLoss
    # for classification but we prefer ROC AUC on the validation set to choose
    # the best checkpoint.
    if dataset == "lipo":
        loss_fn = nn.MSELoss()
        # lower validation loss is better
        better = lambda current, best: current < best
        best_val = float("inf")
        select_by = "loss"
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        # higher ROC AUC is better
        better = lambda current, best: current > best
        best_val = -float("inf")
        select_by = "roc_auc"

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type != "constant":
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            if dataset == "tox21":
                labels = batch.y.float()
                # Some dataloaders may produce a flattened label vector
                # (shape [batch_size * num_tasks]) instead of [batch_size, num_tasks].
                # If that happens, reshape labels to match `out` so boolean
                # masking works as expected.
                if out.dim() >= 2 and labels.dim() == 1 and labels.numel() == out.numel():
                    labels = labels.view_as(out)

                mask = ~torch.isnan(labels)
                if mask.sum() == 0:
                    continue
                loss = loss_fn(out[mask], labels[mask])
            else:
                labels = batch.y.view(-1).float()
                loss = loss_fn(out.squeeze(-1), labels)

            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_losses = []
        # For classification we collect outputs+labels to compute ROC AUC.
        val_scores_per_task = None
        val_labels_per_task = None
        val_scores_all = []
        val_labels_all = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                if dataset == "tox21":
                    labels = batch.y.float()
                    if out.dim() >= 2 and labels.dim() == 1 and labels.numel() == out.numel():
                        labels = labels.view_as(out)

                    mask = ~torch.isnan(labels)
                    if mask.sum() == 0:
                        continue
                    loss = loss_fn(out[mask], labels[mask])

                    if select_by == "roc_auc":
                        probs = torch.sigmoid(out)
                        n_tasks = probs.shape[1] if probs.dim() > 1 else 1
                        if val_scores_per_task is None:
                            val_scores_per_task = [[] for _ in range(n_tasks)]
                            val_labels_per_task = [[] for _ in range(n_tasks)]
                        for t in range(n_tasks):
                            cond = mask[:, t]
                            if cond.sum() > 0:
                                vals = probs[cond, t].cpu().numpy().tolist()
                                labs = labels[cond, t].cpu().numpy().tolist()
                                val_scores_per_task[t].extend(vals)
                                val_labels_per_task[t].extend(labs)
                else:
                    labels = batch.y.view(-1).float()
                    loss = loss_fn(out.squeeze(-1), labels)

                    if select_by == "roc_auc":
                        probs = torch.sigmoid(out.squeeze(-1))
                        val_scores_all.extend(probs.cpu().numpy().tolist())
                        val_labels_all.extend(labels.cpu().numpy().tolist())

                val_losses.append(float(loss.item()))

        avg_train = float(np.mean(train_losses)) if train_losses else float("inf")
        avg_val = float(np.mean(val_losses)) if val_losses else float("inf")

        # Compute selection metric (loss for regression, ROC AUC for classification)
        if select_by == "loss":
            selection_metric = avg_val
            metric_name = "val_loss"
        else:
            # classification: compute ROC AUC
            if dataset == "tox21":
                per_task_rocs = []
                if val_scores_per_task is not None:
                    for scores, labs in zip(val_scores_per_task, val_labels_per_task):
                        try:
                            if len(np.unique(labs)) >= 2:
                                per_task_rocs.append(float(roc_auc_score(labs, scores)))
                        except Exception:
                            continue
                selection_metric = float(np.mean(per_task_rocs)) if per_task_rocs else -float("inf")
            else:
                try:
                    if len(np.unique(val_labels_all)) >= 2:
                        selection_metric = float(roc_auc_score(val_labels_all, val_scores_all))
                    else:
                        selection_metric = -float("inf")
                except Exception:
                    selection_metric = -float("inf")
            metric_name = "val_roc_auc"

        print(f"  Epoch {epoch:03d}/{epochs} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f} | {metric_name}={selection_metric:.4f}")

        if better(selection_metric, best_val):
            best_val = selection_metric
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch

    model.load_state_dict(best_state)
    model.eval()

    result = {"best_epoch": int(best_epoch), "best_val_loss": float(best_val), "epochs_ran": int(epochs)}
    if test_loader is not None:
        # The actual point of finetuning: evaluate the real, trained task head end-to-end on
        # test, instead of only ever reporting a re-probed-embeddings number (see
        # evaluate_finetuned_head_on_test's docstring for why that distinction matters).
        result["test_metrics"] = evaluate_finetuned_head_on_test(model, test_loader, dataset, device)
    return result


def get_dataset_splits(dataset: str, random_seed: int | None, lipo_data_dir: str, tox21_data_dir: str, bace_data_dir: str):
    if dataset == "lipo":
        from evaluation.knn_lipo import load_lipo_splits_from_deepchem
        return load_lipo_splits_from_deepchem(lipo_data_dir, "random", split_seed=random_seed)[0]
    if dataset == "bace":
        from evaluation.knn_bace import load_bace_splits_from_deepchem
        return load_bace_splits_from_deepchem(bace_data_dir, "scaffold", split_seed=random_seed)[0]
    if dataset == "tox21":
        from evaluation.knn_tox21 import load_tox21_splits_from_deepchem
        return load_tox21_splits_from_deepchem(tox21_data_dir, "random", split_seed=random_seed)[0]
    raise ValueError(f"Unsupported dataset: {dataset}")


def finetune_and_evaluate_lipo(
    model_name,
    checkpoint_path,
    device,
    seed,
    finetune_epochs,
    batch_size,
    lr,
    weight_decay,
    k_values,
    alphas,
    data_dir,
    n_estimators_list,
    max_depth_list,
):
    from evaluation.knn_lipo import infer_graph_featurization, load_lipo_splits_from_deepchem

    splits, stats = load_lipo_splits_from_deepchem(data_dir, "random", split_seed=seed)
    rows_train, rows_val, rows_test = splits["train"], splits["val"], splits["test"]

    checkpoint = resolve_checkpoint_path(model_name, checkpoint_path, None)
    model, config = load_model_for_finetuning(checkpoint, device, "lipo")
    explicit_h, encode_h = infer_graph_featurization(config)

    train_loader, _, train_invalid, _ = make_graph_loader(rows_train, explicit_h, encode_h, batch_size)
    val_loader, _, val_invalid, _ = make_graph_loader(rows_val, explicit_h, encode_h, batch_size)
    ft_info = finetune_model(model, train_loader, val_loader, device, "lipo", finetune_epochs, lr, weight_decay)

    X_train, y_train, emb_inv_train = build_embedding_features(rows_train, model, device, explicit_h, encode_h)
    X_val, y_val, emb_inv_val = build_embedding_features(rows_val, model, device, explicit_h, encode_h)
    X_test, y_test, emb_inv_test = build_embedding_features(rows_test, model, device, explicit_h, encode_h)

    fp_train, fp_y_train, fp_inv_train = build_fingerprint_features(rows_train)
    fp_val, fp_y_val, fp_inv_val = build_fingerprint_features(rows_val)
    fp_test, fp_y_test, fp_inv_test = build_fingerprint_features(rows_test)

    emb_scaler = StandardScaler()
    X_train = emb_scaler.fit_transform(X_train)
    X_val = emb_scaler.transform(X_val)
    X_test = emb_scaler.transform(X_test)

    fp_scaler = StandardScaler()
    fp_train = fp_scaler.fit_transform(fp_train)
    fp_val = fp_scaler.transform(fp_val)
    fp_test = fp_scaler.transform(fp_test)

    emb_knn = tune_knn_regression(X_train, y_train, X_val, y_val, X_test, y_test, k_values)
    fp_knn = tune_knn_regression(fp_train, fp_y_train, fp_val, fp_y_val, fp_test, fp_y_test, k_values)

    emb_lp = evaluate_linear_probe_regression(X_train, y_train, X_val, y_val, X_test, y_test, alphas)
    fp_lp = evaluate_linear_probe_regression(fp_train, fp_y_train, fp_val, fp_y_val, fp_test, fp_y_test, alphas)

    emb_rf = evaluate_rf_regression(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        n_estimators_list,
        max_depth_list,
    )

    fp_rf = evaluate_rf_regression(
        fp_train,
        fp_y_train,
        fp_val,
        fp_y_val,
        fp_test,
        fp_y_test,
        n_estimators_list,
        max_depth_list,
    )

    return {
        "dataset": "LIPO",
        "splitter": "random",
        "seed": seed,
        "split_stats": stats,
        "finetune": {**ft_info, "n_invalid_graphs_train": int(train_invalid), "n_invalid_graphs_val": int(val_invalid)},
        "embeddings": {"knn": emb_knn, "linear_probe": emb_lp, "random_forest": emb_rf},
        "fingerprints": {"knn": fp_knn, "linear_probe": fp_lp, "random_forest": fp_rf},
        "invalid_smiles": {"embeddings_train": int(emb_inv_train), "embeddings_val": int(emb_inv_val), "embeddings_test": int(emb_inv_test), "fingerprints_train": int(fp_inv_train), "fingerprints_val": int(fp_inv_val), "fingerprints_test": int(fp_inv_test)},
    }


def _binary_split_rows(rows):
    return rows


def finetune_and_evaluate_bace(
    model_name,
    checkpoint_path,
    device,
    seed,
    finetune_epochs,
    batch_size,
    lr,
    weight_decay,
    k_values,
    Cs,
    data_dir,
    n_estimators_list,
    max_depth_list,
):
    from evaluation.knn_bace import infer_graph_featurization, load_bace_splits_from_deepchem

    splits, stats = load_bace_splits_from_deepchem(data_dir, "scaffold", split_seed=seed)
    rows_train, rows_val, rows_test = splits["train"], splits["val"], splits["test"]

    checkpoint = resolve_checkpoint_path(model_name, checkpoint_path, None)
    model, config = load_model_for_finetuning(checkpoint, device, "bace")
    explicit_h, encode_h = infer_graph_featurization(config)

    train_loader, _, train_invalid, _ = make_graph_loader(rows_train, explicit_h, encode_h, batch_size)
    val_loader, _, val_invalid, _ = make_graph_loader(rows_val, explicit_h, encode_h, batch_size)
    ft_info = finetune_model(model, train_loader, val_loader, device, "bace", finetune_epochs, lr, weight_decay)

    X_train, y_train, emb_inv_train = build_embedding_features(rows_train, model, device, explicit_h, encode_h)
    X_val, y_val, emb_inv_val = build_embedding_features(rows_val, model, device, explicit_h, encode_h)
    X_test, y_test, emb_inv_test = build_embedding_features(rows_test, model, device, explicit_h, encode_h)

    fp_train, fp_y_train, fp_inv_train = build_fingerprint_features(rows_train)
    fp_val, fp_y_val, fp_inv_val = build_fingerprint_features(rows_val)
    fp_test, fp_y_test, fp_inv_test = build_fingerprint_features(rows_test)

    emb_scaler = StandardScaler()
    X_train = emb_scaler.fit_transform(X_train)
    X_val = emb_scaler.transform(X_val)
    X_test = emb_scaler.transform(X_test)

    fp_scaler = StandardScaler()
    fp_train = fp_scaler.fit_transform(fp_train)
    fp_val = fp_scaler.transform(fp_val)
    fp_test = fp_scaler.transform(fp_test)

    emb_knn = tune_knn_classification(X_train, y_train.astype(int), X_val, y_val.astype(int), X_test, y_test.astype(int), k_values)
    fp_knn = tune_knn_classification(fp_train, fp_y_train.astype(int), fp_val, fp_y_val.astype(int), fp_test, fp_y_test.astype(int), k_values)

    emb_lp = evaluate_linear_probe_classification(X_train, y_train.astype(int), X_val, y_val.astype(int), X_test, y_test.astype(int), Cs)
    fp_lp = evaluate_linear_probe_classification(fp_train, fp_y_train.astype(int), fp_val, fp_y_val.astype(int), fp_test, fp_y_test.astype(int), Cs)

    emb_rf = evaluate_rf_classification(
        X_train,
        y_train.astype(int),
        X_val,
        y_val.astype(int),
        X_test,
        y_test.astype(int),
        n_estimators_list,
        max_depth_list,
    )

    fp_rf = evaluate_rf_classification(
        fp_train,
        fp_y_train.astype(int),
        fp_val,
        fp_y_val.astype(int),
        fp_test,
        fp_y_test.astype(int),
        n_estimators_list,
        max_depth_list,
    )

    return {
        "dataset": "BACE",
        "splitter": "scaffold",
        "seed": seed,
        "split_stats": stats,
        "finetune": {**ft_info, "n_invalid_graphs_train": int(train_invalid), "n_invalid_graphs_val": int(val_invalid)},
        "embeddings": {"knn": emb_knn, "linear_probe": emb_lp, "random_forest": emb_rf},
        "fingerprints": {"knn": fp_knn, "linear_probe": fp_lp, "random_forest": fp_rf},
        "invalid_smiles": {"embeddings_train": int(emb_inv_train), "embeddings_val": int(emb_inv_val), "embeddings_test": int(emb_inv_test), "fingerprints_train": int(fp_inv_train), "fingerprints_val": int(fp_inv_val), "fingerprints_test": int(fp_inv_test)},
    }


def finetune_and_evaluate_tox21(
    model_name,
    checkpoint_path,
    device,
    seed,
    finetune_epochs,
    batch_size,
    lr,
    weight_decay,
    k_values,
    Cs,
    data_dir,
    n_estimators_list,
    max_depth_list,
):
    from evaluation.knn_tox21 import infer_graph_featurization, load_tox21_splits_from_deepchem

    splits, stats = load_tox21_splits_from_deepchem(data_dir, "random", split_seed=seed)
    checkpoint = resolve_checkpoint_path(model_name, checkpoint_path, None)
    model, config = load_model_for_finetuning(checkpoint, device, "tox21")
    explicit_h, encode_h = infer_graph_featurization(config)

    labels_train = splits["train"]["labels"]
    labels_val = splits["val"]["labels"]
    labels_test = splits["test"]["labels"]
    num_tasks = labels_train.shape[1]

    def build_multi(rows, labels_arr):
        data_list = []
        lab_list = []
        invalid = 0
        for i, smiles in enumerate(rows):
            data = smiles_to_pygdata(str(smiles), explicit_hydrogens=explicit_h, encode_hydrogen_count=encode_h)
            if data is None or data.num_nodes == 0:
                invalid += 1
                continue
            data.y = torch.tensor(labels_arr[i], dtype=torch.float32)
            data_list.append(data)
            lab_list.append(labels_arr[i])
        if len(data_list) == 0:
            raise RuntimeError("No valid Tox21 graphs after conversion.")
        return DataLoader(data_list, batch_size=batch_size, shuffle=True), np.asarray(lab_list), invalid, data_list

    def build_embedding_with_indices(smiles_rows):
        features = []
        kept_indices = []
        invalid = 0

        model.eval()
        with torch.no_grad():
            for row_idx, smiles in enumerate(smiles_rows):
                data = smiles_to_pygdata(str(smiles), explicit_hydrogens=explicit_h, encode_hydrogen_count=encode_h)
                if data is None or data.num_nodes == 0:
                    invalid += 1
                    continue

                data = data.to(device)
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                emb = model.get_embeddings(data.x, data.edge_index, data.edge_attr, batch).squeeze(0)
                features.append(emb.cpu().numpy())
                kept_indices.append(row_idx)

        return np.asarray(features), np.asarray(kept_indices), invalid

    def build_fingerprint_with_indices(smiles_rows):
        features = []
        kept_indices = []
        invalid = 0
        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)

        for row_idx, smiles in enumerate(smiles_rows):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                invalid += 1
                continue

            bitvect = morgan_generator.GetFingerprint(mol)
            arr = np.zeros((FP_NBITS,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bitvect, arr)
            features.append(arr)
            kept_indices.append(row_idx)

        return np.asarray(features), np.asarray(kept_indices), invalid

    train_loader, train_labels, train_invalid, train_graphs = build_multi(splits["train"]["smiles"], labels_train)
    val_loader, val_labels, val_invalid, val_graphs = build_multi(splits["val"]["smiles"], labels_val)

    # Replace the head with a per-task linear head for supervised finetuning.
    model.head = nn.Linear(config.hidden_dim, num_tasks).to(device)
    for param in model.parameters():
        param.requires_grad = True

    ft_info = finetune_model(model, train_loader, val_loader, device, "tox21", finetune_epochs, lr, weight_decay)

    emb_train, idx_train, emb_inv_train = build_embedding_with_indices(splits["train"]["smiles"])
    emb_val, idx_val, emb_inv_val = build_embedding_with_indices(splits["val"]["smiles"])
    emb_test, idx_test, emb_inv_test = build_embedding_with_indices(splits["test"]["smiles"])

    fp_train, fp_idx_train, fp_inv_train = build_fingerprint_with_indices(splits["train"]["smiles"])
    fp_val, fp_idx_val, fp_inv_val = build_fingerprint_with_indices(splits["val"]["smiles"])
    fp_test, fp_idx_test, fp_inv_test = build_fingerprint_with_indices(splits["test"]["smiles"])

    def evaluate_multitask_feature_set(X_train, idx_train, X_val, idx_val, X_test, idx_test, labels_train, labels_val, labels_test):
        per_task = []
        for t in range(num_tasks):
            def pick(X, idxs, labels):
                lab = labels[:, t]
                finite = np.isfinite(lab)
                keep = np.isin(np.arange(len(lab)), idxs) & finite
                X = X[keep]
                y = lab[keep].astype(int)
                binary_mask = np.isin(y, [0, 1])
                return X[binary_mask], y[binary_mask]

            Xtr_raw, ytr = pick(X_train, idx_train, labels_train)
            Xv_raw, yv = pick(X_val, idx_val, labels_val)
            Xt_raw, yt = pick(X_test, idx_test, labels_test)

            if len(np.unique(ytr)) < 2 or len(np.unique(yv)) < 2 or len(np.unique(yt)) < 2:
                continue

            # Per-task scaling: fit on the task's training fold and transform
            # validation/test so preprocessing matches the standalone evaluator.
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr_raw)
            Xv = scaler.transform(Xv_raw)
            Xt = scaler.transform(Xt_raw)

            knn = tune_knn_classification(Xtr, ytr, Xv, yv, Xt, yt, k_values)
            lp = evaluate_linear_probe_classification(Xtr_raw, ytr, Xv_raw, yv, Xt_raw, yt, Cs)
            rf = evaluate_rf_classification(
                Xtr_raw,
                ytr,
                Xv_raw,
                yv,
                Xt_raw,
                yt,
                n_estimators_list,
                max_depth_list,
            )

            per_task.append({"knn": knn, "linear_probe": lp, "random_forest": rf})

        knn_scores = [entry["knn"]["test_metrics"]["roc_auc"] for entry in per_task]
        lp_scores = [entry["linear_probe"]["test_metrics"]["roc_auc"] for entry in per_task]
        rf_scores = [entry["random_forest"]["test_metrics"]["roc_auc"] for entry in per_task]

        return {
            "mean_test_roc_auc": {
                "knn": float(np.mean(knn_scores)) if knn_scores else None,
                "linear_probe": float(np.mean(lp_scores)) if lp_scores else None,
                "random_forest": float(np.mean(rf_scores)) if rf_scores else None,
            },
            "per_task": per_task,
        }

    # For Tox21 multi-task evaluation we must scale features per-task (fit on
    # that task's training split) to match the standalone evaluator. Do not
    # apply a global scaler across tasks — instead perform scaling inside the
    # per-task evaluation below.
    emb_result = evaluate_multitask_feature_set(emb_train, idx_train, emb_val, idx_val, emb_test, idx_test, labels_train, labels_val, labels_test)
    fp_result = evaluate_multitask_feature_set(fp_train, fp_idx_train, fp_val, fp_idx_val, fp_test, fp_idx_test, labels_train, labels_val, labels_test)

    return {
        "dataset": "Tox21",
        "splitter": "random",
        "seed": seed,
        "split_stats": stats,
        "finetune": {**ft_info, "n_invalid_graphs_train": int(train_invalid), "n_invalid_graphs_val": int(val_invalid)},
        "embeddings": emb_result,
        "fingerprints": fp_result,
        "invalid_smiles": {"embeddings_train": int(emb_inv_train), "embeddings_val": int(emb_inv_val), "embeddings_test": int(emb_inv_test), "fingerprints_train": int(fp_inv_train), "fingerprints_val": int(fp_inv_val), "fingerprints_test": int(fp_inv_test)},
    }


def main():
    parser = argparse.ArgumentParser(description="Finetune SSL encoders on labeled tasks, then evaluate embeddings.")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Optional explicit checkpoint path for all models")
    parser.add_argument("--checkpoint-name", type=str, default=None, help="Optional checkpoint filename under models/<model>/checkpoints/")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-weight-decay", type=float, default=1e-5)
    parser.add_argument("--random-split-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--k-values", type=str, default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--alphas", type=str, default=",".join(str(alpha) for alpha in DEFAULT_ALPHAS))
    parser.add_argument("--Cs", type=str, default=",".join(str(c) for c in DEFAULT_CS))
    parser.add_argument("--n-estimators", type=str, default=",".join(str(n) for n in DEFAULT_N_ESTIMATORS))
    parser.add_argument("--max-depths", type=str, default=",".join("None" if depth is None else str(depth) for depth in DEFAULT_MAX_DEPTHS))
    parser.add_argument("--lipo-data-dir", type=str, default="data/MoleculeNet_LIPO_custom")
    parser.add_argument("--tox21-data-dir", type=str, default="data/MoleculeNet_Tox21_custom")
    parser.add_argument("--bace-data-dir", type=str, default="data/MoleculeNet_BACE_custom")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    if not model_names:
        raise ValueError("No model names provided via --models")

    seeds = parse_csv(args.random_split_seeds, int)
    k_values = parse_csv(args.k_values, int)
    alphas = parse_csv(args.alphas, float)
    Cs = parse_csv(args.Cs, float)
    n_estimators_list = parse_csv(args.n_estimators, int)
    max_depth_list = parse_max_depths(args.max_depths)

    for model_name in model_names:
        print(f"\n{'=' * 80}")
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"{'=' * 80}")

        model_out_dir = Path(f"models/{model_name}")
        model_out_dir.mkdir(parents=True, exist_ok=True)
        finetune_log_dir = model_out_dir / "finetune_eval_logs"
        finetune_log_dir.mkdir(parents=True, exist_ok=True)

        all_results = {"LIPO": [], "BACE": [], "Tox21": []}
        knn_seed_summary = {}
        linear_probe_seed_summary = {}
        rf_seed_summary = {}

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            for dataset in ["LIPO", "BACE", "Tox21"]:
                try:
                    if dataset == "LIPO":
                        print("Finetuning and evaluating LIPO...")
                        result = finetune_and_evaluate_lipo(
                            model_name,
                            args.checkpoint_path,
                            device,
                            seed,
                            args.finetune_epochs,
                            args.batch_size,
                            args.finetune_lr,
                            args.finetune_weight_decay,
                            k_values,
                            alphas,
                            args.lipo_data_dir,
                            n_estimators_list,
                            max_depth_list,
                        )
                    elif dataset == "BACE":
                        print("Finetuning and evaluating BACE...")
                        result = finetune_and_evaluate_bace(
                            model_name,
                            args.checkpoint_path,
                            device,
                            seed,
                            args.finetune_epochs,
                            args.batch_size,
                            args.finetune_lr,
                            args.finetune_weight_decay,
                            k_values,
                            Cs,
                            args.bace_data_dir,
                            n_estimators_list,
                            max_depth_list,
                        )
                    else:
                        print("Finetuning and evaluating Tox21...")
                        result = finetune_and_evaluate_tox21(
                            model_name,
                            args.checkpoint_path,
                            device,
                            seed,
                            args.finetune_epochs,
                            args.batch_size,
                            args.finetune_lr,
                            args.finetune_weight_decay,
                            k_values,
                            Cs,
                            args.tox21_data_dir,
                            n_estimators_list,
                            max_depth_list,
                        )

                    all_results[dataset].append(result)

                    update_random_seed_summary(knn_seed_summary, make_summary_result(result, "knn"))
                    update_linear_probe_summary(linear_probe_seed_summary, make_summary_result(result, "linear_probe"))
                    update_rf_random_seed_summary(rf_seed_summary, make_rf_summary_result(result))

                    out_file = finetune_log_dir / f"finetune_eval_{dataset.lower()}_seed{seed}.json"
                    with open(out_file, "w") as f:
                        json.dump(result, f, indent=2, default=lambda obj: obj.tolist() if hasattr(obj, "tolist") else str(obj))
                    print(f"Saved: {out_file}")
                except Exception as exc:
                    print(f"  ERROR in {dataset} seed {seed}: {exc}")

        summary_path = finetune_log_dir / "finetune_eval_many_models_results.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=lambda obj: obj.tolist() if hasattr(obj, "tolist") else str(obj))
        print(f"\nSummary saved to {summary_path}")

        finalized_knn_summary = finalize_random_seed_summary(knn_seed_summary)
        finalized_lp_summary = finalize_linear_probe_summary(linear_probe_seed_summary)
        finalized_rf_summary = finalize_rf_random_seed_summary(rf_seed_summary)
        finetune_artifact_subdir = f"{model_name}/finetune_eval_logs"
        if finalized_knn_summary:
            save_random_seed_artifacts(finetune_artifact_subdir, finalized_knn_summary)
        if finalized_lp_summary:
            save_linear_probe_artifacts(finetune_artifact_subdir, finalized_lp_summary)
        if finalized_rf_summary:
            save_rf_random_seed_artifacts(finetune_artifact_subdir, finalized_rf_summary)

        save_comparison_artifacts(model_name)


if __name__ == "__main__":
    main()

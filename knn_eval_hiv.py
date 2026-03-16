"""
KNN evaluation for SSL encoder embeddings on MoleculeNet HIV.
This script loads a pretrained encoder, extracts one embedding per molecule,
trains a KNN classifier on a train split, and reports binary classification metrics.
Metrics include ROC-AUC, Precision-Recall AUC, F1 score, and Balanced Accuracy.
ROC and Precision-Recall curves are saved to disk for visualization.
"""

import numpy as np
import json
import torch
import deepchem as dc
from pathlib import Path
from datetime import datetime
from model.gine_model import GINEModel
from model.config import ModelConfig
from datahandling.graph_creation import smiles_to_pygdata
from plotting.knn_eval_plots import plot_roc_curve, plot_precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
)


HIV_SPLITTER = "scaffold"  # MoleculeNet standard split for fair benchmark comparison.
HIV_DATA_DIR = "data/MoleculeNet_HIV_custom"


def load_hiv_splits_from_deepchem(data_dir: str, splitter: str):
    """Load HIV via DeepChem and return split-wise SMILES/labels with basic stats."""
    tasks, datasets, _ = dc.molnet.load_hiv(
        featurizer=dc.feat.RawFeaturizer(),
        splitter=splitter,
        transformers=[],
        reload=True,
        data_dir=data_dir,
        save_dir=data_dir,
    )

    train_ds, val_ds, test_ds = datasets
    split_map = {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    }

    rows_by_split = {}
    stats = {
        "task": tasks[0] if tasks else "HIV_active",
        "splitter": splitter,
    }

    for split_name, split_ds in split_map.items():
        labels = split_ds.y.reshape(-1)
        ids = split_ds.ids
        rows = []
        skipped_non_finite = 0
        skipped_non_binary = 0

        for smiles, label in zip(ids, labels):
            if not np.isfinite(label):
                skipped_non_finite += 1
                continue

            y = int(label)
            if y not in (0, 1):
                skipped_non_binary += 1
                continue

            rows.append((str(smiles), y))

        rows_by_split[split_name] = rows
        stats[f"n_{split_name}_deepchem"] = int(len(labels))
        stats[f"n_{split_name}_usable_labels"] = int(len(rows))
        stats[f"n_{split_name}_skipped_non_finite"] = int(skipped_non_finite)
        stats[f"n_{split_name}_skipped_non_binary"] = int(skipped_non_binary)

    return rows_by_split, stats


rows_by_split, split_stats = load_hiv_splits_from_deepchem(
    data_dir=HIV_DATA_DIR,
    splitter=HIV_SPLITTER,
)

# Initialize model using checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssl_model_name = "GINO_DINO_DELANEY"
checkpoint = torch.load(
    f"./models/{ssl_model_name}/checkpoints/best_model.pth",
    map_location=device,
    weights_only=False,
)
config = ModelConfig.from_dict(checkpoint["config"])

model = GINEModel.from_config(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def build_embeddings(rows, model, device):
    """Convert split rows into embeddings and labels, skipping invalid graphs."""
    features = []
    labels = []
    invalid_smiles = 0

    with torch.no_grad():
        for smiles, target in rows:
            data = smiles_to_pygdata(smiles)
            if data is None:
                invalid_smiles += 1
                continue

            if data.num_nodes == 0:
                invalid_smiles += 1
                continue

            data = data.to(device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            embedding = model.get_embeddings(data.x, data.edge_index, data.edge_attr, batch).squeeze(0)

            features.append(embedding.cpu().numpy())
            labels.append(int(target))

    X = np.asarray(features)
    y = np.asarray(labels)
    return X, y, invalid_smiles


X_train, y_train, invalid_train = build_embeddings(rows_by_split["train"], model, device)
X_val, y_val, invalid_val = build_embeddings(rows_by_split["val"], model, device)
X_test, y_test, invalid_test = build_embeddings(rows_by_split["test"], model, device)

if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
    raise RuntimeError(
        "One of the DeepChem HIV splits produced no valid samples after graph conversion. "
        f"Stats: {split_stats}. "
        f"Invalid SMILES train/val/test: {invalid_train}/{invalid_val}/{invalid_test}."
    )

if len(np.unique(y_train)) < 2:
    raise RuntimeError("Training split has fewer than 2 classes after filtering.")

if len(np.unique(y_val)) < 2:
    raise RuntimeError("Validation split has fewer than 2 classes after filtering.")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Tune K on official validation split
k_values = [3, 5, 11, 21, 31, 41, 51]
best_k = None
best_val_auc = -np.inf
best_val_pr_auc = -np.inf
best_val_f1 = -np.inf
eps = 1e-3

for k in k_values:
    knn_val = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn_val.fit(X_train, y_train)
    val_proba = knn_val.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    val_pr_auc = average_precision_score(y_val, val_proba)
    val_f1 = f1_score(y_val, knn_val.predict(X_val))

    print(f"Validation (k={k}) -> ROC-AUC: {val_auc:.4f}, PR-AUC: {val_pr_auc:.4f}, F1: {val_f1:.4f}")
    if val_pr_auc > best_val_pr_auc + eps:
        best_val_pr_auc = val_pr_auc
        best_val_auc = val_auc
        best_val_f1 = val_f1
        best_k = k
    elif abs(val_pr_auc - best_val_pr_auc) <= eps:
        if val_auc > best_val_auc + eps:
            best_val_auc = val_auc
            best_val_f1 = val_f1
            best_k = k
        elif abs(val_auc - best_val_auc) <= eps and val_f1 > best_val_f1 + eps:
            best_val_f1 = val_f1
            best_k = k

# Final KNN evaluation on official test split
knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]

print(f"DeepChem splitter: {split_stats['splitter']}")
print(
    f"DeepChem split sizes (train/val/test): "
    f"{split_stats['n_train_deepchem']}/{split_stats['n_val_deepchem']}/{split_stats['n_test_deepchem']}"
)
print(f"Samples used (train/val/test): {len(y_train)}/{len(y_val)}/{len(y_test)}")
print(f"Invalid SMILES skipped (train/val/test): {invalid_train}/{invalid_val}/{invalid_test}")
print(
    f"Best k (validation): {best_k}  |  "
    f"Val ROC-AUC: {best_val_auc:.4f} | Val PR-AUC: {best_val_pr_auc:.4f} | Val F1: {best_val_f1:.4f}"
)
test_roc_auc = roc_auc_score(y_test, y_proba)
test_pr_auc = average_precision_score(y_test, y_proba)
test_f1 = f1_score(y_test, y_pred)
test_balanced_acc = balanced_accuracy_score(y_test, y_pred)

print(f"ROC-AUC: {test_roc_auc:.4f}")
print(f"PR-AUC: {test_pr_auc:.4f}")
print(f"F1: {test_f1:.4f}")
print(f"Balanced Accuracy: {test_balanced_acc:.4f}")

# Plot ROC and PR curves
roc_path = f"models/{ssl_model_name}/roc_curve_hiv_knn.png"
pr_path = f"models/{ssl_model_name}/pr_curve_hiv_knn.png"
plot_roc_curve(y_test, y_proba, roc_path, model_name=ssl_model_name)
plot_precision_recall_curve(y_test, y_proba, pr_path, model_name=ssl_model_name)

# Save KNN evaluation results to metadata.json
metadata_path = f"models/{ssl_model_name}/metadata.json"
existing_metadata = {}
if Path(metadata_path).exists():
    with open(metadata_path, 'r') as f:
        existing_metadata = json.load(f)

knn_eval_data = {
    "dataset": "HIV",
    "data_source": "deepchem.molnet.load_hiv",
    "splitter": split_stats["splitter"],
    "task": split_stats["task"],
    "evaluation_method": "KNN (k-Nearest Neighbors) with distance weights",
    "best_k": best_k,
    "validation_metrics": {
        "roc_auc": round(best_val_auc, 4),
        "pr_auc": round(best_val_pr_auc, 4),
        "f1": round(best_val_f1, 4),
    },
    "test_metrics": {
        "roc_auc": round(test_roc_auc, 4),
        "pr_auc": round(test_pr_auc, 4),
        "f1": round(test_f1, 4),
        "balanced_accuracy": round(test_balanced_acc, 4),
    },
    "n_deepchem_train": split_stats["n_train_deepchem"],
    "n_deepchem_val": split_stats["n_val_deepchem"],
    "n_deepchem_test": split_stats["n_test_deepchem"],
    "n_train": int(len(y_train)),
    "n_val": int(len(y_val)),
    "n_test": int(len(y_test)),
    "n_invalid_smiles_train": int(invalid_train),
    "n_invalid_smiles_val": int(invalid_val),
    "n_invalid_smiles_test": int(invalid_test),
    "timestamp": datetime.now().isoformat(),
}

existing_metadata["KNN_eval"] = knn_eval_data

with open(metadata_path, 'w') as f:
    json.dump(existing_metadata, f, indent=2)

print(f"\nSaved ROC curve: {roc_path}")
print(f"Saved PR curve: {pr_path}")
print(f"Updated metadata: {metadata_path}")
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
from pathlib import Path
from datetime import datetime
from model.gine_model import GINEModel
from model.config import ModelConfig
from datahandling.graph_creation import smiles_to_pygdata
from plotting.knn_eval_plots import plot_roc_curve, plot_precision_recall_curve
from torch_geometric.datasets import MoleculeNet
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
)


# Load dataset (single-task binary classification) using custom featurization
dataset = MoleculeNet(
    root="data/MoleculeNet_HIV_custom",
    name="HIV",
    from_smiles=smiles_to_pygdata,
)

# Initialize model using checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssl_model_name = "GINE_DINO"
checkpoint = torch.load(
    f"./models/{ssl_model_name}/checkpoints/best_model.pth",
    map_location=device,
    weights_only=False,
)
config = ModelConfig.from_dict(checkpoint["config"])

model = GINEModel.from_config(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Extract embeddings and labels
features = []
labels = []

with torch.no_grad():
    for data in dataset:
        target = data.y.view(-1)[0]
        if not torch.isfinite(target):
            continue

        if data.num_nodes == 0:
            continue

        data = data.to(device)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        embedding = model.get_embeddings(data.x, data.edge_index, data.edge_attr, batch).squeeze(0)

        features.append(embedding.cpu().numpy())
        labels.append(int(target.item()))

X = np.asarray(features)
y = np.asarray(labels)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tune K with stratified cross-validation on training split
k_values = [3, 5, 11, 21, 31, 41, 51]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_k = None
best_cv_auc = -np.inf
best_cv_pr_auc = -np.inf
best_cv_f1 = -np.inf
eps = 1e-3

for k in k_values:
    fold_aucs = []
    fold_pr_aucs = []
    fold_f1s = []
    for tr_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        knn_cv = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn_cv.fit(X_tr, y_tr)
        val_proba = knn_cv.predict_proba(X_val)[:, 1]
        fold_aucs.append(roc_auc_score(y_val, val_proba))
        fold_pr_aucs.append(average_precision_score(y_val, val_proba))
        fold_f1s.append(f1_score(y_val, knn_cv.predict(X_val)))

    mean_auc = float(np.mean(fold_aucs))
    mean_pr_auc = float(np.mean(fold_pr_aucs))
    mean_f1 = float(np.mean(fold_f1s))
    print(f"CV ROC-AUC (k={k}): {mean_auc:.4f}, PR-AUC: {mean_pr_auc:.4f}, F1: {mean_f1:.4f}")
    if mean_pr_auc > best_cv_pr_auc + eps:
        best_cv_pr_auc = mean_pr_auc
        best_cv_auc = mean_auc
        best_cv_f1 = mean_f1
        best_k = k
    elif abs(mean_pr_auc - best_cv_pr_auc) <= eps:
        if mean_auc > best_cv_auc + eps:
            best_cv_auc = mean_auc
            best_cv_f1 = mean_f1
            best_k = k
        elif abs(mean_auc - best_cv_auc) <= eps and mean_f1 > best_cv_f1 + eps:
            best_cv_f1 = mean_f1
            best_k = k

# Final KNN evaluation on held-out test split
knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]

print(f"Samples used: {len(y)}")
print(f"Best k (CV): {best_k}  |  CV ROC-AUC: {best_cv_auc:.4f} |  CV PR-AUC: {best_cv_pr_auc:.4f} | CV F1: {best_cv_f1:.4f}")
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
    "evaluation_method": "KNN (k-Nearest Neighbors) with distance weights",
    "cv_folds": 5,
    "best_k": best_k,
    "cv_metrics": {
        "roc_auc": round(best_cv_auc, 4),
        "pr_auc": round(best_cv_pr_auc, 4),
        "f1": round(best_cv_f1, 4),
    },
    "test_metrics": {
        "roc_auc": round(test_roc_auc, 4),
        "pr_auc": round(test_pr_auc, 4),
        "f1": round(test_f1, 4),
        "balanced_accuracy": round(test_balanced_acc, 4),
    },
    "n_samples": len(y),
    "timestamp": datetime.now().isoformat(),
}

existing_metadata["KNN_eval"] = knn_eval_data

with open(metadata_path, 'w') as f:
    json.dump(existing_metadata, f, indent=2)

print(f"\nSaved ROC curve: {roc_path}")
print(f"Saved PR curve: {pr_path}")
print(f"Updated metadata: {metadata_path}")
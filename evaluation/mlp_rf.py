"""MLP-head and Random Forest probes for frozen embeddings (or fingerprints).

Mirrors evaluation/linear_probe.py's interface (train on frozen features,
tune on validation, report test metrics) but swaps Ridge/LogisticRegression
for a small MLP head (RegressionHead/ClassificationHead from
model/gnn_model.py, trained via gradient descent) and adds a Random Forest
variant using the same train/val/test splits.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from model.gnn_model import ClassificationHead, RegressionHead


def _scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


# f1_score's default (pos_label=1) is used deliberately, not just left at its
# default unexamined: verified against the actual data that label 1 is "the
# class of interest" for both classification datasets this pipeline
# evaluates -- BACE's Class=1 has mean pIC50=7.64 vs Class=0's 5.58 (active
# inhibitor), and Tox21's label 1 is the toxic/positive assay result
# (standard MoleculeNet convention). Note this is NOT the same thing as "the
# minority class" -- BACE's test split actually has MORE actives than
# inactives (92 vs 60), so pos_label=1 there is scoring the majority class,
# which is correct here because it's the class of interest, not because
# it's rare.


def evaluate_mlp_regression(
    X_train, y_train, X_val, y_val, X_test, y_test,
    device, hidden_dim: int | None = None, epochs: int = 150, lr: float = 1e-3, weight_decay: float = 1e-4,
):
    """Train RegressionHead on frozen features via gradient descent; report val/test metrics.

    Selects the best epoch by validation MSE (lower is better) rather than
    just running the fixed epoch budget and reporting whatever state that
    happens to land on -- matches finetune_model()'s own convention
    (finetune_eval_many_models.py) for the exact same reason: a fixed budget
    with no selection lets a probe that overfits early get worse on test for
    no principled reason, and there was no good justification for the frozen
    probe being the one place in the project that didn't do this.
    """
    X_train_s, X_val_s, X_test_s = _scale_features(X_train, X_val, X_test)
    input_dim = X_train.shape[1]
    if hidden_dim is None:
        hidden_dim = max(32, input_dim // 2)

    y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-8)

    head = RegressionHead(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    y_train_t = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32, device=device).unsqueeze(-1)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32, device=device)
    y_val_t = torch.tensor((y_val - y_mean) / y_std, dtype=torch.float32, device=device).unsqueeze(-1)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    best_state = deepcopy(head.state_dict())
    for _ in range(epochs):
        head.train()
        optimizer.zero_grad()
        loss = loss_fn(head(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(head(X_val_t), y_val_t))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        val_pred = head(X_val_t).squeeze(-1).cpu().numpy() * y_std + y_mean
        test_pred = head(X_test_t).squeeze(-1).cpu().numpy() * y_std + y_mean

    return {
        "validation_metrics": {
            "r2": float(r2_score(y_val, val_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
            "mae": float(mean_absolute_error(y_val, val_pred)),
        },
        "test_metrics": {
            "r2": float(r2_score(y_test, test_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
            "mae": float(mean_absolute_error(y_test, test_pred)),
        },
        "test_predictions": test_pred,
    }


def evaluate_mlp_classification(
    X_train, y_train, X_val, y_val, X_test, y_test,
    device, hidden_dim: int | None = None, epochs: int = 150, lr: float = 1e-3, weight_decay: float = 1e-4,
):
    """Train ClassificationHead on frozen features via gradient descent; report val/test metrics.

    Selects the best epoch by validation ROC-AUC (higher is better) -- see
    evaluate_mlp_regression's docstring for the full rationale; same fix,
    mirrored for classification the way finetune_model() selects by val
    ROC-AUC for every non-regression dataset.
    """
    X_train_s, X_val_s, X_test_s = _scale_features(X_train, X_val, X_test)
    input_dim = X_train.shape[1]
    if hidden_dim is None:
        hidden_dim = max(32, input_dim // 2)

    head = ClassificationHead(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(-1)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32, device=device)
    val_has_both_classes = len(np.unique(y_val)) >= 2

    best_val_roc_auc = -float("inf")
    best_state = deepcopy(head.state_dict())
    for _ in range(epochs):
        head.train()
        optimizer.zero_grad()
        loss = loss_fn(head(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        if val_has_both_classes:
            head.eval()
            with torch.no_grad():
                val_proba_epoch = torch.sigmoid(head(X_val_t)).squeeze(-1).cpu().numpy()
            val_roc_auc = float(roc_auc_score(y_val, val_proba_epoch))
            if val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                best_state = deepcopy(head.state_dict())
        else:
            # Can't rank epochs without both classes in val -- fall back to "last epoch wins",
            # the old unconditional behavior, only for this edge case.
            best_state = deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        val_proba = torch.sigmoid(head(X_val_t)).squeeze(-1).cpu().numpy()
        test_proba = torch.sigmoid(head(X_test_t)).squeeze(-1).cpu().numpy()
    val_pred = (val_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    return {
        "validation_metrics": {
            "roc_auc": float(roc_auc_score(y_val, val_proba)),
            "f1": float(f1_score(y_val, val_pred)),
        },
        "test_metrics": {
            "roc_auc": float(roc_auc_score(y_test, test_proba)),
            "f1": float(f1_score(y_test, test_pred)),  # pos_label=1 = class of interest (active/toxic) -- verified, see comment above
            "mcc": float(matthews_corrcoef(y_test, test_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        },
        "test_probabilities": test_proba,
    }


def evaluate_rf_regression(X_train, y_train, X_val, y_val, X_test, y_test, seed: int = 0, n_estimators: int = 300):
    X_train_s, X_val_s, X_test_s = _scale_features(X_train, X_val, X_test)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    val_pred = rf.predict(X_val_s)
    test_pred = rf.predict(X_test_s)

    return {
        "validation_metrics": {
            "r2": float(r2_score(y_val, val_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
            "mae": float(mean_absolute_error(y_val, val_pred)),
        },
        "test_metrics": {
            "r2": float(r2_score(y_test, test_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
            "mae": float(mean_absolute_error(y_test, test_pred)),
        },
        "test_predictions": test_pred,
    }


def evaluate_rf_classification(X_train, y_train, X_val, y_val, X_test, y_test, seed: int = 0, n_estimators: int = 300):
    X_train_s, X_val_s, X_test_s = _scale_features(X_train, X_val, X_test)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    val_proba = rf.predict_proba(X_val_s)[:, 1]
    test_proba = rf.predict_proba(X_test_s)[:, 1]
    val_pred = rf.predict(X_val_s)
    test_pred = rf.predict(X_test_s)

    return {
        "validation_metrics": {
            "roc_auc": float(roc_auc_score(y_val, val_proba)),
            "f1": float(f1_score(y_val, val_pred)),
        },
        "test_metrics": {
            "roc_auc": float(roc_auc_score(y_test, test_proba)),
            "f1": float(f1_score(y_test, test_pred)),  # pos_label=1 = class of interest (active/toxic) -- verified, see comment above
            "mcc": float(matthews_corrcoef(y_test, test_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        },
        "test_probabilities": test_proba,
    }

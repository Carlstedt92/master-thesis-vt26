"""Linear probe helpers for feature matrices."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def _scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled


def evaluate_linear_probe_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
):
    """Train a linear regressor on frozen features and tune alpha on validation."""
    X_train, X_val, X_test = _scale_features(X_train, X_val, X_test)

    best_alpha = None
    best_val_r2 = -np.inf
    best_val_rmse = np.inf
    best_val_mae = np.inf
    eps = 1e-3

    for alpha in alphas:
        probe = Ridge(alpha=alpha)
        probe.fit(X_train, y_train)
        y_val_pred = probe.predict(X_val)

        val_r2 = float(r2_score(y_val, y_val_pred))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        val_mae = float(mean_absolute_error(y_val, y_val_pred))

        if val_r2 > best_val_r2 + eps:
            best_val_r2 = val_r2
            best_val_rmse = val_rmse
            best_val_mae = val_mae
            best_alpha = alpha
        elif abs(val_r2 - best_val_r2) <= eps:
            if val_rmse < best_val_rmse - eps:
                best_val_rmse = val_rmse
                best_val_mae = val_mae
                best_alpha = alpha
            elif abs(val_rmse - best_val_rmse) <= eps and val_mae < best_val_mae - eps:
                best_val_mae = val_mae
                best_alpha = alpha

    probe = Ridge(alpha=best_alpha)
    probe.fit(X_train, y_train)
    y_test_pred = probe.predict(X_test)

    return {
        "best_alpha": float(best_alpha),
        "validation_metrics": {
            "r2": float(best_val_r2),
            "rmse": float(best_val_rmse),
            "mae": float(best_val_mae),
        },
        "test_metrics": {
            "r2": float(r2_score(y_test, y_test_pred)),
            "mse": float(mean_squared_error(y_test, y_test_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "mae": float(mean_absolute_error(y_test, y_test_pred)),
        },
        "test_predictions": y_test_pred,
    }


def evaluate_linear_probe_classification(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    Cs: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
):
    """Train a linear classifier on frozen features and tune C on validation."""
    X_train, X_val, X_test = _scale_features(X_train, X_val, X_test)

    best_C = None
    best_val_roc_auc = -np.inf
    best_val_f1 = -np.inf
    eps = 1e-3

    for C in Cs:
        probe = LogisticRegression(max_iter=5000, C=C)
        probe.fit(X_train, y_train)
        val_proba = probe.predict_proba(X_val)[:, 1]
        val_pred = probe.predict(X_val)
        val_roc_auc = float(roc_auc_score(y_val, val_proba))
        val_f1 = float(f1_score(y_val, val_pred))

        if val_roc_auc > best_val_roc_auc + eps:
            best_val_roc_auc = val_roc_auc
            best_val_f1 = val_f1
            best_C = C
        elif abs(val_roc_auc - best_val_roc_auc) <= eps and val_f1 > best_val_f1 + eps:
            best_val_f1 = val_f1
            best_C = C

    probe = LogisticRegression(max_iter=5000, C=best_C)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)
    y_proba = probe.predict_proba(X_test)[:, 1]

    return {
        "best_C": float(best_C),
        "validation_metrics": {
            "roc_auc": float(best_val_roc_auc),
            "f1": float(best_val_f1),
        },
        "test_metrics": {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "f1": float(f1_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        },
        "test_predictions": y_pred,
        "test_probabilities": y_proba,
    }

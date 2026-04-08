"""kNN evaluation helpers for feature matrices."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def _scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled


def evaluate_knn_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    k_values: Sequence[int] = (3, 5, 11, 21, 31, 41, 51),
):
    """Tune k on validation and report final regression metrics on test."""
    X_train, X_val, X_test = _scale_features(X_train, X_val, X_test)

    best_k = None
    best_val_r2 = -np.inf
    best_val_rmse = np.inf
    best_val_mae = np.inf
    eps = 1e-3

    for k in k_values:
        knn_val = KNeighborsRegressor(n_neighbors=k, weights="distance")
        knn_val.fit(X_train, y_train)
        y_val_pred = knn_val.predict(X_val)

        val_r2 = float(r2_score(y_val, y_val_pred))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        val_mae = float(mean_absolute_error(y_val, y_val_pred))

        if val_r2 > best_val_r2 + eps:
            best_val_r2 = val_r2
            best_val_rmse = val_rmse
            best_val_mae = val_mae
            best_k = k
        elif abs(val_r2 - best_val_r2) <= eps:
            if val_rmse < best_val_rmse - eps:
                best_val_rmse = val_rmse
                best_val_mae = val_mae
                best_k = k
            elif abs(val_rmse - best_val_rmse) <= eps and val_mae < best_val_mae - eps:
                best_val_mae = val_mae
                best_k = k

    knn = KNeighborsRegressor(n_neighbors=best_k, weights="distance")
    knn.fit(X_train, y_train)
    y_test_pred = knn.predict(X_test)

    return {
        "best_k": int(best_k),
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


def evaluate_knn_classification(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    k_values: Sequence[int] = (3, 5, 11, 21, 31, 41, 51),
):
    """Tune k on validation and report final classification metrics on test."""
    X_train, X_val, X_test = _scale_features(X_train, X_val, X_test)

    best_k = None
    best_val_roc_auc = -np.inf
    best_val_f1 = -np.inf
    eps = 1e-3

    for k in k_values:
        knn_val = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn_val.fit(X_train, y_train)

        val_proba = knn_val.predict_proba(X_val)[:, 1]
        val_pred = knn_val.predict(X_val)
        val_roc_auc = float(roc_auc_score(y_val, val_proba))
        val_f1 = float(f1_score(y_val, val_pred))

        if val_roc_auc > best_val_roc_auc + eps:
            best_val_roc_auc = val_roc_auc
            best_val_f1 = val_f1
            best_k = k
        elif abs(val_roc_auc - best_val_roc_auc) <= eps and val_f1 > best_val_f1 + eps:
            best_val_f1 = val_f1
            best_k = k

    knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)[:, 1]

    return {
        "best_k": int(best_k),
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

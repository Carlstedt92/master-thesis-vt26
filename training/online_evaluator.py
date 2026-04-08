"""Online downstream evaluation utilities for SSL training."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from evaluate import DATASET_CONFIG, _build_embeddings_from_model, _load_rows


class OnlineDownstreamEvaluator:
    """Evaluates frozen embeddings during SSL training with fixed-k kNN."""

    def __init__(
        self,
        dataset_names: List[str],
        fixed_k: int,
        include_linear_probe: bool = True,
        linear_probe_alphas: List[float] | None = None,
        fingerprint_radius: int = 2,
        fingerprint_nbits: int = 2048,
    ):
        if fixed_k <= 0:
            raise ValueError("fixed_k must be a positive integer")

        self.dataset_names = dataset_names
        self.fixed_k = fixed_k
        self.include_linear_probe = include_linear_probe
        self.linear_probe_alphas = linear_probe_alphas if linear_probe_alphas else [1.0]
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_nbits = fingerprint_nbits
        self.dataset_context: Dict[str, Dict[str, Any]] = {}

        for dataset_name in self.dataset_names:
            if dataset_name not in DATASET_CONFIG:
                raise ValueError(f"Unsupported dataset for online eval: {dataset_name}")

            rows_by_split, dataset_stats = _load_rows(dataset_name)
            # Keep online eval strictly train+validation; test stays untouched for final reporting.
            rows_by_split = {
                "train": rows_by_split["train"],
                "val": rows_by_split["val"],
            }

            task = DATASET_CONFIG[dataset_name]["task"]
            if task != "regression":
                raise ValueError(
                    f"Online evaluation is regression-only in this workflow, got task={task} for {dataset_name}"
                )

            self.dataset_context[dataset_name] = {
                "task": task,
                "rows_by_split": rows_by_split,
                "dataset_stats": dataset_stats,
            }

    def _scale_train_val(self, train_X, val_X):
        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)
        val_X_scaled = scaler.transform(val_X)
        return train_X_scaled, val_X_scaled

    def _evaluate_fixed_k_knn(self, feature_set: Dict[str, np.ndarray], task: str) -> Dict[str, Any]:
        train_X, val_X = self._scale_train_val(feature_set["train_X"], feature_set["val_X"])
        train_y = feature_set["train_y"]
        val_y = feature_set["val_y"]

        if task != "regression":
            raise ValueError(f"Unsupported task: {task}")

        model = KNeighborsRegressor(n_neighbors=self.fixed_k, weights="distance")
        model.fit(train_X, train_y)
        val_pred = model.predict(val_X)
        return {
            "best_k": int(self.fixed_k),
            "validation_metrics": {
                "r2": float(r2_score(val_y, val_pred)),
                "rmse": float(np.sqrt(mean_squared_error(val_y, val_pred))),
                "mae": float(mean_absolute_error(val_y, val_pred)),
            },
        }

    def _evaluate_linear_probe(self, feature_set: Dict[str, np.ndarray], task: str) -> Dict[str, Any]:
        train_X, val_X = self._scale_train_val(feature_set["train_X"], feature_set["val_X"])
        train_y = feature_set["train_y"]
        val_y = feature_set["val_y"]

        if task != "regression":
            raise ValueError(f"Unsupported task: {task}")

        best_alpha = None
        best_metrics = None
        for alpha in self.linear_probe_alphas:
            probe = Ridge(alpha=alpha)
            probe.fit(train_X, train_y)
            val_pred = probe.predict(val_X)
            metrics = {
                "r2": float(r2_score(val_y, val_pred)),
                "rmse": float(np.sqrt(mean_squared_error(val_y, val_pred))),
                "mae": float(mean_absolute_error(val_y, val_pred)),
            }
            if best_metrics is None or metrics["r2"] > best_metrics["r2"]:
                best_alpha = float(alpha)
                best_metrics = metrics
        return {
            "best_alpha": float(best_alpha),
            "validation_metrics": best_metrics,
        }

    def evaluate_model(self, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
        dataset_results: Dict[str, Any] = {}
        primary_scores = []

        for dataset_name in self.dataset_names:
            context = self.dataset_context[dataset_name]
            task = context["task"]

            embedding_features, embedding_stats = _build_embeddings_from_model(
                context["rows_by_split"], model, device
            )
            embedding_knn = self._evaluate_fixed_k_knn(embedding_features, task)
            embedding_linear_probe = None
            if self.include_linear_probe:
                embedding_linear_probe = self._evaluate_linear_probe(embedding_features, task)

            if task == "regression":
                primary_metric_name = "val_r2"
                primary_metric_value = float(embedding_knn["validation_metrics"]["r2"])

            primary_scores.append(primary_metric_value)

            dataset_results[dataset_name] = {
                "task": task,
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": primary_metric_value,
                "fixed_k": int(self.fixed_k),
                "embeddings": {
                    **embedding_stats,
                    "knn": embedding_knn,
                },
                "dataset_stats": context["dataset_stats"],
            }
            if embedding_linear_probe is not None:
                dataset_results[dataset_name]["embeddings"]["linear_probe"] = embedding_linear_probe

        aggregate_score = float(np.mean(primary_scores)) if primary_scores else float("-inf")

        return {
            "fixed_k": int(self.fixed_k),
            "linear_probe_enabled": bool(self.include_linear_probe),
            "linear_probe_alphas": list(self.linear_probe_alphas),
            "dataset_names": list(self.dataset_names),
            "aggregate_primary_score": aggregate_score,
            "aggregate_primary_score_definition": "mean(validation primary metric across datasets)",
            "datasets": dataset_results,
        }

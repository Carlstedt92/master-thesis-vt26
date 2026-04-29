"""Random Forest evaluation on LIPO, BACE, and Tox21 datasets.

Compares Random Forest performance on SSL embeddings vs Morgan fingerprints.

Examples:
  python rf_eval.py --model GAT_MASK_9M_1 --dataset lipo
  python rf_eval.py --model GAT_MASK_9M_1 --dataset bace
  python rf_eval.py --model GAT_MASK_9M_1 --dataset tox21
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from evaluation.linear_probe import (
    evaluate_linear_probe_regression,
    evaluate_linear_probe_classification,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="SSL model name (models/{name}/)")
    parser.add_argument("--dataset", choices=["lipo", "tox21", "bace"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--random-seed", type=int, default=None, help="Seed for random splits")
    parser.add_argument(
        "--n-estimators",
        type=str,
        default="50,100,200,300",
        help="Comma-separated n_estimators to tune",
    )
    parser.add_argument(
        "--max-depths",
        type=str,
        default="10,20,30,None",
        help="Comma-separated max_depth values to tune (None for unlimited)",
    )
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    device_pref = args.device

    if device_pref == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_pref)

    # Parse hyperparameters
    n_estimators_list = [int(s.strip()) for s in args.n_estimators.split(",") if s.strip()]
    max_depth_list = []
    for s in args.max_depths.split(","):
        s = s.strip()
        if s.lower() == "none":
            max_depth_list.append(None)
        else:
            max_depth_list.append(int(s))

    # Lazy import dataset-specific evaluators to reuse their helpers
    if dataset == "lipo":
        from knn_eval_lip import (
            resolve_checkpoint_path,
            load_lipo_splits_from_deepchem,
            build_embedding_features,
            build_fingerprint_features,
            infer_graph_featurization,
        )

        splits, stats = load_lipo_splits_from_deepchem(
            "data/MoleculeNet_LIPO_custom",
            "random" if args.random_seed is not None else "random",
            split_seed=args.random_seed,
        )
        rows_train = splits["train"]
        rows_val = splits["val"]
        rows_test = splits["test"]

        checkpoint = resolve_checkpoint_path(model_name, args.checkpoint)
        checkpoint_obj = torch.load(checkpoint, map_location=device, weights_only=False)
        from model.config import ModelConfig
        from model.gnn_model import GNNModel

        config = ModelConfig.from_dict(checkpoint_obj["config"])
        explicit_h, encode_h = infer_graph_featurization(config)
        model = GNNModel.from_config(config).to(device)
        model.load_state_dict(checkpoint_obj["model_state_dict"])
        model.eval()

        X_train, y_train, _ = build_embedding_features(
            rows_train, model, device, explicit_h, encode_h
        )
        X_val, y_val, _ = build_embedding_features(rows_val, model, device, explicit_h, encode_h)
        X_test, y_test, _ = build_embedding_features(
            rows_test, model, device, explicit_h, encode_h
        )

        fp_train, _, _ = build_fingerprint_features(rows_train)
        fp_val, _, _ = build_fingerprint_features(rows_val)
        fp_test, _, _ = build_fingerprint_features(rows_test)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        fp_scaler = StandardScaler()
        fp_train = fp_scaler.fit_transform(fp_train)
        fp_val = fp_scaler.transform(fp_val)
        fp_test = fp_scaler.transform(fp_test)

        # Evaluate embeddings
        emb_result = evaluate_rf_regression(
            X_train, y_train, X_val, y_val, X_test, y_test, n_estimators_list, max_depth_list
        )

        # Evaluate fingerprints
        fp_result = evaluate_rf_regression(
            fp_train, y_train, fp_val, y_val, fp_test, y_test, n_estimators_list, max_depth_list
        )

        result = {
            "dataset": "LIPO",
            "primary_metric": "rmse",
            "embeddings": emb_result,
            "fingerprints": fp_result,
        }

    elif dataset == "bace":
        from knn_eval_bace import (
            resolve_checkpoint_path,
            load_bace_splits_from_deepchem,
            build_embedding_features,
            build_fingerprint_features,
            infer_graph_featurization,
        )

        splits, stats = load_bace_splits_from_deepchem("data/MoleculeNet_BACE_custom", "scaffold")
        rows_train = splits["train"]
        rows_val = splits["val"]
        rows_test = splits["test"]

        checkpoint = resolve_checkpoint_path(model_name, args.checkpoint)
        checkpoint_obj = torch.load(checkpoint, map_location=device, weights_only=False)
        from model.config import ModelConfig
        from model.gnn_model import GNNModel

        config = ModelConfig.from_dict(checkpoint_obj["config"])
        explicit_h, encode_h = infer_graph_featurization(config)
        model = GNNModel.from_config(config).to(device)
        model.load_state_dict(checkpoint_obj["model_state_dict"])
        model.eval()

        X_train, y_train, _ = build_embedding_features(
            rows_train, model, device, explicit_h, encode_h
        )
        X_val, y_val, _ = build_embedding_features(rows_val, model, device, explicit_h, encode_h)
        X_test, y_test, _ = build_embedding_features(
            rows_test, model, device, explicit_h, encode_h
        )

        fp_train, _, _ = build_fingerprint_features(rows_train)
        fp_val, _, _ = build_fingerprint_features(rows_val)
        fp_test, _, _ = build_fingerprint_features(rows_test)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        fp_scaler = StandardScaler()
        fp_train = fp_scaler.fit_transform(fp_train)
        fp_val = fp_scaler.transform(fp_val)
        fp_test = fp_scaler.transform(fp_test)

        # Evaluate embeddings
        emb_result = evaluate_rf_classification(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            n_estimators_list,
            max_depth_list,
        )

        # Evaluate fingerprints
        fp_result = evaluate_rf_classification(
            fp_train,
            y_train,
            fp_val,
            y_val,
            fp_test,
            y_test,
            n_estimators_list,
            max_depth_list,
        )

        result = {
            "dataset": "BACE",
            "primary_metric": "roc_auc",
            "embeddings": emb_result,
            "fingerprints": fp_result,
        }

    elif dataset == "tox21":
        from knn_eval_tox21 import (
            resolve_checkpoint_path,
            load_tox21_splits_from_deepchem,
            build_embedding_features,
            build_fingerprint_features,
            infer_graph_featurization,
        )

        splits, stats = load_tox21_splits_from_deepchem(
            "data/MoleculeNet_Tox21_custom", "random", split_seed=args.random_seed
        )
        rows_train = splits["train"]
        rows_val = splits["val"]
        rows_test = splits["test"]

        checkpoint = resolve_checkpoint_path(model_name, args.checkpoint)
        checkpoint_obj = torch.load(checkpoint, map_location=device, weights_only=False)
        from model.config import ModelConfig
        from model.gnn_model import GNNModel

        config = ModelConfig.from_dict(checkpoint_obj["config"])
        explicit_h, encode_h = infer_graph_featurization(config)
        model = GNNModel.from_config(config).to(device)
        model.load_state_dict(checkpoint_obj["model_state_dict"])
        model.eval()

        # Build embeddings
        emb_train, idx_train, _ = build_embedding_features(
            splits["train"]["smiles"], model, device, explicit_h, encode_h
        )
        emb_val, idx_val, _ = build_embedding_features(
            splits["val"]["smiles"], model, device, explicit_h, encode_h
        )
        emb_test, idx_test, _ = build_embedding_features(
            splits["test"]["smiles"], model, device, explicit_h, encode_h
        )

        # Build fingerprints
        fp_train, fp_idx_train, _ = build_fingerprint_features(splits["train"]["smiles"])
        fp_val, fp_idx_val, _ = build_fingerprint_features(splits["val"]["smiles"])
        fp_test, fp_idx_test, _ = build_fingerprint_features(splits["test"]["smiles"])

        labels_train = splits["train"]["labels"]
        labels_val = splits["val"]["labels"]
        labels_test = splits["test"]["labels"]

        num_tasks = labels_train.shape[1]

        per_task_results = []

        def evaluate_tox21_feature_set(
            X_train, idx_train, X_val, idx_val, X_test, idx_test, labels_train, labels_val, labels_test
        ):
            per_task = []

            for t in range(num_tasks):

                def pick(X, idxs, labels):
                    lab = labels[:, t]
                    finite = np.isfinite(lab)
                    keep = np.isin(np.arange(len(lab)), idxs) & finite
                    return X[keep], lab[keep].astype(int)

                Xtr, ytr = pick(X_train, idx_train, labels_train)
                Xv, yv = pick(X_val, idx_val, labels_val)
                Xt, yt = pick(X_test, idx_test, labels_test)

                if len(np.unique(ytr)) < 2 or len(np.unique(yv)) < 2 or len(np.unique(yt)) < 2:
                    continue

                per_task.append(
                    evaluate_rf_classification(
                        Xtr, ytr, Xv, yv, Xt, yt, n_estimators_list, max_depth_list
                    )
                )

            val_scores = [r["validation_metrics"]["roc_auc"] for r in per_task]
            test_scores = [r["test_metrics"]["roc_auc"] for r in per_task]
            return {
                "best_n_estimators": (
                    int(round(np.mean([r["best_n_estimators"] for r in per_task])))
                    if per_task
                    else None
                ),
                "best_max_depth": (
                    int(round(np.mean([r["best_max_depth"] for r in per_task if r["best_max_depth"] is not None])))
                    if per_task
                    else None
                ),
                "validation_metrics": {
                    "roc_auc_mean_tasks": float(np.mean(val_scores)) if val_scores else None,
                    "roc_auc_std_tasks": (
                        float(np.std(val_scores, ddof=1)) if len(val_scores) > 1 else 0.0
                    ),
                },
                "test_metrics": {
                    "roc_auc_mean_tasks": float(np.mean(test_scores)) if test_scores else None,
                    "roc_auc_std_tasks": (
                        float(np.std(test_scores, ddof=1)) if len(test_scores) > 1 else 0.0
                    ),
                },
                "per_task": per_task,
            }

        result = {
            "dataset": "Tox21",
            "primary_metric": "roc_auc_mean_tasks",
            "embeddings": evaluate_tox21_feature_set(
                emb_train, idx_train, emb_val, idx_val, emb_test, idx_test, labels_train, labels_val, labels_test
            ),
            "fingerprints": evaluate_tox21_feature_set(
                fp_train, fp_idx_train, fp_val, fp_idx_val, fp_test, fp_idx_test, labels_train, labels_val, labels_test
            ),
        }

    out_dir = Path(f"models/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rf_eval_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    print(f"Saved RF evaluation results: {out_path}")


def evaluate_rf_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    n_estimators_list,
    max_depth_list,
):
    """Tune RF hyperparameters on validation and evaluate on test."""
    best_n_est = None
    best_max_depth = None
    best_val_r2 = -np.inf
    best_val_rmse = np.inf
    eps = 1e-3

    for n_est in n_estimators_list:
        for max_depth in max_depth_list:
            rf = RandomForestRegressor(
                n_estimators=n_est, max_depth=max_depth, n_jobs=-1, random_state=42
            )
            rf.fit(X_train, y_train)
            y_val_pred = rf.predict(X_val)
            val_r2 = float(r2_score(y_val, y_val_pred))
            val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))

            if val_r2 > best_val_r2 + eps:
                best_val_r2 = val_r2
                best_val_rmse = val_rmse
                best_n_est = n_est
                best_max_depth = max_depth
            elif abs(val_r2 - best_val_r2) <= eps and val_rmse < best_val_rmse - eps:
                best_val_rmse = val_rmse
                best_n_est = n_est
                best_max_depth = max_depth

    rf = RandomForestRegressor(
        n_estimators=best_n_est, max_depth=best_max_depth, n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    y_test_pred = rf.predict(X_test)

    return {
        "best_n_estimators": int(best_n_est),
        "best_max_depth": best_max_depth,
        "validation_metrics": {
            "r2": float(best_val_r2),
            "rmse": float(best_val_rmse),
        },
        "test_metrics": {
            "r2": float(r2_score(y_test, y_test_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "mae": float(mean_absolute_error(y_test, y_test_pred)),
        },
        "test_predictions": y_test_pred,
    }


def evaluate_rf_classification(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    n_estimators_list,
    max_depth_list,
):
    """Tune RF hyperparameters on validation and evaluate on test."""
    best_n_est = None
    best_max_depth = None
    best_val_roc_auc = -np.inf
    best_val_f1 = -np.inf
    eps = 1e-3

    for n_est in n_estimators_list:
        for max_depth in max_depth_list:
            rf = RandomForestClassifier(
                n_estimators=n_est, max_depth=max_depth, n_jobs=-1, random_state=42
            )
            rf.fit(X_train, y_train)
            val_proba = rf.predict_proba(X_val)[:, 1]
            val_pred = rf.predict(X_val)
            val_roc_auc = float(roc_auc_score(y_val, val_proba))
            val_f1 = float(f1_score(y_val, val_pred))

            if val_roc_auc > best_val_roc_auc + eps:
                best_val_roc_auc = val_roc_auc
                best_val_f1 = val_f1
                best_n_est = n_est
                best_max_depth = max_depth
            elif abs(val_roc_auc - best_val_roc_auc) <= eps and val_f1 > best_val_f1 + eps:
                best_val_f1 = val_f1
                best_n_est = n_est
                best_max_depth = max_depth

    rf = RandomForestClassifier(
        n_estimators=best_n_est, max_depth=best_max_depth, n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    return {
        "best_n_estimators": int(best_n_est),
        "best_max_depth": best_max_depth,
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


if __name__ == "__main__":
    main()

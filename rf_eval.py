"""Random Forest evaluation on LIPO, BACE, and Tox21 datasets.

Compares Random Forest performance on SSL embeddings vs Morgan fingerprints.

Examples:
    uv run python rf_eval.py --model GAT_MASK_9M_1 --dataset lipo
    uv run python rf_eval.py --model GAT_MASK_9M_1 --dataset bace
    uv run python rf_eval.py --model GAT_MASK_9M_1 --dataset tox21
    uv run python rf_eval.py --model GAT_MASK_9M_1 --dataset all
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
from plotting.rf_boxplots import (
    update_random_seed_summary,
    finalize_random_seed_summary,
    save_random_seed_artifacts,
)


def load_model_from_checkpoint(checkpoint_path, device):
    from model.config import ModelConfig
    from model.gnn_model import GNNModel

    checkpoint_obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint_obj["config"])
    model = GNNModel.from_config(config).to(device)
    model.load_state_dict(checkpoint_obj["model_state_dict"])
    model.eval()
    return config, model


def evaluate_lipo(model_name, checkpoint_path, device, random_seed, n_estimators_list, max_depth_list):
    from evaluation.knn_lipo import (
        resolve_checkpoint_path,
        load_lipo_splits_from_deepchem,
        build_embedding_features,
        build_fingerprint_features,
        infer_graph_featurization,
    )

    splits, _ = load_lipo_splits_from_deepchem(
        "data/MoleculeNet_LIPO_custom",
        "random",
        split_seed=random_seed,
    )
    rows_train = splits["train"]
    rows_val = splits["val"]
    rows_test = splits["test"]

    checkpoint = resolve_checkpoint_path(model_name, checkpoint_path)
    config, model = load_model_from_checkpoint(checkpoint, device)
    explicit_h, encode_h = infer_graph_featurization(config)

    X_train, y_train, _ = build_embedding_features(rows_train, model, device, explicit_h, encode_h)
    X_val, y_val, _ = build_embedding_features(rows_val, model, device, explicit_h, encode_h)
    X_test, y_test, _ = build_embedding_features(rows_test, model, device, explicit_h, encode_h)

    fp_train, _, _ = build_fingerprint_features(rows_train)
    fp_val, _, _ = build_fingerprint_features(rows_val)
    fp_test, _, _ = build_fingerprint_features(rows_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    fp_scaler = StandardScaler()
    fp_train = fp_scaler.fit_transform(fp_train)
    fp_val = fp_scaler.transform(fp_val)
    fp_test = fp_scaler.transform(fp_test)

    emb_result = evaluate_rf_regression(
        X_train, y_train, X_val, y_val, X_test, y_test, n_estimators_list, max_depth_list
    )
    fp_result = evaluate_rf_regression(
        fp_train, y_train, fp_val, y_val, fp_test, y_test, n_estimators_list, max_depth_list
    )

    return {
        "dataset": "LIPO",
        "primary_metric": "rmse",
        "embeddings": emb_result,
        "fingerprints": fp_result,
    }


def evaluate_bace(model_name, checkpoint_path, device, n_estimators_list, max_depth_list, split_seed=None):
    from evaluation.knn_bace import (
        resolve_checkpoint_path,
        load_bace_splits_from_deepchem,
        build_embedding_features,
        build_fingerprint_features,
        infer_graph_featurization,
    )

    splits, _ = load_bace_splits_from_deepchem(
        "data/MoleculeNet_BACE_custom",
        "scaffold",
        split_seed=split_seed,
    )
    rows_train = splits["train"]
    rows_val = splits["val"]
    rows_test = splits["test"]

    checkpoint = resolve_checkpoint_path(model_name, checkpoint_path)
    config, model = load_model_from_checkpoint(checkpoint, device)
    explicit_h, encode_h = infer_graph_featurization(config)

    X_train, y_train, _ = build_embedding_features(rows_train, model, device, explicit_h, encode_h)
    X_val, y_val, _ = build_embedding_features(rows_val, model, device, explicit_h, encode_h)
    X_test, y_test, _ = build_embedding_features(rows_test, model, device, explicit_h, encode_h)

    fp_train, _, _ = build_fingerprint_features(rows_train)
    fp_val, _, _ = build_fingerprint_features(rows_val)
    fp_test, _, _ = build_fingerprint_features(rows_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    fp_scaler = StandardScaler()
    fp_train = fp_scaler.fit_transform(fp_train)
    fp_val = fp_scaler.transform(fp_val)
    fp_test = fp_scaler.transform(fp_test)

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

    return {
        "dataset": "BACE",
        "primary_metric": "roc_auc",
        "embeddings": emb_result,
        "fingerprints": fp_result,
    }


def evaluate_tox21(model_name, checkpoint_path, device, random_seed, n_estimators_list, max_depth_list):
    from evaluation.knn_tox21 import (
        resolve_checkpoint_path,
        load_tox21_splits_from_deepchem,
        build_embedding_features,
        build_fingerprint_features,
        infer_graph_featurization,
    )

    splits, _ = load_tox21_splits_from_deepchem(
        "data/MoleculeNet_Tox21_custom", "random", split_seed=random_seed
    )

    checkpoint = resolve_checkpoint_path(model_name, checkpoint_path)
    config, model = load_model_from_checkpoint(checkpoint, device)
    explicit_h, encode_h = infer_graph_featurization(config)

    emb_train, idx_train, _ = build_embedding_features(
        splits["train"]["smiles"], model, device, explicit_h, encode_h
    )
    emb_val, idx_val, _ = build_embedding_features(
        splits["val"]["smiles"], model, device, explicit_h, encode_h
    )
    emb_test, idx_test, _ = build_embedding_features(
        splits["test"]["smiles"], model, device, explicit_h, encode_h
    )

    fp_train, fp_idx_train, _ = build_fingerprint_features(splits["train"]["smiles"])
    fp_val, fp_idx_val, _ = build_fingerprint_features(splits["val"]["smiles"])
    fp_test, fp_idx_test, _ = build_fingerprint_features(splits["test"]["smiles"])

    labels_train = splits["train"]["labels"]
    labels_val = splits["val"]["labels"]
    labels_test = splits["test"]["labels"]
    num_tasks = labels_train.shape[1]

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
                evaluate_rf_classification(Xtr, ytr, Xv, yv, Xt, yt, n_estimators_list, max_depth_list)
            )

        val_scores = [r["validation_metrics"]["roc_auc"] for r in per_task]
        test_scores = [r["test_metrics"]["roc_auc"] for r in per_task]
        return {
            "best_n_estimators": (
                int(round(np.mean([r["best_n_estimators"] for r in per_task]))) if per_task else None
            ),
            "best_max_depth": (
                int(round(np.mean([r["best_max_depth"] for r in per_task if r["best_max_depth"] is not None])))
                if per_task
                else None
            ),
            "validation_metrics": {
                "roc_auc_mean_tasks": float(np.mean(val_scores)) if val_scores else None,
                "roc_auc_std_tasks": float(np.std(val_scores, ddof=1)) if len(val_scores) > 1 else 0.0,
            },
            "test_metrics": {
                "roc_auc_mean_tasks": float(np.mean(test_scores)) if test_scores else None,
                "roc_auc_std_tasks": float(np.std(test_scores, ddof=1)) if len(test_scores) > 1 else 0.0,
            },
            "per_task": per_task,
        }

    return {
        "dataset": "Tox21",
        "primary_metric": "roc_auc_mean_tasks",
        "embeddings": evaluate_tox21_feature_set(
            emb_train, idx_train, emb_val, idx_val, emb_test, idx_test, labels_train, labels_val, labels_test
        ),
        "fingerprints": evaluate_tox21_feature_set(
            fp_train, fp_idx_train, fp_val, fp_idx_val, fp_test, fp_idx_test, labels_train, labels_val, labels_test
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="SSL model name (models/{name}/)")
    parser.add_argument("--dataset", choices=["all", "lipo", "tox21", "bace"], default="all")
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
    parser.add_argument(
        "--random-split-seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds used only when a dataset split is random.",
    )
    parser.add_argument(
        "--allow-partial-results",
        action="store_true",
        help="Do not fail the run if some dataset/seed evaluations fail.",
    )
    args = parser.parse_args()

    model_name = args.model
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

    random_split_seeds = [int(value.strip()) for value in args.random_split_seeds.split(",") if value.strip()]
    if not random_split_seeds:
        raise ValueError("At least one --random-split-seeds value is required.")

    out_dir = Path(f"models/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_failures = []
    random_seed_summary = {}

    if args.dataset in {"all", "lipo"}:
        print(f"\n--- LIPO (random seeds: {random_split_seeds}) ---")
        for seed in random_split_seeds:
            try:
                print(f"\n[LIPO] Running seed={seed}")
                result = evaluate_lipo(
                    model_name, args.checkpoint, device, seed, n_estimators_list, max_depth_list
                )
                # Save per-eval log under eval_logs/random_forest/LIPO/{model}/
                try:
                    model_eval_dir = Path(f"models/{model_name}") / "eval_logs" / "random_forest" / "LIPO"
                    model_eval_dir.mkdir(parents=True, exist_ok=True)
                    eval_file = model_eval_dir / f"rf_lipo_seed{seed}.json"
                    with open(eval_file, 'w') as ef:
                        json.dump(result, ef, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
                except Exception:
                    pass
                result["split_seed"] = seed
                update_random_seed_summary(random_seed_summary, result)
            except Exception as exc:
                print(f"[FAILED][LIPO][seed={seed}] {model_name}: {exc}")
                model_failures.append(f"LIPO seed={seed}: {exc}")

    if args.dataset in {"all", "tox21"}:
        print(f"\n--- Tox21 (random seeds: {random_split_seeds}) ---")
        for seed in random_split_seeds:
            try:
                print(f"\n[Tox21] Running seed={seed}")
                result = evaluate_tox21(
                    model_name, args.checkpoint, device, seed, n_estimators_list, max_depth_list
                )
                # Save per-eval log under eval_logs/random_forest/TOX21/{model}/
                try:
                    model_eval_dir = Path(f"models/{model_name}") / "eval_logs" / "random_forest" / "TOX21"
                    model_eval_dir.mkdir(parents=True, exist_ok=True)
                    eval_file = model_eval_dir / f"rf_tox21_seed{seed}.json"
                    with open(eval_file, 'w') as ef:
                        json.dump(result, ef, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
                except Exception:
                    pass
                result["split_seed"] = seed
                update_random_seed_summary(random_seed_summary, result)
            except Exception as exc:
                print(f"[FAILED][Tox21][seed={seed}] {model_name}: {exc}")
                model_failures.append(f"Tox21 seed={seed}: {exc}")

    if args.dataset in {"all", "bace"}:
        print(f"\n--- BACE (scaffold, seeds: {random_split_seeds}) ---")
        for seed in random_split_seeds:
            try:
                print(f"\n[BACE] Running seed={seed}")
                result = evaluate_bace(
                    model_name,
                    args.checkpoint,
                    device,
                    n_estimators_list,
                    max_depth_list,
                    split_seed=seed,
                )
                # Save per-eval log under eval_logs/random_forest/BACE/{model}/
                try:
                    model_eval_dir = Path(f"models/{model_name}") / "eval_logs" / "random_forest" / "BACE"
                    model_eval_dir.mkdir(parents=True, exist_ok=True)
                    eval_file = model_eval_dir / f"rf_bace_seed{seed}.json"
                    with open(eval_file, 'w') as ef:
                        json.dump(result, ef, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
                except Exception:
                    pass
                result["split_seed"] = seed
                update_random_seed_summary(random_seed_summary, result)
            except Exception as exc:
                print(f"[FAILED][BACE][seed={seed}] {model_name}: {exc}")
                model_failures.append(f"BACE seed={seed}: {exc}")

    finalized_summary = finalize_random_seed_summary(random_seed_summary)
    if finalized_summary:
        save_random_seed_artifacts(model_name, finalized_summary)

    if model_failures:
        print(f"\nEncountered {len(model_failures)} failed evaluations for {model_name}:")
        for item in model_failures:
            print(f"  - {item}")
        if not args.allow_partial_results:
            raise RuntimeError(
                "Evaluation completed with failures. "
                "Re-run with --allow-partial-results to keep partial outputs without failing."
            )


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
    candidate_validation_metrics = []

    for n_est in n_estimators_list:
        for max_depth in max_depth_list:
            rf = RandomForestRegressor(
                n_estimators=n_est, max_depth=max_depth, n_jobs=-1, random_state=42
            )
            rf.fit(X_train, y_train)
            y_val_pred = rf.predict(X_val)
            val_r2 = float(r2_score(y_val, y_val_pred))
            val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
            candidate_validation_metrics.append(
                {
                    "n_estimators": int(n_est),
                    "max_depth": max_depth,
                    "r2": val_r2,
                    "rmse": val_rmse,
                }
            )

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
        "candidate_validation_metrics": candidate_validation_metrics,
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
    candidate_validation_metrics = []

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
            candidate_validation_metrics.append(
                {
                    "n_estimators": int(n_est),
                    "max_depth": max_depth,
                    "roc_auc": val_roc_auc,
                    "f1": val_f1,
                }
            )

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
        "candidate_validation_metrics": candidate_validation_metrics,
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

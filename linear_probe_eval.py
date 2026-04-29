"""Run linear-probe evaluation (frozen backbone) on LIPO, Tox21, and BACE.

This script loads a trained SSL checkpoint, extracts frozen graph embeddings
for train/val/test splits and runs a sklearn linear probe (Ridge / LogisticRegression)
using helpers in `evaluation/linear_probe.py`.

Examples:
  python linear_probe_eval.py --model GDZ_GAT_TEST --dataset lipo
  python linear_probe_eval.py --model GDZ_GAT_TEST --dataset tox21 --random-seed 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

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
    parser.add_argument("--random-seed", type=int, default=None, help="Seed for random splits (when applicable)")
    parser.add_argument("--alphas", type=str, default="0.01,0.1,1.0,10.0", help="Comma-separated alphas for Ridge")
    parser.add_argument("--Cs", type=str, default="0.01,0.1,1.0,10.0", help="Comma-separated Cs for LogisticRegression")
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    device_pref = args.device

    if device_pref == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_pref)

    # Lazy import dataset-specific evaluators to reuse their helpers
    if dataset == "lipo":
        from knn_eval_lip import (
            resolve_checkpoint_path,
            load_lipo_splits_from_deepchem,
            build_embedding_features,
            build_fingerprint_features,
            infer_graph_featurization,
        )

        splits, stats = load_lipo_splits_from_deepchem('data/MoleculeNet_LIPO_custom', 'random' if args.random_seed is not None else 'random', split_seed=args.random_seed)
        rows_train = splits['train']
        rows_val = splits['val']
        rows_test = splits['test']

        checkpoint = resolve_checkpoint_path(model_name, args.checkpoint)
        checkpoint_obj = torch.load(checkpoint, map_location=device, weights_only=False)
        from model.config import ModelConfig
        from model.gnn_model import GNNModel

        config = ModelConfig.from_dict(checkpoint_obj['config'])
        explicit_h, encode_h = infer_graph_featurization(config)
        model = GNNModel.from_config(config).to(device)
        model.load_state_dict(checkpoint_obj['model_state_dict'])
        model.eval()

        X_train, y_train, _ = build_embedding_features(rows_train, model, device, explicit_h, encode_h)
        X_val, y_val, _ = build_embedding_features(rows_val, model, device, explicit_h, encode_h)
        X_test, y_test, _ = build_embedding_features(rows_test, model, device, explicit_h, encode_h)

        fp_train_X, fp_train_y, _ = build_fingerprint_features(rows_train)
        fp_val_X, fp_val_y, _ = build_fingerprint_features(rows_val)
        fp_test_X, fp_test_y, _ = build_fingerprint_features(rows_test)

        alphas = [float(s) for s in args.alphas.split(',') if s]
        result = {
            "dataset": "LIPO",
            "primary_metric": "rmse",
            "embeddings": evaluate_linear_probe_regression(X_train, y_train, X_val, y_val, X_test, y_test, alphas),
            "fingerprints": evaluate_linear_probe_regression(fp_train_X, fp_train_y, fp_val_X, fp_val_y, fp_test_X, fp_test_y, alphas),
        }

    elif dataset == "bace":
        from knn_eval_bace import (
            resolve_checkpoint_path,
            load_bace_splits_from_deepchem,
            build_embedding_features,
            build_fingerprint_features,
            infer_graph_featurization,
        )

        splits, stats = load_bace_splits_from_deepchem('data/MoleculeNet_BACE_custom', 'scaffold')
        rows_train = splits['train']
        rows_val = splits['val']
        rows_test = splits['test']

        checkpoint = resolve_checkpoint_path(model_name, args.checkpoint)
        checkpoint_obj = torch.load(checkpoint, map_location=device, weights_only=False)
        from model.config import ModelConfig
        from model.gnn_model import GNNModel

        config = ModelConfig.from_dict(checkpoint_obj['config'])
        explicit_h, encode_h = infer_graph_featurization(config)
        model = GNNModel.from_config(config).to(device)
        model.load_state_dict(checkpoint_obj['model_state_dict'])
        model.eval()

        X_train, y_train, _ = build_embedding_features(rows_train, model, device, explicit_h, encode_h)
        X_val, y_val, _ = build_embedding_features(rows_val, model, device, explicit_h, encode_h)
        X_test, y_test, _ = build_embedding_features(rows_test, model, device, explicit_h, encode_h)

        fp_train_X, fp_train_y, _ = build_fingerprint_features(rows_train)
        fp_val_X, fp_val_y, _ = build_fingerprint_features(rows_val)
        fp_test_X, fp_test_y, _ = build_fingerprint_features(rows_test)

        Cs = [float(s) for s in args.Cs.split(',') if s]
        result = {
            "dataset": "BACE",
            "primary_metric": "roc_auc",
            "embeddings": evaluate_linear_probe_classification(X_train, y_train, X_val, y_val, X_test, y_test, Cs),
            "fingerprints": evaluate_linear_probe_classification(fp_train_X, fp_train_y, fp_val_X, fp_val_y, fp_test_X, fp_test_y, Cs),
        }

    elif dataset == "tox21":
        # Tox21 is multi-task. We'll run a probe per task and aggregate ROC-AUC.
        from knn_eval_tox21 import (
            resolve_checkpoint_path,
            load_tox21_splits_from_deepchem,
            build_embedding_features,
            build_fingerprint_features,
            infer_graph_featurization,
        )

        splits, stats = load_tox21_splits_from_deepchem('data/MoleculeNet_Tox21_custom', 'random', split_seed=args.random_seed)
        # splits contain smi arrays and labels
        checkpoint = resolve_checkpoint_path(model_name, args.checkpoint)
        checkpoint_obj = torch.load(checkpoint, map_location=device, weights_only=False)
        from model.config import ModelConfig
        from model.gnn_model import GNNModel

        config = ModelConfig.from_dict(checkpoint_obj['config'])
        explicit_h, encode_h = infer_graph_featurization(config)
        model = GNNModel.from_config(config).to(device)
        model.load_state_dict(checkpoint_obj['model_state_dict'])
        model.eval()

        # Build embeddings for smiles arrays using tox21 helper which returns kept indices
        emb_train, idx_train, _ = build_embedding_features(splits['train']['smiles'], model, device, explicit_h, encode_h)
        emb_val, idx_val, _ = build_embedding_features(splits['val']['smiles'], model, device, explicit_h, encode_h)
        emb_test, idx_test, _ = build_embedding_features(splits['test']['smiles'], model, device, explicit_h, encode_h)

        fp_train, fp_idx_train, _ = build_fingerprint_features(splits['train']['smiles'])
        fp_val, fp_idx_val, _ = build_fingerprint_features(splits['val']['smiles'])
        fp_test, fp_idx_test, _ = build_fingerprint_features(splits['test']['smiles'])

        labels_train = splits['train']['labels']
        labels_val = splits['val']['labels']
        labels_test = splits['test']['labels']

        num_tasks = labels_train.shape[1]
        Cs = [float(s) for s in args.Cs.split(',') if s]

        per_task_results = []
        def evaluate_tox21_feature_set(X_train, idx_train, X_val, idx_val, X_test, idx_test, labels_train, labels_val, labels_test):
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

                per_task.append(evaluate_linear_probe_classification(Xtr, ytr, Xv, yv, Xt, yt, Cs))

            val_scores = [r['validation_metrics']['roc_auc'] for r in per_task]
            test_scores = [r['test_metrics']['roc_auc'] for r in per_task]
            return {
                'best_C': int(round(np.mean([r['best_C'] for r in per_task]))) if per_task else None,
                'validation_metrics': {
                    'roc_auc_mean_tasks': float(np.mean(val_scores)) if val_scores else None,
                    'roc_auc_std_tasks': float(np.std(val_scores, ddof=1)) if len(val_scores) > 1 else 0.0 if val_scores else None,
                },
                'test_metrics': {
                    'roc_auc_mean_tasks': float(np.mean(test_scores)) if test_scores else None,
                    'roc_auc_std_tasks': float(np.std(test_scores, ddof=1)) if len(test_scores) > 1 else 0.0 if test_scores else None,
                },
                'per_task': per_task,
            }

        result = {
            'dataset': 'Tox21',
            'primary_metric': 'roc_auc_mean_tasks',
            'embeddings': evaluate_tox21_feature_set(emb_train, idx_train, emb_val, idx_val, emb_test, idx_test, labels_train, labels_val, labels_test),
            'fingerprints': evaluate_tox21_feature_set(fp_train, fp_idx_train, fp_val, fp_idx_val, fp_test, fp_idx_test, labels_train, labels_val, labels_test),
        }

    out_dir = Path(f"models/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"linear_probe_{dataset}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

    print(f"Saved linear-probe results: {out_path}")


if __name__ == '__main__':
    main()

"""Train a linear head on top of a checkpointed model.

This script supports two modes:
 - frozen backbone (only train the linear head)
 - finetune backbone (train head + encoder)

It handles LIPO (regression), BACE (binary classification), and Tox21 (multi-task classification).

Example:
    python linear_probe_finetune.py --model GDZ_GAT_TEST --dataset lipo --epochs 10 --batch-size 32
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

from evaluation.linear_probe import (
    evaluate_linear_probe_regression,
    evaluate_linear_probe_classification,
)
from model.gnn_model import GNNModel


def load_checkpoint_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    from model.config import ModelConfig
    config = ModelConfig.from_dict(checkpoint['config'])
    model = GNNModel.from_config(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config


def rows_to_dataset(rows, explicit_hydrogens, encode_hydrogen_count):
    from datahandling.graph_creation import smiles_to_pygdata

    data_list = []
    labels = []
    for smiles, target in rows:
        data = smiles_to_pygdata(smiles, explicit_hydrogens=explicit_hydrogens, encode_hydrogen_count=encode_hydrogen_count)
        if data is None or data.num_nodes == 0:
            continue
        data_list.append(data)
        labels.append(float(target))

    return data_list, np.asarray(labels, dtype=np.float32)


def evaluate_model_on_split(model, data_list, labels, device, task='regression'):
    model.eval()
    preds = []
    with torch.no_grad():
        for data in data_list:
            data = data.to(device)
            batch = Batch.from_data_list([data])
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds.append(out.squeeze(0).cpu().numpy())
    preds = np.asarray(preds)

    if task == 'regression':
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        r2 = float(r2_score(labels, preds))
        rmse = float(np.sqrt(mean_squared_error(labels, preds)))
        mae = float(mean_absolute_error(labels, preds))
        return {'r2': r2, 'rmse': rmse, 'mae': mae}, preds
    else:
        from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
        prob = torch.sigmoid(torch.from_numpy(preds)).numpy() if preds.ndim == 2 and preds.shape[1] == 1 else preds
        pred_labels = (prob >= 0.5).astype(int).ravel()
        roc = float(roc_auc_score(labels, prob))
        f1 = float(f1_score(labels, pred_labels))
        bal = float(balanced_accuracy_score(labels, pred_labels))
        return {'roc_auc': roc, 'f1': f1, 'balanced_accuracy': bal}, pred_labels


def train_loop(model, optimizer, loss_fn, train_loader, device):
    model.train()
    total_loss = 0.0
    for batch_list, batch_labels in train_loader:
        # batch_list: list of PyG Data objects
        batch = Batch.from_data_list(batch_list).to(device)
        labels = torch.tensor(batch_labels, dtype=torch.float32, device=device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
    return total_loss


def make_batches(data_list, labels, batch_size):
    # Return list of (list_of_Data, np_labels)
    items = list(zip(data_list, labels.tolist()))
    batches = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i+batch_size]
        batch_list, batch_labels = zip(*chunk)
        batches.append((list(batch_list), np.asarray(batch_labels, dtype=np.float32)))
    return batches


def evaluate_fingerprint_regression_rows(rows_train, rows_val, rows_test, build_fingerprint_features, alphas):
    fp_train_X, fp_train_y, _ = build_fingerprint_features(rows_train)
    fp_val_X, fp_val_y, _ = build_fingerprint_features(rows_val)
    fp_test_X, fp_test_y, _ = build_fingerprint_features(rows_test)
    return evaluate_linear_probe_regression(fp_train_X, fp_train_y, fp_val_X, fp_val_y, fp_test_X, fp_test_y, alphas)


def evaluate_fingerprint_classification_rows(rows_train, rows_val, rows_test, build_fingerprint_features, Cs):
    fp_train_X, fp_train_y, _ = build_fingerprint_features(rows_train)
    fp_val_X, fp_val_y, _ = build_fingerprint_features(rows_val)
    fp_test_X, fp_test_y, _ = build_fingerprint_features(rows_test)
    return evaluate_linear_probe_classification(fp_train_X, fp_train_y, fp_val_X, fp_val_y, fp_test_X, fp_test_y, Cs)


def evaluate_fingerprint_tox21(smiles_train, smiles_val, smiles_test, labels_train, labels_val, labels_test, build_fingerprint_features, Cs):
    fp_train_X, fp_idx_train, _ = build_fingerprint_features(smiles_train)
    fp_val_X, fp_idx_val, _ = build_fingerprint_features(smiles_val)
    fp_test_X, fp_idx_test, _ = build_fingerprint_features(smiles_test)

    num_tasks = labels_train.shape[1]

    per_task = []
    for t in range(num_tasks):
        def pick(X, idxs, labels):
            lab = labels[:, t]
            finite = np.isfinite(lab)
            keep = np.isin(np.arange(len(lab)), idxs) & finite
            return X[keep], lab[keep].astype(int)

        Xtr, ytr = pick(fp_train_X, fp_idx_train, labels_train)
        Xv, yv = pick(fp_val_X, fp_idx_val, labels_val)
        Xt, yt = pick(fp_test_X, fp_idx_test, labels_test)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=["lipo", "bace", "tox21"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.dataset == 'lipo':
        from evaluation.knn_lipo import resolve_checkpoint_path, load_lipo_splits_from_deepchem, build_fingerprint_features, infer_graph_featurization
        splits, stats = load_lipo_splits_from_deepchem('data/MoleculeNet_LIPO_custom', 'random')
        rows_train = splits['train']
        rows_val = splits['val']
        rows_test = splits['test']

        checkpoint = resolve_checkpoint_path(args.model, args.checkpoint)
        model, config = load_checkpoint_model(checkpoint, device)
        explicit_h, encode_h = infer_graph_featurization(config)

        # Replace head with regression head
        from model.gnn_model import RegressionHead
        model.head = RegressionHead(input_dim=config.hidden_dim, hidden_dim=max(64, config.hidden_dim // 2), output_dim=1)

        for p in model.parameters():
            p.requires_grad = True

        train_data, train_labels = rows_to_dataset(rows_train, explicit_h, encode_h)
        val_data, val_labels = rows_to_dataset(rows_val, explicit_h, encode_h)
        test_data, test_labels = rows_to_dataset(rows_test, explicit_h, encode_h)

        train_batches = make_batches(train_data, train_labels, args.batch_size)

        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.MSELoss()

        best_val = float('inf')
        best_state = None
        for epoch in range(args.epochs):
            epoch_loss = train_loop(model, optimizer, loss_fn, train_batches, device)
            val_metrics, _ = evaluate_model_on_split(model, val_data, val_labels, device, task='regression')
            print(f"Epoch {epoch+1}/{args.epochs} loss={epoch_loss:.4f} val_rmse={val_metrics['rmse']:.4f}")
            if val_metrics['rmse'] < best_val:
                best_val = val_metrics['rmse']
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # Save best and final predictions
        if best_state is not None:
            model.load_state_dict(best_state)
        test_metrics, test_preds = evaluate_model_on_split(model, test_data, test_labels, device, task='regression')

        fingerprint_result = evaluate_fingerprint_regression_rows(rows_train, rows_val, rows_test, build_fingerprint_features, [0.01, 0.1, 1.0, 10.0])

        out = {
            'test_metrics': test_metrics,
            'fingerprints': fingerprint_result,
            'n_train': len(train_data),
            'n_val': len(val_data),
            'n_test': len(test_data),
        }

    elif args.dataset == 'bace':
        from evaluation.knn_bace import resolve_checkpoint_path, load_bace_splits_from_deepchem, build_fingerprint_features, infer_graph_featurization
        splits, stats = load_bace_splits_from_deepchem('data/MoleculeNet_BACE_custom', 'scaffold')
        rows_train = splits['train']
        rows_val = splits['val']
        rows_test = splits['test']

        checkpoint = resolve_checkpoint_path(args.model, args.checkpoint)
        model, config = load_checkpoint_model(checkpoint, device)
        explicit_h, encode_h = infer_graph_featurization(config)

        from model.gnn_model import ClassificationHead
        model.head = ClassificationHead(input_dim=config.hidden_dim, hidden_dim=max(64, config.hidden_dim // 2), output_dim=1)

        for p in model.parameters():
            p.requires_grad = True

        train_data, train_labels = rows_to_dataset(rows_train, explicit_h, encode_h)
        val_data, val_labels = rows_to_dataset(rows_val, explicit_h, encode_h)
        test_data, test_labels = rows_to_dataset(rows_test, explicit_h, encode_h)

        train_batches = make_batches(train_data, train_labels, args.batch_size)

        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        best_val = float('-inf')
        best_state = None
        for epoch in range(args.epochs):
            epoch_loss = train_loop(model, optimizer, loss_fn, train_batches, device)
            val_metrics, _ = evaluate_model_on_split(model, val_data, val_labels, device, task='classification')
            print(f"Epoch {epoch+1}/{args.epochs} loss={epoch_loss:.4f} val_roc={val_metrics['roc_auc']:.4f}")
            if val_metrics['roc_auc'] > best_val:
                best_val = val_metrics['roc_auc']
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        test_metrics, test_preds = evaluate_model_on_split(model, test_data, test_labels, device, task='classification')

        fingerprint_result = evaluate_fingerprint_classification_rows(rows_train, rows_val, rows_test, build_fingerprint_features, [0.01, 0.1, 1.0, 10.0])

        out = {
            'test_metrics': test_metrics,
            'fingerprints': fingerprint_result,
            'n_train': len(train_data),
            'n_val': len(val_data),
            'n_test': len(test_data),
        }

    elif args.dataset == 'tox21':
        # Multi-task head with one logit per task
        from evaluation.knn_tox21 import resolve_checkpoint_path, load_tox21_splits_from_deepchem, build_fingerprint_features, infer_graph_featurization
        splits, stats = load_tox21_splits_from_deepchem('data/MoleculeNet_Tox21_custom', 'random')
        checkpoint = resolve_checkpoint_path(args.model, args.checkpoint)
        model, config = load_checkpoint_model(checkpoint, device)
        explicit_h, encode_h = infer_graph_featurization(config)

        labels_train = splits['train']['labels']
        labels_val = splits['val']['labels']
        labels_test = splits['test']['labels']
        num_tasks = labels_train.shape[1]

        # build Data lists that include per-sample full label vector (with NaNs)
        from datahandling.graph_creation import smiles_to_pygdata
        def build_multi(rows, labels_arr):
            data_list = []
            lab_list = []
            for i, smi in enumerate(rows):
                data = smiles_to_pygdata(str(smi), explicit_hydrogens=explicit_h, encode_hydrogen_count=encode_h)
                if data is None or data.num_nodes == 0:
                    continue
                data_list.append(data)
                lab_list.append(labels_arr[i])
            return data_list, np.asarray(lab_list)

        train_data, train_labels = build_multi(splits['train']['smiles'], labels_train)
        val_data, val_labels = build_multi(splits['val']['smiles'], labels_val)
        test_data, test_labels = build_multi(splits['test']['smiles'], labels_test)

        # Replace head with linear multi-task head
        from torch import nn as _nn
        model.head = _nn.Linear(config.hidden_dim, num_tasks)

        for p in model.parameters():
            p.requires_grad = True

        # Training uses BCEWithLogits with masking for NaNs
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

        best_val = float('-inf')
        best_state = None
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for i in range(0, len(train_data), args.batch_size):
                batch_list = train_data[i:i+args.batch_size]
                batch_labels = train_labels[i:i+args.batch_size]
                batch = Batch.from_data_list(batch_list).to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                # out shape (batch_size, num_tasks)
                labels_tensor = torch.tensor(batch_labels, dtype=torch.float32, device=device)
                mask = ~torch.isnan(labels_tensor)
                if mask.sum() == 0:
                    continue
                loss = nn.BCEWithLogitsLoss()(out[mask], labels_tensor[mask])
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())

            # Evaluate per-task mean ROC-AUC on validation
            # We'll compute per-task ROC-AUC where there are positive and negative examples
            model.eval()
            val_preds = []
            with torch.no_grad():
                for data in val_data:
                    d = data.to(device)
                    b = Batch.from_data_list([d])
                    p = model(b.x, b.edge_index, b.edge_attr, b.batch).cpu().numpy()
                    val_preds.append(p.ravel())
            val_preds = np.asarray(val_preds)
            per_task_aucs = []
            from sklearn.metrics import roc_auc_score
            for t in range(num_tasks):
                lab = val_labels[:, t]
                finite_mask = np.isfinite(lab)
                if finite_mask.sum() < 2:
                    continue
                try:
                    auc = roc_auc_score(lab[finite_mask].astype(int), val_preds[finite_mask, t])
                    per_task_aucs.append(auc)
                except Exception:
                    continue
            mean_auc = float(np.mean(per_task_aucs)) if per_task_aucs else None
            print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss:.4f} val_mean_roc_auc={mean_auc}")
            if mean_auc is not None and mean_auc > best_val:
                best_val = mean_auc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation on test
        model.eval()
        test_preds = []
        with torch.no_grad():
            for data in test_data:
                d = data.to(device)
                b = Batch.from_data_list([d])
                p = model(b.x, b.edge_index, b.edge_attr, b.batch).cpu().numpy()
                test_preds.append(p.ravel())
        test_preds = np.asarray(test_preds)
        per_task_aucs = []
        from sklearn.metrics import roc_auc_score
        for t in range(num_tasks):
            lab = test_labels[:, t]
            finite_mask = np.isfinite(lab)
            if finite_mask.sum() < 2:
                continue
            try:
                auc = roc_auc_score(lab[finite_mask].astype(int), test_preds[finite_mask, t])
                per_task_aucs.append(auc)
            except Exception:
                continue
        fingerprint_result = evaluate_fingerprint_tox21(
            splits['train']['smiles'],
            splits['val']['smiles'],
            splits['test']['smiles'],
            labels_train,
            labels_val,
            labels_test,
            build_fingerprint_features,
            [0.01, 0.1, 1.0, 10.0],
        )

        out = {
            'per_task_aucs': per_task_aucs,
            'mean_test_roc_auc': float(np.mean(per_task_aucs)) if per_task_aucs else None,
            'fingerprints': fingerprint_result,
        }

    # Save results
    out_dir = Path(f"models/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"linear_probe_finetune_{args.dataset}.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

    print(f"Saved finetune linear-probe results: {out_path}")


if __name__ == '__main__':
    main()

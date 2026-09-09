"""Evaluate SSL encoders on a ZINC subset with RDKit-computed LogP as target.

Two downstream probes on frozen embeddings, both under 5-fold CV:
  1. MLP regression head (RegressionHead from model/gnn_model.py), trained via
     gradient descent, encoder frozen.
  2. Random Forest regressor (sklearn) on the same frozen embeddings.

Uses each model's "latest epoch" checkpoint (final_model.pth, the student),
not the best-eval checkpoint.

Usage:
  uv run python zinc_logp_probe_eval.py
  uv run python zinc_logp_probe_eval.py --n-molecules 200 --n-folds 3  # quick smoke test
"""

import argparse
import glob
import json
import random

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from datahandling.graph_creation import smiles_to_pygdata
from model.config import ModelConfig
from model.gnn_model import GNNModel, RegressionHead

DEFAULT_MODELS = {
    "MASK_5pct": "models/COMPARE_B1024_H1024_MASK_25EP_GAT_9M_ZINC/checkpoints/final_model.pth",
    "MASK_10pct": "models/COMPARE_B1024_H1024_MASK10PCT_50EP_GAT_9M_ZINC/checkpoints/final_model.pth",
    "MASK_30pct": "models/COMPARE_B1024_H1024_MASK30PCT_30EP_GAT_9M_ZINC/checkpoints/final_model.pth",
}

ZINC_DATA_DIR = "data/zinc/zinc_9M_data"
N_MOLECULES = 5000
N_FOLDS = 5
SEED = 1


def sample_zinc_molecules(data_dir: str, n_molecules: int, seed: int):
    """Sample SMILES from ZINC .smi files and compute LogP for each via RDKit."""
    rng = random.Random(seed)
    files = sorted(glob.glob(f"{data_dir}/*.smi"))
    rng.shuffle(files)

    pairs = []
    for path in files:
        with open(path, "r") as f:
            next(f, None)  # header line
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                smiles = parts[0]
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                logp = Descriptors.MolLogP(mol)
                pairs.append((smiles, logp))
                if len(pairs) >= n_molecules:
                    return pairs
    return pairs


def infer_featurization(config):
    explicit_hydrogens = getattr(config, "explicit_hydrogens", None)
    encode_hydrogen_count = getattr(config, "encode_hydrogen_count", None)
    if explicit_hydrogens is not None and encode_hydrogen_count is not None:
        return bool(explicit_hydrogens), bool(encode_hydrogen_count)
    return True, False


def build_embeddings(pairs, model, device, explicit_hydrogens, encode_hydrogen_count):
    """Frozen encoder forward pass -> (X embeddings, y logp) numpy arrays."""
    features = []
    labels = []
    invalid = 0
    model.eval()
    with torch.no_grad():
        for smiles, logp in pairs:
            data = smiles_to_pygdata(
                smiles, explicit_hydrogens=explicit_hydrogens, encode_hydrogen_count=encode_hydrogen_count
            )
            if data is None or data.num_nodes == 0:
                invalid += 1
                continue
            data = data.to(device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            emb = model.get_embeddings(data.x, data.edge_index, data.edge_attr, batch).squeeze(0)
            features.append(emb.cpu().numpy())
            labels.append(logp)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32), invalid


def train_eval_mlp_fold(X_train, y_train, X_val, y_val, hidden_dim, device, epochs=150, lr=1e-3):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-8)

    head = RegressionHead(input_dim=X_train.shape[1], hidden_dim=hidden_dim // 2, output_dim=1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    y_train_t = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32, device=device).unsqueeze(-1)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32, device=device)

    head.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = head(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

    head.eval()
    with torch.no_grad():
        val_pred = (head(X_val_t).squeeze(-1).cpu().numpy()) * y_std + y_mean

    return {
        "r2": float(r2_score(y_val, val_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
        "mae": float(mean_absolute_error(y_val, val_pred)),
    }


def train_eval_rf_fold(X_train, y_train, X_val, y_val, seed):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    val_pred = rf.predict(X_val_s)

    return {
        "r2": float(r2_score(y_val, val_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
        "mae": float(mean_absolute_error(y_val, val_pred)),
    }


def cross_validate(X, y, hidden_dim, device, seed, n_folds):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    mlp_metrics = {"r2": [], "rmse": [], "mae": []}
    rf_metrics = {"r2": [], "rmse": [], "mae": []}

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        mlp_result = train_eval_mlp_fold(X_train, y_train, X_val, y_val, hidden_dim, device)
        rf_result = train_eval_rf_fold(X_train, y_train, X_val, y_val, seed + fold_idx)

        for k in mlp_metrics:
            mlp_metrics[k].append(mlp_result[k])
            rf_metrics[k].append(rf_result[k])

        print(
            f"    fold {fold_idx + 1}/{n_folds}: "
            f"MLP R2={mlp_result['r2']:.4f} RMSE={mlp_result['rmse']:.4f} | "
            f"RF R2={rf_result['r2']:.4f} RMSE={rf_result['rmse']:.4f}"
        )

    def summarize(metrics):
        return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in metrics.items()}

    return {"mlp": summarize(mlp_metrics), "rf": summarize(rf_metrics)}


def main():
    parser = argparse.ArgumentParser(description="ZINC-subset LogP probe: MLP head vs Random Forest on frozen embeddings.")
    parser.add_argument("--n-molecules", type=int, default=N_MOLECULES)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="zinc_logp_probe_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"Sampling {args.n_molecules} ZINC molecules and computing LogP...")
    pairs = sample_zinc_molecules(ZINC_DATA_DIR, args.n_molecules, args.seed)
    print(f"  got {len(pairs)} valid (smiles, logp) pairs")

    results = {}
    for model_name, checkpoint_path in DEFAULT_MODELS.items():
        print(f"\n=== {model_name} ({checkpoint_path}) ===")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ModelConfig.from_dict(checkpoint["config"])
        explicit_h, encode_h = infer_featurization(config)

        model = GNNModel.from_config(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        X, y, invalid = build_embeddings(pairs, model, device, explicit_h, encode_h)
        print(f"  embeddings built: {len(y)} valid, {invalid} invalid SMILES, dim={X.shape[1]}")

        cv_result = cross_validate(X, y, hidden_dim=config.hidden_dim, device=device, seed=args.seed, n_folds=args.n_folds)
        results[model_name] = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
            "n_samples": len(y),
            "n_invalid_smiles": invalid,
            "cv_results": cv_result,
        }

        m = cv_result["mlp"]
        r = cv_result["rf"]
        print(f"  MLP:  R2={m['r2']['mean']:.4f}±{m['r2']['std']:.4f}  RMSE={m['rmse']['mean']:.4f}±{m['rmse']['std']:.4f}")
        print(f"  RF:   R2={r['r2']['mean']:.4f}±{r['r2']['std']:.4f}  RMSE={r['rmse']['mean']:.4f}±{r['rmse']['std']:.4f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()

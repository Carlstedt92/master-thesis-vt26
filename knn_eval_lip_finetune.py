"""KNN comparison on MoleculeNet LIPO after supervised fine-tuning.

Protocol:
- Load LIPO from DeepChem with MoleculeNet random split.
- Initialize from SSL checkpoint encoder weights.
- Fine-tune on LIPO train split (validated on val split) with regression head.
- Build two feature sets per split:
  1) Fine-tuned GINE embeddings
  2) Morgan fingerprints (RDKit)
- Tune K on validation split for each feature type independently.
- Evaluate on test split and plot both methods side-by-side.
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json

import deepchem as dc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from datahandling.graph_creation import smiles_to_pygdata
from model.config import ModelConfig
from model.gnn_model import GNNModel


LIPO_SPLITTER = "random"
LIPO_DATA_DIR = "data/MoleculeNet_LIPO_custom"

SSL_MODEL_NAME = "GINE_DINO_ZINC_2.5"
CHECKPOINT_PATH = f"models/{SSL_MODEL_NAME}/checkpoints/best_online_eval_model.pth"

# Fine-tuning hyperparameters
FT_EPOCHS = 30
FT_BATCH_SIZE = 256
FT_LEARNING_RATE = 1e-4
FT_WEIGHT_DECAY = 1e-5
FT_EARLY_STOP_PATIENCE = 6

FP_RADIUS = 2
FP_NBITS = 2048
K_VALUES = [3, 5, 11, 21, 31, 41, 51]


def load_lipo_splits_from_deepchem(data_dir: str, splitter: str):
    """Load LIPO via DeepChem and return split-wise rows plus stats."""
    tasks, datasets, _ = dc.molnet.load_lipo(
        featurizer=dc.feat.RawFeaturizer(),
        splitter=splitter,
        transformers=[],
        reload=True,
        data_dir=data_dir,
        save_dir=data_dir,
    )

    train_ds, val_ds, test_ds = datasets
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds}

    rows_by_split = {}
    stats = {"task": tasks[0] if tasks else "lipo", "splitter": splitter}

    for split_name, split_ds in split_map.items():
        labels = split_ds.y.reshape(-1)
        ids = split_ds.ids
        rows = []
        skipped_non_finite = 0

        for smiles, label in zip(ids, labels):
            if not np.isfinite(label):
                skipped_non_finite += 1
                continue
            rows.append((str(smiles), float(label)))

        rows_by_split[split_name] = rows
        stats[f"n_{split_name}_deepchem"] = int(len(labels))
        stats[f"n_{split_name}_usable_labels"] = int(len(rows))
        stats[f"n_{split_name}_skipped_non_finite"] = int(skipped_non_finite)

    return rows_by_split, stats


def build_graph_dataset(rows):
    """Create list of PyG graphs with regression targets from SMILES rows."""
    graphs = []
    invalid_smiles = 0

    for smiles, target in rows:
        data = smiles_to_pygdata(smiles)
        if data is None or data.num_nodes == 0:
            invalid_smiles += 1
            continue

        data.y = torch.tensor([float(target)], dtype=torch.float32)
        graphs.append(data)

    return graphs, invalid_smiles


def load_finetune_model(device):
    """Load SSL checkpoint and initialize regression model with encoder weights."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    checkpoint_config = ModelConfig.from_dict(checkpoint["config"])

    model = GNNModel.from_config(checkpoint_config, head_type="regression").to(device)

    model_state = model.state_dict()
    encoder_only = {
        k: v for k, v in checkpoint["model_state_dict"].items() if k.startswith("encoder")
    }
    model_state.update(encoder_only)
    model.load_state_dict(model_state)

    return model, checkpoint_config


def run_finetuning(model, train_loader, val_loader, device):
    """Fine-tune model on LIPO train set with early stopping on val MSE."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FT_LEARNING_RATE,
        weight_decay=FT_WEIGHT_DECAY,
    )

    best_val_mse = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    patience = 0

    for epoch in range(1, FT_EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
            target = batch.y.view(-1).float()
            loss = F.mse_loss(preds, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
                target = batch.y.view(-1).float()
                val_loss = F.mse_loss(preds, target)
                val_losses.append(float(val_loss.item()))

        avg_train_mse = float(np.mean(train_losses)) if train_losses else float("inf")
        avg_val_mse = float(np.mean(val_losses)) if val_losses else float("inf")

        print(
            f"[Fine-tune] Epoch {epoch:02d}/{FT_EPOCHS} | "
            f"Train MSE={avg_train_mse:.4f} | Val MSE={avg_val_mse:.4f}"
        )

        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= FT_EARLY_STOP_PATIENCE:
                print(
                    f"[Fine-tune] Early stopping at epoch {epoch} "
                    f"(best epoch: {best_epoch}, best val MSE: {best_val_mse:.4f})"
                )
                break

    model.load_state_dict(best_state)
    model.eval()
    return {
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val_mse),
        "epochs_ran": int(epoch),
    }


def build_embedding_features(rows, model, device):
    """Build embeddings and labels from SMILES rows."""
    features = []
    labels = []
    invalid_smiles = 0

    with torch.no_grad():
        for smiles, target in rows:
            data = smiles_to_pygdata(smiles)
            if data is None or data.num_nodes == 0:
                invalid_smiles += 1
                continue

            data = data.to(device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            emb = model.get_embeddings(data.x, data.edge_index, data.edge_attr, batch).squeeze(0)
            features.append(emb.cpu().numpy())
            labels.append(float(target))

    return np.asarray(features), np.asarray(labels), invalid_smiles


def build_fingerprint_features(rows, radius: int = FP_RADIUS, nbits: int = FP_NBITS):
    """Build Morgan fingerprints and labels from SMILES rows."""
    features = []
    labels = []
    invalid_smiles = 0
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    for smiles, target in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles += 1
            continue

        bitvect = morgan_generator.GetFingerprint(mol)
        arr = np.zeros((nbits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bitvect, arr)
        features.append(arr)
        labels.append(float(target))

    return np.asarray(features), np.asarray(labels), invalid_smiles


def tune_and_eval_knn_regression(X_train, y_train, X_val, y_val, X_test, y_test, k_values):
    """Tune K on validation split and evaluate best model on test split."""
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

    test_metrics = {
        "r2": float(r2_score(y_test, y_test_pred)),
        "mse": float(mean_squared_error(y_test, y_test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "mae": float(mean_absolute_error(y_test, y_test_pred)),
    }
    val_metrics = {
        "r2": float(best_val_r2),
        "rmse": float(best_val_rmse),
        "mae": float(best_val_mae),
    }

    return best_k, val_metrics, test_metrics, y_test_pred


def plot_lipo_comparison(y_test, y_pred_emb, y_pred_fp, emb_metrics, fp_metrics, output_path):
    """Create one figure with side-by-side prediction comparisons."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ymin = float(min(np.min(y_test), np.min(y_pred_emb), np.min(y_pred_fp)))
    ymax = float(max(np.max(y_test), np.max(y_pred_emb), np.max(y_pred_fp)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    axes[0].scatter(y_test, y_pred_emb, alpha=0.6, s=18)
    axes[0].plot([ymin, ymax], [ymin, ymax], "k--", linewidth=1)
    axes[0].set_title(
        "Fine-tuned Embeddings\n"
        f"R2={emb_metrics['r2']:.3f}, RMSE={emb_metrics['rmse']:.3f}, MAE={emb_metrics['mae']:.3f}"
    )
    axes[0].set_xlabel("True logP")
    axes[0].set_ylabel("Predicted logP")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_test, y_pred_fp, alpha=0.6, s=18, color="tab:orange")
    axes[1].plot([ymin, ymax], [ymin, ymax], "k--", linewidth=1)
    axes[1].set_title(
        "Morgan Fingerprints\n"
        f"R2={fp_metrics['r2']:.3f}, RMSE={fp_metrics['rmse']:.3f}, MAE={fp_metrics['mae']:.3f}"
    )
    axes[1].set_xlabel("True logP")
    axes[1].grid(alpha=0.3)

    fig.suptitle("LIPO Test Set: KNN Comparison (Fine-tuned Encoder)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    rows_by_split, split_stats = load_lipo_splits_from_deepchem(
        data_dir=LIPO_DATA_DIR,
        splitter=LIPO_SPLITTER,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_finetune_model(device)

    train_graphs, ft_inv_train = build_graph_dataset(rows_by_split["train"])
    val_graphs, ft_inv_val = build_graph_dataset(rows_by_split["val"])

    if len(train_graphs) < 10 or len(val_graphs) < 10:
        raise RuntimeError(
            "Too few valid graphs for fine-tuning. "
            f"Invalid train/val: {ft_inv_train}/{ft_inv_val}."
        )

    train_loader = DataLoader(train_graphs, batch_size=FT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=FT_BATCH_SIZE, shuffle=False)

    print("Starting supervised fine-tuning on LIPO train split...")
    ft_info = run_finetuning(model, train_loader, val_loader, device)

    emb_train_X, emb_train_y, emb_inv_train = build_embedding_features(rows_by_split["train"], model, device)
    emb_val_X, emb_val_y, emb_inv_val = build_embedding_features(rows_by_split["val"], model, device)
    emb_test_X, emb_test_y, emb_inv_test = build_embedding_features(rows_by_split["test"], model, device)

    fp_train_X, fp_train_y, fp_inv_train = build_fingerprint_features(rows_by_split["train"])
    fp_val_X, fp_val_y, fp_inv_val = build_fingerprint_features(rows_by_split["val"])
    fp_test_X, fp_test_y, fp_inv_test = build_fingerprint_features(rows_by_split["test"])

    if len(emb_train_y) < 10 or len(emb_val_y) < 10 or len(emb_test_y) < 10:
        raise RuntimeError(
            "Too few valid embedding samples after graph conversion. "
            f"Invalid SMILES train/val/test: {emb_inv_train}/{emb_inv_val}/{emb_inv_test}."
        )

    if len(fp_train_y) < 10 or len(fp_val_y) < 10 or len(fp_test_y) < 10:
        raise RuntimeError(
            "Too few valid fingerprint samples after RDKit conversion. "
            f"Invalid SMILES train/val/test: {fp_inv_train}/{fp_inv_val}/{fp_inv_test}."
        )

    emb_scaler = StandardScaler()
    emb_train_X = emb_scaler.fit_transform(emb_train_X)
    emb_val_X = emb_scaler.transform(emb_val_X)
    emb_test_X = emb_scaler.transform(emb_test_X)

    fp_scaler = StandardScaler()
    fp_train_X = fp_scaler.fit_transform(fp_train_X)
    fp_val_X = fp_scaler.transform(fp_val_X)
    fp_test_X = fp_scaler.transform(fp_test_X)

    emb_best_k, emb_val_metrics, emb_test_metrics, emb_pred_test = tune_and_eval_knn_regression(
        emb_train_X,
        emb_train_y,
        emb_val_X,
        emb_val_y,
        emb_test_X,
        emb_test_y,
        K_VALUES,
    )

    fp_best_k, fp_val_metrics, fp_test_metrics, fp_pred_test = tune_and_eval_knn_regression(
        fp_train_X,
        fp_train_y,
        fp_val_X,
        fp_val_y,
        fp_test_X,
        fp_test_y,
        K_VALUES,
    )

    print(f"DeepChem splitter: {split_stats['splitter']}")
    print(
        "DeepChem split sizes (train/val/test): "
        f"{split_stats['n_train_deepchem']}/{split_stats['n_val_deepchem']}/{split_stats['n_test_deepchem']}"
    )
    print(
        f"Fine-tune summary: best_epoch={ft_info['best_epoch']}, "
        f"best_val_mse={ft_info['best_val_mse']:.4f}, epochs_ran={ft_info['epochs_ran']}"
    )

    print("\nFine-tuned embeddings")
    print(
        f"Samples used (train/val/test): {len(emb_train_y)}/{len(emb_val_y)}/{len(emb_test_y)} | "
        f"Invalid SMILES: {emb_inv_train}/{emb_inv_val}/{emb_inv_test}"
    )
    print(
        f"Best k: {emb_best_k} | "
        f"Val R2={emb_val_metrics['r2']:.4f}, RMSE={emb_val_metrics['rmse']:.4f}, MAE={emb_val_metrics['mae']:.4f} | "
        f"Test R2={emb_test_metrics['r2']:.4f}, RMSE={emb_test_metrics['rmse']:.4f}, MAE={emb_test_metrics['mae']:.4f}"
    )

    print("\nFingerprints (Morgan)")
    print(
        f"Samples used (train/val/test): {len(fp_train_y)}/{len(fp_val_y)}/{len(fp_test_y)} | "
        f"Invalid SMILES: {fp_inv_train}/{fp_inv_val}/{fp_inv_test}"
    )
    print(
        f"Best k: {fp_best_k} | "
        f"Val R2={fp_val_metrics['r2']:.4f}, RMSE={fp_val_metrics['rmse']:.4f}, MAE={fp_val_metrics['mae']:.4f} | "
        f"Test R2={fp_test_metrics['r2']:.4f}, RMSE={fp_test_metrics['rmse']:.4f}, MAE={fp_test_metrics['mae']:.4f}"
    )

    plot_path = f"models/{SSL_MODEL_NAME}/knn_lipo_finetuned_embeddings_vs_fingerprints.png"
    plot_lipo_comparison(
        y_test=emb_test_y,
        y_pred_emb=emb_pred_test,
        y_pred_fp=fp_pred_test,
        emb_metrics=emb_test_metrics,
        fp_metrics=fp_test_metrics,
        output_path=plot_path,
    )
    print(f"\nSaved comparison plot: {plot_path}")

    metadata_path = Path(f"models/{SSL_MODEL_NAME}/metadata.json")
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    metadata["KNN_eval_LIPO_finetuned"] = {
        "dataset": "LIPO",
        "data_source": "deepchem.molnet.load_lipo",
        "splitter": split_stats["splitter"],
        "task": split_stats["task"],
        "k_values": K_VALUES,
        "fingerprint": {
            "type": "Morgan",
            "radius": FP_RADIUS,
            "nbits": FP_NBITS,
        },
        "embeddings_model": SSL_MODEL_NAME,
        "finetune": {
            "head_type": "regression",
            "epochs": FT_EPOCHS,
            "batch_size": FT_BATCH_SIZE,
            "learning_rate": FT_LEARNING_RATE,
            "weight_decay": FT_WEIGHT_DECAY,
            "early_stop_patience": FT_EARLY_STOP_PATIENCE,
            "best_epoch": ft_info["best_epoch"],
            "best_val_mse": round(ft_info["best_val_mse"], 6),
            "epochs_ran": ft_info["epochs_ran"],
            "n_invalid_graphs_train": int(ft_inv_train),
            "n_invalid_graphs_val": int(ft_inv_val),
        },
        "embeddings": {
            "best_k": int(emb_best_k),
            "validation_metrics": {k: round(v, 4) for k, v in emb_val_metrics.items()},
            "test_metrics": {k: round(v, 4) for k, v in emb_test_metrics.items()},
            "n_train": int(len(emb_train_y)),
            "n_val": int(len(emb_val_y)),
            "n_test": int(len(emb_test_y)),
            "n_invalid_smiles_train": int(emb_inv_train),
            "n_invalid_smiles_val": int(emb_inv_val),
            "n_invalid_smiles_test": int(emb_inv_test),
        },
        "fingerprints": {
            "best_k": int(fp_best_k),
            "validation_metrics": {k: round(v, 4) for k, v in fp_val_metrics.items()},
            "test_metrics": {k: round(v, 4) for k, v in fp_test_metrics.items()},
            "n_train": int(len(fp_train_y)),
            "n_val": int(len(fp_val_y)),
            "n_test": int(len(fp_test_y)),
            "n_invalid_smiles_train": int(fp_inv_train),
            "n_invalid_smiles_val": int(fp_inv_val),
            "n_invalid_smiles_test": int(fp_inv_test),
        },
        "plot_path": plot_path,
        "timestamp": datetime.now().isoformat(),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Updated metadata: {metadata_path}")


if __name__ == "__main__":
    main()

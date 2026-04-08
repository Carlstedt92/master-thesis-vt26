"""KNN comparison on MoleculeNet HIV using embeddings vs Morgan fingerprints.

Protocol:
- Load HIV from DeepChem with configurable split.
- Build two feature sets per split:
  1) GINE embeddings
  2) Morgan fingerprints (RDKit)
- Tune K on validation split for each feature type independently using ROC-AUC.
- Evaluate on test split and plot side-by-side ROC curves.
"""

from datetime import datetime
from pathlib import Path
import json

import deepchem as dc
import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from datahandling.graph_creation import smiles_to_pygdata
from model.config import ModelConfig
from model.gnn_model import GNNModel


HIV_SPLITTER = "scaffold"
HIV_DATA_DIR = "data/MoleculeNet_HIV_custom"

SSL_MODEL_NAME = "GDZ_5000Epochs"
CHECKPOINT_PATH = f"models/{SSL_MODEL_NAME}/checkpoints/best_model.pth"

FP_RADIUS = 2
FP_NBITS = 2048
K_VALUES = [3, 5, 11, 21, 31, 41, 51]


def load_hiv_splits_from_deepchem(data_dir: str, splitter: str):
    """Load HIV via DeepChem and return split-wise SMILES/labels with basic stats."""
    tasks, datasets, _ = dc.molnet.load_hiv(
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
    stats = {"task": tasks[0] if tasks else "HIV_active", "splitter": splitter}

    for split_name, split_ds in split_map.items():
        labels = split_ds.y.reshape(-1)
        ids = split_ds.ids
        rows = []
        skipped_non_finite = 0
        skipped_non_binary = 0

        for smiles, label in zip(ids, labels):
            if not np.isfinite(label):
                skipped_non_finite += 1
                continue

            y = int(label)
            if y not in (0, 1):
                skipped_non_binary += 1
                continue

            rows.append((str(smiles), y))

        rows_by_split[split_name] = rows
        stats[f"n_{split_name}_deepchem"] = int(len(labels))
        stats[f"n_{split_name}_usable_labels"] = int(len(rows))
        stats[f"n_{split_name}_skipped_non_finite"] = int(skipped_non_finite)
        stats[f"n_{split_name}_skipped_non_binary"] = int(skipped_non_binary)

    return rows_by_split, stats


def build_embedding_features(rows, model, device):
    """Convert split rows into embeddings and labels, skipping invalid graphs."""
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
            labels.append(int(target))

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
        labels.append(int(target))

    return np.asarray(features), np.asarray(labels), invalid_smiles


def tune_and_eval_knn_classification(X_train, y_train, X_val, y_val, X_test, y_test, k_values):
    """Tune K on validation ROC-AUC and evaluate best model on test split."""
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

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    }
    val_metrics = {
        "roc_auc": float(best_val_roc_auc),
        "f1": float(best_val_f1),
    }

    return best_k, val_metrics, test_metrics, y_pred, y_proba


def plot_hiv_roc_comparison(
    y_test_emb,
    y_proba_emb,
    y_test_fp,
    y_proba_fp,
    emb_metrics,
    fp_metrics,
    output_path,
):
    """Create one figure with side-by-side ROC curves for embeddings and fingerprints."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fpr_emb, tpr_emb, _ = roc_curve(y_test_emb, y_proba_emb)
    fpr_fp, tpr_fp, _ = roc_curve(y_test_fp, y_proba_fp)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_emb, tpr_emb, linewidth=2, label=f"Embeddings (ROC-AUC={emb_metrics['roc_auc']:.3f})")
    plt.plot(fpr_fp, tpr_fp, linewidth=2, label=f"Morgan (ROC-AUC={fp_metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("HIV Test Set: ROC Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def main():
    rows_by_split, split_stats = load_hiv_splits_from_deepchem(
        data_dir=HIV_DATA_DIR,
        splitter=HIV_SPLITTER,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    config = ModelConfig.from_dict(checkpoint["config"])

    model = GNNModel.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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

    if len(np.unique(emb_train_y)) < 2 or len(np.unique(emb_val_y)) < 2:
        raise RuntimeError("Embeddings split has fewer than 2 classes after filtering.")

    if len(np.unique(fp_train_y)) < 2 or len(np.unique(fp_val_y)) < 2:
        raise RuntimeError("Fingerprint split has fewer than 2 classes after filtering.")

    emb_scaler = StandardScaler()
    emb_train_X = emb_scaler.fit_transform(emb_train_X)
    emb_val_X = emb_scaler.transform(emb_val_X)
    emb_test_X = emb_scaler.transform(emb_test_X)

    fp_scaler = StandardScaler()
    fp_train_X = fp_scaler.fit_transform(fp_train_X)
    fp_val_X = fp_scaler.transform(fp_val_X)
    fp_test_X = fp_scaler.transform(fp_test_X)

    emb_best_k, emb_val_metrics, emb_test_metrics, _, emb_test_proba = tune_and_eval_knn_classification(
        emb_train_X,
        emb_train_y,
        emb_val_X,
        emb_val_y,
        emb_test_X,
        emb_test_y,
        K_VALUES,
    )

    fp_best_k, fp_val_metrics, fp_test_metrics, _, fp_test_proba = tune_and_eval_knn_classification(
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

    print("\nEmbeddings")
    print(
        f"Samples used (train/val/test): {len(emb_train_y)}/{len(emb_val_y)}/{len(emb_test_y)} | "
        f"Invalid SMILES: {emb_inv_train}/{emb_inv_val}/{emb_inv_test}"
    )
    print(
        f"Best k: {emb_best_k} | "
        f"Val ROC-AUC={emb_val_metrics['roc_auc']:.4f}, F1={emb_val_metrics['f1']:.4f} | "
        f"Test ROC-AUC={emb_test_metrics['roc_auc']:.4f}, "
        f"F1={emb_test_metrics['f1']:.4f}, BAcc={emb_test_metrics['balanced_accuracy']:.4f}"
    )

    print("\nFingerprints (Morgan)")
    print(
        f"Samples used (train/val/test): {len(fp_train_y)}/{len(fp_val_y)}/{len(fp_test_y)} | "
        f"Invalid SMILES: {fp_inv_train}/{fp_inv_val}/{fp_inv_test}"
    )
    print(
        f"Best k: {fp_best_k} | "
        f"Val ROC-AUC={fp_val_metrics['roc_auc']:.4f}, F1={fp_val_metrics['f1']:.4f} | "
        f"Test ROC-AUC={fp_test_metrics['roc_auc']:.4f}, "
        f"F1={fp_test_metrics['f1']:.4f}, BAcc={fp_test_metrics['balanced_accuracy']:.4f}"
    )

    plot_path = f"models/{SSL_MODEL_NAME}/knn_hiv_embeddings_vs_fingerprints.png"
    plot_hiv_roc_comparison(
        y_test_emb=emb_test_y,
        y_proba_emb=emb_test_proba,
        y_test_fp=fp_test_y,
        y_proba_fp=fp_test_proba,
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

    metadata["KNN_eval_HIV"] = {
        "dataset": "HIV",
        "data_source": "deepchem.molnet.load_hiv",
        "splitter": split_stats["splitter"],
        "task": split_stats["task"],
        "k_values": K_VALUES,
        "evaluation_primary_metric": "roc_auc",
        "fingerprint": {
            "type": "Morgan",
            "radius": FP_RADIUS,
            "nbits": FP_NBITS,
        },
        "embeddings_model": SSL_MODEL_NAME,
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

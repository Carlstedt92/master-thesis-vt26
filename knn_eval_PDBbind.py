"""
KNN regression for GINE_DINO embeddings on PDBbind loaded via DeepChem.
This script loads a pretrained encoder, extracts one embedding per ligand,
trains a KNN regressor on a train split, and reports regression metrics.
Metrics include R2 score, Mean Squared Error, Root Mean Squared Error, and Mean Absolute Error.
"""

import numpy as np
import json
import torch
import deepchem as dc
from pathlib import Path
from datetime import datetime
from rdkit import Chem
from model.gine_model import GINEModel
from model.config import ModelConfig
from datahandling.graph_creation import smiles_to_pygdata
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


PDBBIND_SET = "core"  # Options: core, refined, general
PDBBIND_DATA_DIR = "data/deepchem_pdbbind"
PDBBIND_POCKET = True


def _get_pdbbind_folder_name(set_name: str) -> str:
    if set_name == "core":
        return "v2013-core"
    if set_name == "refined":
        return "refined-set"
    if set_name == "general":
        return "v2019-other-PL"
    raise ValueError(f"Unsupported PDBbind set_name: {set_name}")


def load_pdbbind_rows_from_deepchem(
    set_name: str,
    data_dir: str,
    pocket: bool,
) -> tuple[list[str], list[float], dict]:
    """Load PDBbind via DeepChem and extract ligand SMILES + targets.

    Returns:
        Tuple of (smiles_list, target_list, stats)
    """
    # Trigger DeepChem download/cache and obtain labels/ids.
    # We do not use DeepChem features, only IDs and targets.
    tasks, datasets, _ = dc.molnet.load_pdbbind(
        featurizer=dc.feat.RdkitGridFeaturizer(
            nb_rotations=0,
            feature_types=[],
            ecfp_power=0,
            splif_power=0,
        ),
        splitter=None,
        transformers=[],
        reload=True,
        set_name=set_name,
        pocket=pocket,
        data_dir=data_dir,
        save_dir=data_dir,
    )

    dataset = datasets[0]
    pdb_ids = [str(pdb_id) for pdb_id in dataset.ids]
    labels = dataset.y.reshape(-1)

    base_folder = Path(data_dir) / _get_pdbbind_folder_name(set_name)

    smiles_list: list[str] = []
    target_list: list[float] = []

    skipped_non_finite_targets = 0
    missing_ligand_files = 0
    invalid_ligand_mols = 0

    for pdb_id, label in zip(pdb_ids, labels):
        if not np.isfinite(label):
            skipped_non_finite_targets += 1
            continue

        ligand_sdf = base_folder / pdb_id / f"{pdb_id}_ligand.sdf"
        if not ligand_sdf.exists():
            missing_ligand_files += 1
            continue

        supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
        mol = supplier[0] if len(supplier) > 0 else None
        if mol is None:
            invalid_ligand_mols += 1
            continue

        smiles = Chem.MolToSmiles(mol)
        if not smiles:
            invalid_ligand_mols += 1
            continue

        smiles_list.append(smiles)
        target_list.append(float(label))

    stats = {
        "task": tasks[0] if tasks else "-logKd/Ki",
        "total_deepchem_samples": len(labels),
        "skipped_non_finite_targets": skipped_non_finite_targets,
        "missing_ligand_files": missing_ligand_files,
        "invalid_ligand_mols": invalid_ligand_mols,
    }

    return smiles_list, target_list, stats


# Load PDBbind rows from DeepChem
smiles_list, target_list, deepchem_stats = load_pdbbind_rows_from_deepchem(
    set_name=PDBBIND_SET,
    data_dir=PDBBIND_DATA_DIR,
    pocket=PDBBIND_POCKET,
)

# Initialize model using checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssl_model_name = "GINE_DINO_ZINC"
checkpoint = torch.load(
    f"./models/{ssl_model_name}/checkpoints/best_model.pth",
    map_location=device,
    weights_only=False,
)
config = ModelConfig.from_dict(checkpoint["config"])

model = GINEModel.from_config(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Extract embeddings and labels
features = []
labels = []
invalid_smiles = 0

with torch.no_grad():
    for smiles, target_value in zip(smiles_list, target_list):
        data = smiles_to_pygdata(smiles)
        if data is None:
            invalid_smiles += 1
            continue

        if data.num_nodes == 0:
            invalid_smiles += 1
            continue

        data = data.to(device)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        embedding = model.get_embeddings(data.x, data.edge_index, data.edge_attr, batch).squeeze(0)

        features.append(embedding.cpu().numpy())
        labels.append(target_value)

X = np.asarray(features)
y = np.asarray(labels)

if len(y) < 10:
    raise RuntimeError(
        f"Too few valid samples ({len(y)}) after filtering. "
        f"DeepChem stats: {deepchem_stats}, invalid graph SMILES: {invalid_smiles}."
    )

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tune K with cross-validation on training split
k_values = [3, 5, 11, 21, 31, 41, 51]
cv = KFold(n_splits=5, shuffle=True, random_state=42)
best_k = None
best_cv_r2 = -np.inf
best_cv_rmse = np.inf
best_cv_mae = np.inf
eps = 1e-3

for k in k_values:
    fold_r2s = []
    fold_rmses = []
    fold_maes = []
    for tr_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        knn_cv = KNeighborsRegressor(n_neighbors=k, weights="distance")
        knn_cv.fit(X_tr, y_tr)
        y_val_pred = knn_cv.predict(X_val)
        fold_r2s.append(r2_score(y_val, y_val_pred))
        fold_rmses.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        fold_maes.append(mean_absolute_error(y_val, y_val_pred))

    mean_r2 = float(np.mean(fold_r2s))
    mean_rmse = float(np.mean(fold_rmses))
    mean_mae = float(np.mean(fold_maes))
    print(f"CV (k={k}) -> R2: {mean_r2:.4f}, RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}")

    if mean_r2 > best_cv_r2 + eps:
        best_cv_r2 = mean_r2
        best_cv_rmse = mean_rmse
        best_cv_mae = mean_mae
        best_k = k
    elif abs(mean_r2 - best_cv_r2) <= eps:
        if mean_rmse < best_cv_rmse - eps:
            best_cv_rmse = mean_rmse
            best_cv_mae = mean_mae
            best_k = k
        elif abs(mean_rmse - best_cv_rmse) <= eps and mean_mae < best_cv_mae - eps:
            best_cv_mae = mean_mae
            best_k = k

# Final KNN evaluation on held-out test split
knn = KNeighborsRegressor(n_neighbors=best_k, weights="distance")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"Samples used: {len(y)}")
print(f"DeepChem total samples: {deepchem_stats['total_deepchem_samples']}")
print(f"Skipped non-finite targets: {deepchem_stats['skipped_non_finite_targets']}")
print(f"Missing ligand files: {deepchem_stats['missing_ligand_files']}")
print(f"Invalid ligand mols: {deepchem_stats['invalid_ligand_mols']}")
print(f"Invalid SMILES skipped: {invalid_smiles}")
print(
    f"Best k (CV): {best_k}  |  "
    f"CV R2: {best_cv_r2:.4f} | CV RMSE: {best_cv_rmse:.4f} | CV MAE: {best_cv_mae:.4f}"
)

test_r2 = r2_score(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = float(np.sqrt(test_mse))
test_mae = mean_absolute_error(y_test, y_pred)

print(f"Test R2: {test_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Save KNN evaluation results to metadata.json
metadata_path = f"models/{ssl_model_name}/metadata.json"
existing_metadata = {}
if Path(metadata_path).exists():
    with open(metadata_path, 'r') as f:
        existing_metadata = json.load(f)

knn_eval_data = {
    "dataset": "PDBbind",
    "data_source": "deepchem.molnet.load_pdbbind",
    "pdbbind_set": PDBBIND_SET,
    "pdbbind_task": deepchem_stats["task"],
    "pdbbind_data_dir": PDBBIND_DATA_DIR,
    "pocket": PDBBIND_POCKET,
    "evaluation_method": "KNN (k-Nearest Neighbors) with distance weights",
    "cv_folds": 5,
    "best_k": best_k,
    "cv_metrics": {
        "r2": round(best_cv_r2, 4),
        "rmse": round(best_cv_rmse, 4),
        "mae": round(best_cv_mae, 4),
    },
    "test_metrics": {
        "r2": round(test_r2, 4),
        "mse": round(test_mse, 4),
        "rmse": round(test_rmse, 4),
        "mae": round(test_mae, 4),
    },
    "n_samples": len(y),
    "n_deepchem_samples": deepchem_stats["total_deepchem_samples"],
    "n_skipped_non_finite_targets": deepchem_stats["skipped_non_finite_targets"],
    "n_missing_ligand_files": deepchem_stats["missing_ligand_files"],
    "n_invalid_ligand_mols": deepchem_stats["invalid_ligand_mols"],
    "n_invalid_smiles": invalid_smiles,
    "timestamp": datetime.now().isoformat(),
}

existing_metadata["KNN_eval_PDBbind"] = knn_eval_data

with open(metadata_path, 'w') as f:
    json.dump(existing_metadata, f, indent=2)

print(f"Updated metadata: {metadata_path}")
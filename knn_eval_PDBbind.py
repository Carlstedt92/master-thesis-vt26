"""
KNN regression for SSL embeddings on PDBbind loaded via DeepChem.

This script follows a MoleculeNet-style protocol: choose one PDBbind subset
(`core`, `refined`, or `general`), let DeepChem create train/validation/test
splits within that subset, tune k on validation, and report final test metrics.

Model input in this script is ligand-only SMILES embeddings.
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


PDBBIND_DATA_DIR = "data/deepchem_pdbbind"
PDBBIND_POCKET = True
PDBBIND_SET = "core"  # Options: core, refined, general
PDBBIND_SPLITTER = "random"


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
    """Load a full PDBbind set via DeepChem and extract ligand SMILES + targets.

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


def load_pdbbind_splits_from_deepchem(
    set_name: str,
    data_dir: str,
    pocket: bool,
    splitter: str,
) -> tuple[dict[str, tuple[list[str], list[float]]], dict]:
    """Load one PDBbind set and return train/val/test rows from DeepChem split outputs."""
    tasks, datasets, _ = dc.molnet.load_pdbbind(
        featurizer=dc.feat.RdkitGridFeaturizer(
            nb_rotations=0,
            feature_types=[],
            ecfp_power=0,
            splif_power=0,
        ),
        splitter=splitter,
        transformers=[],
        reload=True,
        set_name=set_name,
        pocket=pocket,
        data_dir=data_dir,
        save_dir=data_dir,
    )

    if len(datasets) < 3:
        raise RuntimeError(
            f"Expected train/val/test splits for set '{set_name}', got {len(datasets)} datasets."
        )

    train_dataset = datasets[0]
    val_dataset = datasets[1]
    test_dataset = datasets[2]
    base_folder = Path(data_dir) / _get_pdbbind_folder_name(set_name)

    def _rows_from_dataset(dataset) -> tuple[list[str], list[float], dict]:
        pdb_ids = [str(pdb_id) for pdb_id in dataset.ids]
        labels = dataset.y.reshape(-1)

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
            "total_deepchem_samples": len(labels),
            "skipped_non_finite_targets": skipped_non_finite_targets,
            "missing_ligand_files": missing_ligand_files,
            "invalid_ligand_mols": invalid_ligand_mols,
        }
        return smiles_list, target_list, stats

    train_smiles, train_targets, train_stats = _rows_from_dataset(train_dataset)
    val_smiles, val_targets, val_stats = _rows_from_dataset(val_dataset)
    test_smiles, test_targets, test_stats = _rows_from_dataset(test_dataset)

    stats = {
        "task": tasks[0] if tasks else "-logKd/Ki",
        "set_name": set_name,
        "splitter": splitter,
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
    }
    rows = {
        "train": (train_smiles, train_targets),
        "val": (val_smiles, val_targets),
        "test": (test_smiles, test_targets),
    }
    return rows, stats


def extract_embeddings(
    model: GINEModel,
    device: torch.device,
    smiles_list: list[str],
    target_list: list[float],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract one embedding per valid SMILES and return X, y, invalid count."""
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

    return np.asarray(features), np.asarray(labels), invalid_smiles


# Load train/validation/test splits from DeepChem within the selected PDBbind set
pdbbind_rows, pdbbind_stats = load_pdbbind_splits_from_deepchem(
    set_name=PDBBIND_SET,
    data_dir=PDBBIND_DATA_DIR,
    pocket=PDBBIND_POCKET,
    splitter=PDBBIND_SPLITTER,
)

# Initialize model using checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssl_model_name = "GINO_DINO_DELANEY"
checkpoint = torch.load(
    f"./models/{ssl_model_name}/checkpoints/best_model.pth",
    map_location=device,
    weights_only=False,
)
config = ModelConfig.from_dict(checkpoint["config"])

model = GINEModel.from_config(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Build train/val/test embeddings from selected split
train_smiles, train_targets = pdbbind_rows["train"]
val_smiles, val_targets = pdbbind_rows["val"]
test_smiles, test_targets = pdbbind_rows["test"]

X_train, y_train, invalid_smiles_train = extract_embeddings(
    model=model,
    device=device,
    smiles_list=train_smiles,
    target_list=train_targets,
)
X_val, y_val, invalid_smiles_val = extract_embeddings(
    model=model,
    device=device,
    smiles_list=val_smiles,
    target_list=val_targets,
)
X_test, y_test, invalid_smiles_test = extract_embeddings(
    model=model,
    device=device,
    smiles_list=test_smiles,
    target_list=test_targets,
)

if len(y_train) < 10 or len(y_val) < 10 or len(y_test) < 10:
    raise RuntimeError(
        "Too few valid samples after filtering. "
        f"PDBbind stats: {pdbbind_stats}; "
        f"invalid graph SMILES train/val/test: "
        f"{invalid_smiles_train}/{invalid_smiles_val}/{invalid_smiles_test}."
    )

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Tune K on validation split
k_values = [3, 5, 11, 21, 31, 41, 51]
best_k = None
best_val_r2 = -np.inf
best_val_rmse = np.inf
best_val_mae = np.inf
eps = 1e-3

for k in k_values:
    knn_val = KNeighborsRegressor(n_neighbors=k, weights="distance")
    knn_val.fit(X_train, y_train)
    y_val_pred = knn_val.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    val_mae = mean_absolute_error(y_val, y_val_pred)

    print(f"Validation (k={k}) -> R2: {val_r2:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")

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

# Final KNN evaluation on held-out test split
knn = KNeighborsRegressor(n_neighbors=best_k, weights="distance")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"Protocol: MoleculeNet-style train/val/test split within PDBbind {PDBBIND_SET}")
print(
    f"DeepChem split sizes (train/val/test): "
    f"{pdbbind_stats['train']['total_deepchem_samples']}/"
    f"{pdbbind_stats['val']['total_deepchem_samples']}/"
    f"{pdbbind_stats['test']['total_deepchem_samples']}"
)
print(f"Samples used (train/val/test): {len(y_train)}/{len(y_val)}/{len(y_test)}")
print(
    f"Invalid SMILES skipped (train/val/test): "
    f"{invalid_smiles_train}/{invalid_smiles_val}/{invalid_smiles_test}"
)
print(
    f"Best k (validation): {best_k}  |  "
    f"Val R2: {best_val_r2:.4f} | Val RMSE: {best_val_rmse:.4f} | Val MAE: {best_val_mae:.4f}"
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
    "protocol": "single_set_train_val_test_split",
    "pdbbind_set": PDBBIND_SET,
    "splitter": PDBBIND_SPLITTER,
    "pdbbind_task": pdbbind_stats["task"],
    "pdbbind_data_dir": PDBBIND_DATA_DIR,
    "pocket": PDBBIND_POCKET,
    "evaluation_method": "KNN (k-Nearest Neighbors) with distance weights",
    "best_k": best_k,
    "validation_metrics": {
        "r2": round(best_val_r2, 4),
        "rmse": round(best_val_rmse, 4),
        "mae": round(best_val_mae, 4),
    },
    "test_metrics": {
        "r2": round(test_r2, 4),
        "mse": round(test_mse, 4),
        "rmse": round(test_rmse, 4),
        "mae": round(test_mae, 4),
    },
    "n_deepchem_train": pdbbind_stats["train"]["total_deepchem_samples"],
    "n_deepchem_val": pdbbind_stats["val"]["total_deepchem_samples"],
    "n_deepchem_test": pdbbind_stats["test"]["total_deepchem_samples"],
    "n_train": int(len(y_train)),
    "n_val": int(len(y_val)),
    "n_test": int(len(y_test)),
    "n_invalid_smiles_train": int(invalid_smiles_train),
    "n_invalid_smiles_val": int(invalid_smiles_val),
    "n_invalid_smiles_test": int(invalid_smiles_test),
    "timestamp": datetime.now().isoformat(),
}

existing_metadata["KNN_eval_PDBbind"] = knn_eval_data

with open(metadata_path, 'w') as f:
    json.dump(existing_metadata, f, indent=2)

print(f"Updated metadata: {metadata_path}")
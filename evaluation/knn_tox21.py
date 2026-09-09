"""KNN comparison on MoleculeNet Tox21 using embeddings vs Morgan fingerprints.

Protocol:
- Load Tox21 from DeepChem with configurable split (default random).
- Build two feature sets per split:
  1) SSL graph embeddings
  2) Morgan fingerprints (RDKit)
- Evaluate all tasks and report mean/std ROC-AUC across tasks.
- Save per-task metrics plus aggregate summary.
"""

import argparse
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
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from datahandling.graph_creation import smiles_to_pygdata
from model.config import ModelConfig
from model.gnn_model import GNNModel


TOX21_SPLITTER = "random"
TOX21_DATA_DIR = "data/MoleculeNet_Tox21_custom"

SSL_MODEL_NAME = "GDZ_5000Epochs"
CHECKPOINT_PATH = None

FP_RADIUS = 2
FP_NBITS = 2048
K_VALUES = [3, 5, 11, 21, 31, 41, 51]


def resolve_checkpoint_path(ssl_model_name: str, checkpoint_path: str | None = None) -> str:
    """Resolve checkpoint path with fallback to best_model.pth."""
    if checkpoint_path:
        return checkpoint_path

    checkpoint_dir = Path(f"models/{ssl_model_name}/checkpoints")
    fallback = checkpoint_dir / "best_model.pth"
    alternative = checkpoint_dir / "best_online_eval_model.pth"

    if fallback.exists():
        return str(fallback)
    if alternative.exists():
        return str(alternative)

    raise FileNotFoundError(
        f"No checkpoint found for {ssl_model_name}. Tried:\n"
        f"  - {fallback}\n"
        f"  - {alternative}"
    )


def resolve_torch_device(device_preference: str = "auto") -> torch.device:
    """Resolve torch device; auto mode falls back to CPU if CUDA init fails."""
    if device_preference == "cpu":
        return torch.device("cpu")

    if device_preference == "cuda":
        return torch.device("cuda")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        _ = torch.empty(1, device="cuda")
        return torch.device("cuda")
    except Exception as exc:
        print(f"Warning: CUDA unavailable at runtime ({exc}). Falling back to CPU.")
        return torch.device("cpu")


def infer_graph_featurization(config):
    """Infer graph featurization flags from checkpoint config when needed."""
    explicit_hydrogens = getattr(config, "explicit_hydrogens", None)
    encode_hydrogen_count = getattr(config, "encode_hydrogen_count", None)

    if explicit_hydrogens is not None and encode_hydrogen_count is not None:
        return bool(explicit_hydrogens), bool(encode_hydrogen_count)

    num_features = int(getattr(config, "num_features", 24))
    if num_features == 20:
        return False, False
    if num_features == 24:
        return True, False
    if num_features == 25:
        return False, True

    raise ValueError(
        "Cannot infer graph featurization from config. "
        f"Explicit flags missing and num_features={num_features} is unsupported."
    )


def load_tox21_splits_from_deepchem(data_dir: str, splitter: str, split_seed: int | None = None):
    """Load Tox21 and return split-wise smiles/labels with task names and stats."""
    save_dir = data_dir
    if splitter == "random" and split_seed is not None:
        save_dir = str(Path(data_dir) / "seeded_splits" / f"random_seed_{split_seed}")

    tasks, datasets, _ = dc.molnet.load_tox21(
        featurizer=dc.feat.RawFeaturizer(),
        splitter=splitter,
        seed=split_seed,
        transformers=[],
        reload=True,
        data_dir=data_dir,
        save_dir=save_dir,
    )

    split_map = {"train": datasets[0], "val": datasets[1], "test": datasets[2]}
    data_by_split = {}
    stats = {
        "splitter": splitter,
        "split_seed": split_seed,
        "n_tasks": int(len(tasks)),
        "task_names": [str(task) for task in tasks],
    }

    for split_name, split_ds in split_map.items():
        labels = split_ds.y
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        ids = np.asarray(split_ds.ids).astype(str)
        data_by_split[split_name] = {
            "smiles": ids,
            "labels": labels,
        }

        stats[f"n_{split_name}_deepchem"] = int(len(ids))

    return data_by_split, stats


def build_embedding_features(
    smiles_array,
    model,
    device,
    explicit_hydrogens: bool = True,
    encode_hydrogen_count: bool = False,
    use_extended_features: bool = False,
    scale_eccentricity: bool = False,
):
    """Build embeddings and return kept row indices into the source array."""
    features = []
    kept_indices = []
    invalid_smiles = 0

    with torch.no_grad():
        for idx, smiles in enumerate(smiles_array):
            data = smiles_to_pygdata(
                str(smiles),
                use_extended_features=use_extended_features,
                scale_eccentricity=scale_eccentricity,
                explicit_hydrogens=explicit_hydrogens,
                encode_hydrogen_count=encode_hydrogen_count,
            )
            if data is None or data.num_nodes == 0:
                invalid_smiles += 1
                continue

            data = data.to(device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            emb = model.get_embeddings(data.x, data.edge_index, data.edge_attr, batch).squeeze(0)
            features.append(emb.cpu().numpy())
            kept_indices.append(idx)

    return np.asarray(features), np.asarray(kept_indices, dtype=np.int64), invalid_smiles


def build_fingerprint_features(smiles_array, radius: int = FP_RADIUS, nbits: int = FP_NBITS):
    """Build Morgan fingerprints and return kept row indices into the source array."""
    features = []
    kept_indices = []
    invalid_smiles = 0
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    for idx, smiles in enumerate(smiles_array):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            invalid_smiles += 1
            continue

        bitvect = morgan_generator.GetFingerprint(mol)
        arr = np.zeros((nbits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bitvect, arr)
        features.append(arr)
        kept_indices.append(idx)

    return np.asarray(features), np.asarray(kept_indices, dtype=np.int64), invalid_smiles


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

    return int(best_k), val_metrics, test_metrics


def _evaluate_task(
    task_name,
    task_index,
    emb_split_data,
    fp_split_data,
):
    """Evaluate one Tox21 task for embeddings and fingerprints."""
    def _prepare(split_data):
        labels = split_data["labels"][:, task_index]
        finite_mask = np.isfinite(labels)
        labels = labels[finite_mask]
        X = split_data["X"][finite_mask]
        y = labels.astype(int)

        binary_mask = np.isin(y, [0, 1])
        X = X[binary_mask]
        y = y[binary_mask]
        return X, y

    emb_train_X, emb_train_y = _prepare(emb_split_data["train"])
    emb_val_X, emb_val_y = _prepare(emb_split_data["val"])
    emb_test_X, emb_test_y = _prepare(emb_split_data["test"])

    fp_train_X, fp_train_y = _prepare(fp_split_data["train"])
    fp_val_X, fp_val_y = _prepare(fp_split_data["val"])
    fp_test_X, fp_test_y = _prepare(fp_split_data["test"])

    if min(len(emb_train_y), len(emb_val_y), len(emb_test_y)) < 10:
        return None, f"Embeddings task {task_name} has too few samples after filtering."

    if min(len(fp_train_y), len(fp_val_y), len(fp_test_y)) < 10:
        return None, f"Fingerprints task {task_name} has too few samples after filtering."

    if len(np.unique(emb_train_y)) < 2 or len(np.unique(emb_val_y)) < 2 or len(np.unique(emb_test_y)) < 2:
        return None, f"Embeddings task {task_name} has <2 classes in one split after filtering."

    if len(np.unique(fp_train_y)) < 2 or len(np.unique(fp_val_y)) < 2 or len(np.unique(fp_test_y)) < 2:
        return None, f"Fingerprints task {task_name} has <2 classes in one split after filtering."

    emb_scaler = StandardScaler()
    emb_train_X = emb_scaler.fit_transform(emb_train_X)
    emb_val_X = emb_scaler.transform(emb_val_X)
    emb_test_X = emb_scaler.transform(emb_test_X)

    fp_scaler = StandardScaler()
    fp_train_X = fp_scaler.fit_transform(fp_train_X)
    fp_val_X = fp_scaler.transform(fp_val_X)
    fp_test_X = fp_scaler.transform(fp_test_X)

    emb_best_k, emb_val_metrics, emb_test_metrics = tune_and_eval_knn_classification(
        emb_train_X,
        emb_train_y,
        emb_val_X,
        emb_val_y,
        emb_test_X,
        emb_test_y,
        K_VALUES,
    )
    fp_best_k, fp_val_metrics, fp_test_metrics = tune_and_eval_knn_classification(
        fp_train_X,
        fp_train_y,
        fp_val_X,
        fp_val_y,
        fp_test_X,
        fp_test_y,
        K_VALUES,
    )

    return {
        "task": task_name,
        "task_index": int(task_index),
        "embeddings": {
            "best_k": int(emb_best_k),
            "validation_metrics": emb_val_metrics,
            "test_metrics": emb_test_metrics,
            "n_train": int(len(emb_train_y)),
            "n_val": int(len(emb_val_y)),
            "n_test": int(len(emb_test_y)),
        },
        "fingerprints": {
            "best_k": int(fp_best_k),
            "validation_metrics": fp_val_metrics,
            "test_metrics": fp_test_metrics,
            "n_train": int(len(fp_train_y)),
            "n_val": int(len(fp_val_y)),
            "n_test": int(len(fp_test_y)),
        },
    }, None


def plot_tox21_mean_performance(emb_mean, emb_std, fp_mean, fp_std, output_path):
    """Plot mean ROC-AUC across tasks with std error bars."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.bar(
        [0, 1],
        [emb_mean, fp_mean],
        yerr=[emb_std, fp_std],
        capsize=5,
        color=["tab:blue", "tab:orange"],
    )
    plt.xticks([0, 1], ["Embeddings", "Fingerprints"])
    plt.ylabel("ROC-AUC")
    plt.ylim(0.0, 1.0)
    plt.title("Tox21 Mean ROC-AUC Across Tasks")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def run_knn_eval_tox21(
    ssl_model_name: str = SSL_MODEL_NAME,
    checkpoint_path: str | None = CHECKPOINT_PATH,
    tox21_data_dir: str = TOX21_DATA_DIR,
    tox21_splitter: str = TOX21_SPLITTER,
    split_seed: int | None = None,
    device_preference: str = "auto",
    save_plot: bool = True,
):
    
    data_by_split, split_stats = load_tox21_splits_from_deepchem(
        data_dir=tox21_data_dir,
        splitter=tox21_splitter,
        split_seed=split_seed,
    )

    checkpoint_path = resolve_checkpoint_path(ssl_model_name, checkpoint_path)
    device = resolve_torch_device(device_preference)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint["config"])
    explicit_hydrogens, encode_hydrogen_count = infer_graph_featurization(config)

    model = GNNModel.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    emb_split_data = {}
    fp_split_data = {}

    for split_name in ("train", "val", "test"):
        smiles = data_by_split[split_name]["smiles"]
        labels = data_by_split[split_name]["labels"]

        emb_X, emb_kept_idx, emb_invalid = build_embedding_features(
            smiles,
            model,
            device,
            explicit_hydrogens=explicit_hydrogens,
            encode_hydrogen_count=encode_hydrogen_count,
        )
        fp_X, fp_kept_idx, fp_invalid = build_fingerprint_features(smiles)

        emb_split_data[split_name] = {
            "X": emb_X,
            "labels": labels[emb_kept_idx],
            "n_invalid_smiles": int(emb_invalid),
        }
        fp_split_data[split_name] = {
            "X": fp_X,
            "labels": labels[fp_kept_idx],
            "n_invalid_smiles": int(fp_invalid),
        }

    per_task_results = []
    skipped_tasks = []
    for task_index, task_name in enumerate(split_stats["task_names"]):
        result, skip_reason = _evaluate_task(task_name, task_index, emb_split_data, fp_split_data)
        if result is None:
            skipped_tasks.append({"task": task_name, "task_index": int(task_index), "reason": skip_reason})
            continue
        per_task_results.append(result)

    if not per_task_results:
        raise RuntimeError("No Tox21 tasks could be evaluated after filtering.")

    emb_val_aucs = np.asarray([r["embeddings"]["validation_metrics"]["roc_auc"] for r in per_task_results], dtype=float)
    emb_test_aucs = np.asarray([r["embeddings"]["test_metrics"]["roc_auc"] for r in per_task_results], dtype=float)
    fp_val_aucs = np.asarray([r["fingerprints"]["validation_metrics"]["roc_auc"] for r in per_task_results], dtype=float)
    fp_test_aucs = np.asarray([r["fingerprints"]["test_metrics"]["roc_auc"] for r in per_task_results], dtype=float)

    emb_val_mean = float(np.mean(emb_val_aucs))
    emb_test_mean = float(np.mean(emb_test_aucs))
    fp_val_mean = float(np.mean(fp_val_aucs))
    fp_test_mean = float(np.mean(fp_test_aucs))

    emb_val_std = float(np.std(emb_val_aucs, ddof=0))
    emb_test_std = float(np.std(emb_test_aucs, ddof=0))
    fp_val_std = float(np.std(fp_val_aucs, ddof=0))
    fp_test_std = float(np.std(fp_test_aucs, ddof=0))

    print(f"DeepChem splitter: {split_stats['splitter']}")
    if split_stats.get("split_seed") is not None:
        print(f"DeepChem split seed: {split_stats['split_seed']}")
    print(
        "Embedding featurization: "
        f"explicit_hydrogens={explicit_hydrogens}, "
        f"encode_hydrogen_count={encode_hydrogen_count}, "
        f"num_features={getattr(config, 'num_features', 'unknown')}"
    )
    print(
        "DeepChem split sizes (train/val/test): "
        f"{split_stats['n_train_deepchem']}/{split_stats['n_val_deepchem']}/{split_stats['n_test_deepchem']}"
    )
    print(f"Evaluated tasks: {len(per_task_results)} / {split_stats['n_tasks']}")
    if skipped_tasks:
        print(f"Skipped tasks: {len(skipped_tasks)}")

    print("\nTox21 mean ROC-AUC across tasks")
    print(
        f"Embeddings: val={emb_val_mean:.4f} +/- {emb_val_std:.4f} | "
        f"test={emb_test_mean:.4f} +/- {emb_test_std:.4f}"
    )
    print(
        f"Fingerprints: val={fp_val_mean:.4f} +/- {fp_val_std:.4f} | "
        f"test={fp_test_mean:.4f} +/- {fp_test_std:.4f}"
    )

    split_seed = split_stats.get("split_seed")
    seed_suffix = f"_seed{split_seed}" if split_seed is not None else ""
    plot_path = None
    if save_plot:
        plot_path = f"models/{ssl_model_name}/knn_tox21_mean_rocauc_embeddings_vs_fingerprints{seed_suffix}.png"
        plot_tox21_mean_performance(
            emb_mean=emb_test_mean,
            emb_std=emb_test_std,
            fp_mean=fp_test_mean,
            fp_std=fp_test_std,
            output_path=plot_path,
        )
        print(f"\nSaved comparison plot: {plot_path}")

    metadata_path = Path(f"models/{ssl_model_name}/metadata.json")
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    metadata_key = "KNN_eval_Tox21" if split_seed is None else f"KNN_eval_Tox21_seed{split_seed}"

    # Save structured eval log under the model folder and write a pointer into metadata
    model_dir = Path(f"models/{ssl_model_name}")
    try:
        model_eval_dir = model_dir / "eval_logs" / "knn" / "TOX21"
        model_eval_dir.mkdir(parents=True, exist_ok=True)
        eval_filename = f"knn_tox21{seed_suffix}.json"
        eval_file_rel = Path("eval_logs") / "knn" / "TOX21" / eval_filename
        eval_entry = {
            "result": {
                "dataset": "Tox21",
                "splitter": split_stats["splitter"],
                "split_seed": split_stats.get("split_seed"),
                "embeddings": {"validation_metrics": {"roc_auc_mean_tasks": emb_val_mean, "roc_auc_std_tasks": emb_val_std}, "test_metrics": {"roc_auc_mean_tasks": emb_test_mean, "roc_auc_std_tasks": emb_test_std}},
                "fingerprints": {"validation_metrics": {"roc_auc_mean_tasks": fp_val_mean, "roc_auc_std_tasks": fp_val_std}, "test_metrics": {"roc_auc_mean_tasks": fp_test_mean, "roc_auc_std_tasks": fp_test_std}},
            },
            "metadata": split_stats,
            "per_task": per_task_results,
            "skipped_tasks": skipped_tasks,
        }
        with open(model_eval_dir / eval_filename, "w") as f:
            json.dump(eval_entry, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

        metadata[metadata_key] = {"eval_log": str(eval_file_rel)}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Wrote eval log and updated metadata pointer: {metadata_path} -> {eval_file_rel}")
    except Exception:
        metadata[metadata_key] = {"note": "eval log unavailable"}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    # Also save a per-evaluation JSON log under eval_logs/knn/TOX21/{model}/
    try:
        eval_dir = Path(f"eval_logs/knn/TOX21/{ssl_model_name}")
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_file = eval_dir / f"knn_tox21{seed_suffix}.json"
        eval_entry = {
            "result": {
                "dataset": "Tox21",
                "splitter": split_stats["splitter"],
                "split_seed": split_stats.get("split_seed"),
                "embeddings": {
                    "test_metrics": {"roc_auc_mean_tasks": emb_test_mean, "roc_auc_std_tasks": emb_test_std},
                    "validation_metrics": {"roc_auc_mean_tasks": emb_val_mean, "roc_auc_std_tasks": emb_val_std},
                },
                "fingerprints": {
                    "test_metrics": {"roc_auc_mean_tasks": fp_test_mean, "roc_auc_std_tasks": fp_test_std},
                    "validation_metrics": {"roc_auc_mean_tasks": fp_val_mean, "roc_auc_std_tasks": fp_val_std},
                },
            },
            "metadata": split_stats,
        }
        with open(eval_file, "w") as f:
            json.dump(eval_entry, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    except Exception:
        pass

    return {
        "dataset": "Tox21",
        "splitter": split_stats["splitter"],
        "split_seed": split_stats.get("split_seed"),
        "primary_metric": "roc_auc_mean_tasks",
        "embeddings": {
            "best_k": int(round(np.mean([r["embeddings"]["best_k"] for r in per_task_results]))),
            "validation_metrics": {
                "roc_auc_mean_tasks": emb_val_mean,
                "roc_auc_std_tasks": emb_val_std,
            },
            "test_metrics": {
                "roc_auc_mean_tasks": emb_test_mean,
                "roc_auc_std_tasks": emb_test_std,
            },
        },
        "fingerprints": {
            "best_k": int(round(np.mean([r["fingerprints"]["best_k"] for r in per_task_results]))),
            "validation_metrics": {
                "roc_auc_mean_tasks": fp_val_mean,
                "roc_auc_std_tasks": fp_val_std,
            },
            "test_metrics": {
                "roc_auc_mean_tasks": fp_test_mean,
                "roc_auc_std_tasks": fp_test_std,
            },
        },
        "per_task": per_task_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Tox21 KNN eval for one SSL model.")
    parser.add_argument("--model-name", type=str, default=SSL_MODEL_NAME)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--tox21-data-dir", type=str, default=TOX21_DATA_DIR)
    parser.add_argument("--splitter", type=str, default=TOX21_SPLITTER)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    run_knn_eval_tox21(
        ssl_model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        tox21_data_dir=args.tox21_data_dir,
        tox21_splitter=args.splitter,
        split_seed=args.split_seed,
        device_preference=args.device,
    )


if __name__ == "__main__":
    main()

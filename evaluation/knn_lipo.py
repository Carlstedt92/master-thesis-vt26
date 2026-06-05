"""KNN comparison on MoleculeNet LIPO using embeddings vs Morgan fingerprints.

Protocol:
- Load LIPO from DeepChem with MoleculeNet random split.
- Build two feature sets per split:
  1) GINE_DINO_ZINC graph embeddings
  2) Morgan fingerprints (RDKit)
- Tune K on validation split for each feature type independently.
- Evaluate on test split and plot both methods side-by-side.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from datahandling.graph_creation import smiles_to_pygdata
from model.config import ModelConfig
from model.gnn_model import GNNModel


LIPO_SPLITTER = "random"
LIPO_DATA_DIR = "data/MoleculeNet_LIPO_custom"

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
        # Trigger CUDA context init to catch environment/device errors early.
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


def load_lipo_splits_from_deepchem(data_dir: str, splitter: str, split_seed: int | None = None):
    """Load LIPO via DeepChem and return split-wise rows plus stats."""
    save_dir = data_dir
    if splitter == "random" and split_seed is not None:
        save_dir = str(Path(data_dir) / "seeded_splits" / f"random_seed_{split_seed}")

    tasks, datasets, _ = dc.molnet.load_lipo(
        featurizer=dc.feat.RawFeaturizer(),
        splitter=splitter,
        seed=split_seed,
        transformers=[],
        reload=True,
        data_dir=data_dir,
        save_dir=save_dir,
    )

    train_ds, val_ds, test_ds = datasets
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds}

    rows_by_split = {}
    stats = {"task": tasks[0] if tasks else "lipo", "splitter": splitter, "split_seed": split_seed}

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


def build_embedding_features(
    rows,
    model,
    device,
    explicit_hydrogens: bool = True,
    encode_hydrogen_count: bool = False,
):
    """Build SSL embeddings and labels from SMILES rows."""
    features = []
    labels = []
    invalid_smiles = 0

    with torch.no_grad():
        for smiles, target in rows:
            data = smiles_to_pygdata(
                smiles,
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


def plot_lipo_comparison(
    y_test,
    y_pred_emb,
    y_pred_fp,
    emb_metrics,
    fp_metrics,
    output_path,
):
    """Create one figure with side-by-side prediction comparisons."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ymin = float(min(np.min(y_test), np.min(y_pred_emb), np.min(y_pred_fp)))
    ymax = float(max(np.max(y_test), np.max(y_pred_emb), np.max(y_pred_fp)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    axes[0].scatter(y_test, y_pred_emb, alpha=0.6, s=18)
    axes[0].plot([ymin, ymax], [ymin, ymax], "k--", linewidth=1)
    axes[0].set_title(
        "Embeddings (GINE_DINO_ZINC)\n"
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

    fig.suptitle("LIPO Test Set: KNN Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def run_knn_eval_lipo(
    ssl_model_name: str = SSL_MODEL_NAME,
    checkpoint_path: str | None = CHECKPOINT_PATH,
    lipo_data_dir: str = LIPO_DATA_DIR,
    lipo_splitter: str = LIPO_SPLITTER,
    split_seed: int | None = None,
    device_preference: str = "auto",
    save_plot: bool = True,
):
    rows_by_split, split_stats = load_lipo_splits_from_deepchem(
        data_dir=lipo_data_dir,
        splitter=lipo_splitter,
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

    emb_train_X, emb_train_y, emb_inv_train = build_embedding_features(
        rows_by_split["train"],
        model,
        device,
        explicit_hydrogens=explicit_hydrogens,
        encode_hydrogen_count=encode_hydrogen_count,
    )
    emb_val_X, emb_val_y, emb_inv_val = build_embedding_features(
        rows_by_split["val"],
        model,
        device,
        explicit_hydrogens=explicit_hydrogens,
        encode_hydrogen_count=encode_hydrogen_count,
    )
    emb_test_X, emb_test_y, emb_inv_test = build_embedding_features(
        rows_by_split["test"],
        model,
        device,
        explicit_hydrogens=explicit_hydrogens,
        encode_hydrogen_count=encode_hydrogen_count,
    )

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
    print(f"\nEmbeddings ({ssl_model_name})")
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

    split_seed = split_stats.get("split_seed")
    seed_suffix = f"_seed{split_seed}" if split_seed is not None else ""
    plot_path = None
    if save_plot:
        plot_path = f"models/{ssl_model_name}/knn_lipo_embeddings_vs_fingerprints{seed_suffix}.png"
        plot_lipo_comparison(
            y_test=emb_test_y,
            y_pred_emb=emb_pred_test,
            y_pred_fp=fp_pred_test,
            emb_metrics=emb_test_metrics,
            fp_metrics=fp_test_metrics,
            output_path=plot_path,
        )
        print(f"\nSaved comparison plot: {plot_path}")

    metadata_path = Path(f"models/{ssl_model_name}/metadata.json")
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    metadata_key = "KNN_eval_LIPO" if split_seed is None else f"KNN_eval_LIPO_seed{split_seed}"

    # Save structured eval log under the model folder and write a pointer into metadata
    model_dir = Path(f"models/{ssl_model_name}")
    try:
        model_eval_dir = model_dir / "eval_logs" / "knn" / "LIPO"
        model_eval_dir.mkdir(parents=True, exist_ok=True)
        eval_filename = f"knn_lipo{seed_suffix}.json"
        eval_file_rel = Path("eval_logs") / "knn" / "LIPO" / eval_filename
        # write eval file
        eval_entry = {
            "result": {
                "dataset": "LIPO",
                "splitter": split_stats["splitter"],
                "split_seed": split_stats.get("split_seed"),
                "embeddings": {
                    "test_metrics": emb_test_metrics,
                    "validation_metrics": emb_val_metrics,
                    "best_k": int(emb_best_k),
                },
                "fingerprints": {
                    "test_metrics": fp_test_metrics,
                    "validation_metrics": fp_val_metrics,
                    "best_k": int(fp_best_k),
                },
            },
            "metadata": split_stats,
        }
        with open(model_eval_dir / eval_filename, "w") as f:
            json.dump(eval_entry, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

        # update metadata with a lightweight pointer to the eval log
        metadata[metadata_key] = {"eval_log": str(eval_file_rel)}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Wrote eval log and updated metadata pointer: {metadata_path} -> {eval_file_rel}")
    except Exception:
        # best-effort: if writing eval log fails, still keep previous metadata behavior minimal
        metadata[metadata_key] = {"note": "eval log unavailable"}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    # Also save a per-evaluation JSON log under eval_logs/knn/LIPO/{model}/
    try:
        eval_dir = Path(f"eval_logs/knn/LIPO/{ssl_model_name}")
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_file = eval_dir / f"knn_lipo{seed_suffix}.json"
        eval_entry = {
            "result": {
                "dataset": "LIPO",
                "splitter": split_stats["splitter"],
                "split_seed": split_stats.get("split_seed"),
                "embeddings": {
                    "test_metrics": emb_test_metrics,
                    "validation_metrics": emb_val_metrics,
                    "best_k": int(emb_best_k),
                },
                "fingerprints": {
                    "test_metrics": fp_test_metrics,
                    "validation_metrics": fp_val_metrics,
                    "best_k": int(fp_best_k),
                },
            },
            "metadata": split_stats,
        }
        with open(eval_file, "w") as f:
            json.dump(eval_entry, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    except Exception:
        pass

    return {
        "dataset": "LIPO",
        "splitter": split_stats["splitter"],
        "split_seed": split_stats.get("split_seed"),
        "primary_metric": "rmse",
        "embeddings": {
            "best_k": int(emb_best_k),
            "validation_metrics": emb_val_metrics,
            "test_metrics": emb_test_metrics,
        },
        "fingerprints": {
            "best_k": int(fp_best_k),
            "validation_metrics": fp_val_metrics,
            "test_metrics": fp_test_metrics,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run LIPO KNN eval for one SSL model.")
    parser.add_argument("--model-name", type=str, default=SSL_MODEL_NAME)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--lipo-data-dir", type=str, default=LIPO_DATA_DIR)
    parser.add_argument("--splitter", type=str, default=LIPO_SPLITTER)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    run_knn_eval_lipo(
        ssl_model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        lipo_data_dir=args.lipo_data_dir,
        lipo_splitter=args.splitter,
        split_seed=args.split_seed,
        device_preference=args.device,
    )


if __name__ == "__main__":
    main()

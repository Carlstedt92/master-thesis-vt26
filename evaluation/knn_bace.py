"""KNN comparison on MoleculeNet BACE using embeddings vs Morgan fingerprints.

Protocol:
- Load BACE from DeepChem with configurable split.
- Build two feature sets per split:
  1) SSL graph embeddings
  2) Morgan fingerprints (RDKit)
- Tune K on validation split for each feature type independently using ROC-AUC.
- Evaluate on test split and plot side-by-side ROC curves.
"""

import argparse
from datetime import datetime
from pathlib import Path
import json

import deepchem as dc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


BACE_SPLITTER = "scaffold"  # briefly switched to "random" (5-seed variance, vs. scaffold's single fixed
# split with no variance estimate) but reverted -- every existing BACE result in the project (every model
# evaluated so far except NODE_RECON_TEST_60EP) was computed under scaffold, and switching the default
# broke direct comparability against them. Pass splitter="random" explicitly for a real variance estimate
# when that's worth more than comparability against prior results.
BACE_DATA_DIR = "data/MoleculeNet_BACE_custom"

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


def load_bace_splits_from_deepchem(data_dir: str, splitter: str, split_seed: int | None = None):
    """Load BACE and return split-wise rows + stats."""
    save_dir = data_dir
    if splitter == "random" and split_seed is not None:
        save_dir = str(Path(data_dir) / "seeded_splits" / f"random_seed_{split_seed}")

    tasks, datasets, _ = dc.molnet.load_bace_classification(
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
    stats = {"task": tasks[0] if tasks else "Class", "splitter": splitter}

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


_ADMET_BENCHMARK_SEEDS = (1, 2, 3, 4, 5)  # same official seeds as evaluation/tdc_datasets.py's
# admet_benchmark protocol -- kept as the same tuple/name convention deliberately.

_bace_fixed_test_cache = None


def load_bace_admet_benchmark_splits(data_dir: str, split_seed: int):
    """BACE, TDC-admet_benchmark-style: a FIXED test set (identical across all
    5 seeds) plus a seeded scaffold reshuffle of the remaining train+val pool.

    DeepChem's own ScaffoldSplitter accepts a `seed` argument but never
    actually uses it anywhere in ScaffoldSplitter.split() (verified by
    reading the source) -- calling load_bace_splits_from_deepchem with
    different seeds silently returns the IDENTICAL split every time, which is
    exactly why BACE has had zero seed variance all along. TDC's own scaffold
    splitter (tdc.utils.split.create_scaffold_split) DOES use the seed --
    random.shuffle on the scaffold groups -- so this reuses that actual
    function (a real project dependency already, not reimplemented here)
    instead of DeepChem's inert one.

    The fixed test set is DeepChem's original scaffold-split test set --
    reused as-is (cached process-wide, since it's deterministic and
    seed-independent) so every BACE result already computed this session
    stays comparable to whatever runs under this new protocol; only how
    train/val get built changes.
    """
    if split_seed not in _ADMET_BENCHMARK_SEEDS:
        raise ValueError(
            f"BACE admet_benchmark splitter requires split_seed in {_ADMET_BENCHMARK_SEEDS} "
            f"(matching the TDC datasets' official leaderboard seeds), got {split_seed}."
        )

    global _bace_fixed_test_cache
    if _bace_fixed_test_cache is None:
        fixed_rows, _ = load_bace_splits_from_deepchem(data_dir, "scaffold", split_seed=None)
        _bace_fixed_test_cache = fixed_rows["test"]
        _bace_pool_rows = fixed_rows["train"] + fixed_rows["val"]
        globals()["_bace_pool_rows_cache"] = _bace_pool_rows
    fixed_test = _bace_fixed_test_cache
    pool_rows = globals()["_bace_pool_rows_cache"]

    from tdc.utils.split import create_scaffold_split
    pool_df = pd.DataFrame(pool_rows, columns=["Drug", "Y"])
    split = create_scaffold_split(pool_df, seed=split_seed, frac=[0.875, 0.125, 0.0], entity="Drug")

    def _rows_from_df(df):
        return [(str(row.Drug), int(row.Y)) for row in df.itertuples(index=False)]

    rows_by_split = {
        "train": _rows_from_df(split["train"]),
        "val": _rows_from_df(split["valid"]),
        "test": fixed_test,
    }
    stats = {
        "task": "Class",
        "splitter": "admet_benchmark",
        "split_seed": split_seed,
        "n_train": len(rows_by_split["train"]),
        "n_val": len(rows_by_split["val"]),
        "n_test": len(rows_by_split["test"]),
    }
    return rows_by_split, stats


def build_embedding_features(
    rows,
    model,
    device,
    explicit_hydrogens: bool = True,
    encode_hydrogen_count: bool = False,
    use_extended_features: bool = False,
    scale_eccentricity: bool = False,
):
    """Convert split rows into embeddings and labels, skipping invalid graphs."""
    features = []
    labels = []
    invalid_smiles = 0

    with torch.no_grad():
        for smiles, target in rows:
            data = smiles_to_pygdata(
                smiles,
                explicit_hydrogens=explicit_hydrogens,
                encode_hydrogen_count=encode_hydrogen_count,
                use_extended_features=use_extended_features,
                scale_eccentricity=scale_eccentricity,
            )
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

    return best_k, val_metrics, test_metrics, y_proba


def plot_bace_roc_comparison(
    y_test_emb,
    y_proba_emb,
    y_test_fp,
    y_proba_fp,
    emb_metrics,
    fp_metrics,
    output_path,
):
    """Create ROC comparison plot for embeddings vs fingerprints."""
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
    plt.title("BACE Test Set: ROC Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def run_knn_eval_bace(
    ssl_model_name: str = SSL_MODEL_NAME,
    checkpoint_path: str | None = CHECKPOINT_PATH,
    bace_data_dir: str = BACE_DATA_DIR,
    bace_splitter: str = BACE_SPLITTER,
    split_seed: int | None = None,
    device_preference: str = "auto",
    save_plot: bool = True,
):
    rows_by_split, split_stats = load_bace_splits_from_deepchem(
        data_dir=bace_data_dir,
        splitter=bace_splitter,
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

    emb_best_k, emb_val_metrics, emb_test_metrics, emb_test_proba = tune_and_eval_knn_classification(
        emb_train_X,
        emb_train_y,
        emb_val_X,
        emb_val_y,
        emb_test_X,
        emb_test_y,
        K_VALUES,
    )

    fp_best_k, fp_val_metrics, fp_test_metrics, fp_test_proba = tune_and_eval_knn_classification(
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
        "Embedding featurization: "
        f"explicit_hydrogens={explicit_hydrogens}, "
        f"encode_hydrogen_count={encode_hydrogen_count}, "
        f"num_features={getattr(config, 'num_features', 'unknown')}"
    )
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

    plot_path = None
    if save_plot:
        plot_path = f"models/{ssl_model_name}/knn_bace_embeddings_vs_fingerprints.png"
        plot_bace_roc_comparison(
            y_test_emb=emb_test_y,
            y_proba_emb=emb_test_proba,
            y_test_fp=fp_test_y,
            y_proba_fp=fp_test_proba,
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

    seed_suffix = f"_seed{split_seed}" if split_seed is not None else ""

    # Save per-eval log under the model folder and write a pointer into metadata
    model_dir = Path(f"models/{ssl_model_name}")
    try:
        model_eval_dir = model_dir / "eval_logs" / "knn" / "BACE"
        model_eval_dir.mkdir(parents=True, exist_ok=True)
        eval_filename = f"knn_bace{seed_suffix}.json"
        eval_file_rel = Path("eval_logs") / "knn" / "BACE" / eval_filename
        eval_entry = {
            "result": {
                "dataset": "BACE",
                "splitter": split_stats["splitter"],
                "task": split_stats["task"],
                "embeddings": {"best_k": int(emb_best_k), "validation_metrics": emb_val_metrics, "test_metrics": emb_test_metrics},
                "fingerprints": {"best_k": int(fp_best_k), "validation_metrics": fp_val_metrics, "test_metrics": fp_test_metrics},
            },
            "metadata": split_stats,
        }
        with open(model_eval_dir / eval_filename, "w") as f:
            json.dump(eval_entry, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

        metadata["KNN_eval_BACE"] = {"eval_log": str(eval_file_rel)}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Wrote eval log and updated metadata pointer: {metadata_path} -> {eval_file_rel}")
    except Exception:
        metadata["KNN_eval_BACE"] = {"note": "eval log unavailable"}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return {
        "dataset": "BACE",
        "splitter": split_stats["splitter"],
        "split_seed": split_stats.get("split_seed"),
        "primary_metric": "roc_auc",
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
    parser = argparse.ArgumentParser(description="Run BACE KNN eval for one SSL model.")
    parser.add_argument("--model-name", type=str, default=SSL_MODEL_NAME)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--bace-data-dir", type=str, default=BACE_DATA_DIR)
    parser.add_argument("--splitter", type=str, default=BACE_SPLITTER)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    run_knn_eval_bace(
        ssl_model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        bace_data_dir=args.bace_data_dir,
        bace_splitter=args.splitter,
        device_preference=args.device,
    )


if __name__ == "__main__":
    main()

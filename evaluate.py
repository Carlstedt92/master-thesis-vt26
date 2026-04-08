"""Run end-to-end kNN and linear-probe evaluation for checkpoints.

Fingerprints are cached once per dataset, while embeddings are extracted
transiently for each checkpoint and never written to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import deepchem as dc
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from datahandling.graph_creation import smiles_to_pygdata
from evaluation import (
    evaluate_knn_classification,
    evaluate_knn_regression,
    evaluate_linear_probe_classification,
    evaluate_linear_probe_regression,
)
from model.config import ModelConfig
from model.gnn_model import GNNModel


DATASET_CONFIG = {
    "lipo": {
        "loader": dc.molnet.load_lipo,
        "splitter": "random",
        "data_dir": "data/MoleculeNet_LIPO_custom",
        "task": "regression",
    },
    "hiv": {
        "loader": dc.molnet.load_hiv,
        "splitter": "scaffold",
        "data_dir": "data/MoleculeNet_HIV_custom",
        "task": "classification",
    },
}


def _parse_csv(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _load_rows(dataset_name: str):
    cfg = DATASET_CONFIG[dataset_name]
    tasks, datasets, _ = cfg["loader"](
        featurizer=dc.feat.RawFeaturizer(),
        splitter=cfg["splitter"],
        transformers=[],
        reload=True,
        data_dir=cfg["data_dir"],
        save_dir=cfg["data_dir"],
    )

    split_map = {"train": datasets[0], "val": datasets[1], "test": datasets[2]}
    rows_by_split = {}
    stats = {"dataset": dataset_name, "task": tasks[0] if tasks else dataset_name, "splitter": cfg["splitter"]}

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

            if cfg["task"] == "classification":
                label_value = int(label)
                if label_value not in (0, 1):
                    skipped_non_binary += 1
                    continue
            else:
                label_value = float(label)

            rows.append((str(smiles), label_value))

        rows_by_split[split_name] = rows
        stats[f"n_{split_name}"] = int(len(rows))
        stats[f"n_{split_name}_skipped_non_finite"] = int(skipped_non_finite)
        stats[f"n_{split_name}_skipped_non_binary"] = int(skipped_non_binary)

    return rows_by_split, stats


def _fingerprint_cache_path(dataset_name: str) -> Path:
    return Path(f"evaluation_cache/{dataset_name}_morgan.npz")


def _build_fingerprints(rows_by_split, radius: int = 2, nbits: int = 2048):
    arrays = {}
    metadata = {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    for split_name, rows in rows_by_split.items():
        features = []
        labels = []
        invalid_smiles = 0

        for smiles, target in rows:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_smiles += 1
                continue

            bitvect = generator.GetFingerprint(mol)
            arr = np.zeros((nbits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bitvect, arr)
            features.append(arr)
            labels.append(target)

        arrays[f"{split_name}_X"] = np.asarray(features)
        arrays[f"{split_name}_y"] = np.asarray(labels)
        metadata[f"n_{split_name}_invalid_smiles"] = int(invalid_smiles)

    return arrays, metadata


def _load_or_build_fingerprints(dataset_name: str, rows_by_split, radius: int, nbits: int):
    cache_path = _fingerprint_cache_path(dataset_name)
    metadata_path = cache_path.with_suffix(".json")

    if cache_path.exists() and metadata_path.exists():
        print(f"  ✓ Fingerprint cache hit: {cache_path}")
        archive = np.load(cache_path, allow_pickle=True)
        feature_set = {
            "train_X": archive["train_X"],
            "train_y": archive["train_y"],
            "val_X": archive["val_X"],
            "val_y": archive["val_y"],
            "test_X": archive["test_X"],
            "test_y": archive["test_y"],
        }
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return feature_set, metadata, False

    print(f"  • Building fingerprint cache: {cache_path}")
    arrays, metadata = _build_fingerprints(rows_by_split, radius=radius, nbits=nbits)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    metadata.update(
        {
            "dataset": dataset_name,
            "feature_type": "fingerprints",
            "fingerprint_radius": int(radius),
            "fingerprint_nbits": int(nbits),
            "output_path": str(cache_path),
        }
    )
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    feature_set = {
        "train_X": arrays["train_X"],
        "train_y": arrays["train_y"],
        "val_X": arrays["val_X"],
        "val_y": arrays["val_y"],
        "test_X": arrays["test_X"],
        "test_y": arrays["test_y"],
    }
    return feature_set, metadata, True


def _load_checkpoint_paths(checkpoints_dir: str):
    directory = Path(checkpoints_dir)
    checkpoint_paths = sorted(directory.glob("*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    best_model = [path for path in checkpoint_paths if path.name == "best_model.pth"]
    regular = [path for path in checkpoint_paths if path.name != "best_model.pth"]
    return best_model + regular


def _select_top_n_by_ssl_loss(checkpoint_paths, top_n: int):
    """Select top-N checkpoints by lowest stored SSL loss in checkpoint payload."""
    scored = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        loss_value = checkpoint.get("loss", float("inf"))
        try:
            loss_value = float(loss_value)
        except (TypeError, ValueError):
            loss_value = float("inf")
        scored.append((loss_value, checkpoint_path.name, checkpoint_path))

    scored.sort(key=lambda item: (item[0], item[1]))
    selected = scored[:top_n]

    print(f"\nSelecting top {len(selected)} checkpoints by lowest SSL loss:")
    for rank, (loss_value, _, checkpoint_path) in enumerate(selected, start=1):
        print(f"  {rank:02d}. {checkpoint_path.name} | loss={loss_value:.6f}")

    return [item[2] for item in selected]


def _evaluate_feature_set(feature_set, task: str, k_values, alphas):
    def _strip_prediction_arrays(result_dict: dict):
        # Keep scalar metrics in the summary and drop large ndarray payloads.
        result_dict.pop("test_predictions", None)
        result_dict.pop("test_probabilities", None)
        return result_dict

    if task == "regression":
        knn_result = evaluate_knn_regression(
            feature_set["train_X"],
            feature_set["train_y"],
            feature_set["val_X"],
            feature_set["val_y"],
            feature_set["test_X"],
            feature_set["test_y"],
            k_values=k_values,
        )
        probe_result = evaluate_linear_probe_regression(
            feature_set["train_X"],
            feature_set["train_y"],
            feature_set["val_X"],
            feature_set["val_y"],
            feature_set["test_X"],
            feature_set["test_y"],
            alphas=alphas,
        )
        return {
            "knn": _strip_prediction_arrays(knn_result),
            "linear_probe": _strip_prediction_arrays(probe_result),
        }

    if task == "classification":
        knn_result = evaluate_knn_classification(
            feature_set["train_X"],
            feature_set["train_y"],
            feature_set["val_X"],
            feature_set["val_y"],
            feature_set["test_X"],
            feature_set["test_y"],
            k_values=k_values,
        )
        probe_result = evaluate_linear_probe_classification(
            feature_set["train_X"],
            feature_set["train_y"],
            feature_set["val_X"],
            feature_set["val_y"],
            feature_set["test_X"],
            feature_set["test_y"],
            Cs=alphas,
        )
        return {
            "knn": _strip_prediction_arrays(knn_result),
            "linear_probe": _strip_prediction_arrays(probe_result),
        }

    raise ValueError(f"Unsupported task: {task}")


def _build_embeddings_from_model(rows_by_split, model: torch.nn.Module, device: torch.device):
    arrays = {}
    for split_name, rows in rows_by_split.items():
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
                embedding = model.get_embeddings(data.x, data.edge_index, data.edge_attr, batch).squeeze(0)
                features.append(embedding.cpu().numpy())
                labels.append(target)

        arrays[f"{split_name}_X"] = np.asarray(features)
        arrays[f"{split_name}_y"] = np.asarray(labels)
        arrays[f"{split_name}_invalid_smiles"] = int(invalid_smiles)

    feature_set = {}
    embedding_stats = {}
    for split_name in ("train", "val", "test"):
        x_key = f"{split_name}_X"
        y_key = f"{split_name}_y"
        invalid_key = f"{split_name}_invalid_smiles"

        if x_key in arrays and y_key in arrays:
            feature_set[x_key] = arrays[x_key]
            feature_set[y_key] = arrays[y_key]
        if invalid_key in arrays:
            embedding_stats[f"n_{split_name}_invalid_smiles"] = int(arrays[invalid_key])

    return feature_set, embedding_stats


def _build_embeddings(rows_by_split, checkpoint_path: Path, device: torch.device):
    print(f"  • Extracting embeddings from {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint["config"])
    model = GNNModel.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return _build_embeddings_from_model(rows_by_split, model, device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints end to end with cached fingerprints.")
    parser.add_argument("--checkpoints-dir", required=True, help="Directory containing model checkpoints.")
    parser.add_argument(
        "--datasets",
        default="lipo,hiv",
        help="Comma-separated datasets to evaluate (lipo,hiv).",
    )
    parser.add_argument(
        "--task",
        choices=("regression", "classification", "auto"),
        default="auto",
        help="Task type; auto uses each dataset's default.",
    )
    parser.add_argument(
        "--k-values",
        default="3,5,11,21,31,41,51",
        help="Comma-separated k values for kNN.",
    )
    parser.add_argument(
        "--alphas",
        default="0.01,0.1,1,10",
        help="Comma-separated regularization values for linear probing.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save a JSON summary. Defaults to models/<model_name>/evaluation/results.json.",
    )
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-nbits", type=int, default=2048)
    parser.add_argument(
        "--top-n-by-ssl-loss",
        type=int,
        default=None,
        help="Evaluate only the top-N checkpoints with the lowest stored SSL loss.",
    )

    args = parser.parse_args()
    k_values = _parse_csv(args.k_values, int)
    alphas = _parse_csv(args.alphas, float)

    checkpoint_paths = _load_checkpoint_paths(args.checkpoints_dir)
    if args.top_n_by_ssl_loss is not None:
        if args.top_n_by_ssl_loss <= 0:
            raise ValueError("--top-n-by-ssl-loss must be a positive integer.")
        checkpoint_paths = _select_top_n_by_ssl_loss(checkpoint_paths, args.top_n_by_ssl_loss)

    dataset_names = _parse_csv(args.datasets, str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary = {
        "checkpoints_dir": str(args.checkpoints_dir),
        "datasets": {},
    }

    for dataset_name in dataset_names:
        print(f"\n=== Dataset: {dataset_name} ===")
        rows_by_split, dataset_stats = _load_rows(dataset_name)
        fingerprint_features, fingerprint_stats, fingerprint_built = _load_or_build_fingerprints(
            dataset_name,
            rows_by_split,
            radius=args.fingerprint_radius,
            nbits=args.fingerprint_nbits,
        )

        dataset_task = DATASET_CONFIG[dataset_name]["task"] if args.task == "auto" else args.task
        checkpoint_results = {}
        total_checkpoints = len(checkpoint_paths)

        # Evaluate fingerprints once per dataset
        print(f"  • Evaluating Morgan fingerprints baseline...")
        fingerprint_result = _evaluate_feature_set(fingerprint_features, dataset_task, k_values, alphas)
        print(f"    ✓ Fingerprints | best_k={fingerprint_result['knn'].get('best_k')}, best_alpha={fingerprint_result['linear_probe'].get('best_alpha')}")

        # Evaluate each checkpoint's embeddings
        print(f"  • Checkpoints to evaluate: {total_checkpoints}")
        for checkpoint_index, checkpoint_path in enumerate(checkpoint_paths, start=1):
            print(f"  [{checkpoint_index}/{total_checkpoints}] {checkpoint_path.name}")
            embeddings_features, embedding_stats = _build_embeddings(rows_by_split, checkpoint_path, device)

            checkpoint_result = {
                "embeddings": {
                    **embedding_stats,
                    **_evaluate_feature_set(embeddings_features, dataset_task, k_values, alphas),
                }
            }

            checkpoint_results[checkpoint_path.name] = checkpoint_result

            emb_best = checkpoint_result["embeddings"]["knn"].get("best_k")
            print(f"    ✓ Done | best_k={emb_best}")

        summary["datasets"][dataset_name] = {
            **dataset_stats,
            "fingerprints": {
                **fingerprint_stats,
                **fingerprint_result,
            },
            "fingerprints_cache_path": str(_fingerprint_cache_path(dataset_name)),
            "fingerprints_cache_built": bool(fingerprint_built),
            "checkpoints": checkpoint_results,
        }

    print(json.dumps(summary, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved summary: {output_path}")
    else:
        model_name = Path(args.checkpoints_dir).parent.name
        default_output = Path(f"models/{model_name}/evaluation/results.json")
        default_output.parent.mkdir(parents=True, exist_ok=True)
        with open(default_output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved summary: {default_output}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()

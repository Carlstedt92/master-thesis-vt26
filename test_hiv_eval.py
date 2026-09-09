"""
Standalone, one-off HIV (MoleculeNet) eval -- NOT wired into
eval_many_models_mlp_rf.py. Reuses the exact same MLP/RF/fingerprint
machinery (evaluation/mlp_rf.py) so numbers are directly comparable to the
LIPO/Tox21/BACE results already produced by the real pipeline, but keeps
this as a quick standalone check before deciding whether HIV is worth
properly integrating (evaluation/knn_hiv.py + wiring into
eval_many_models_mlp_rf.py).

HIV: ~41k molecules, single binary task (replication inhibition), scaffold
split (same convention as BACE). Severely class-imbalanced (~3.5% actives)
-- reporting ROC-AUC, F1, and balanced accuracy, not just ROC-AUC alone.
"""

import argparse
from pathlib import Path

import deepchem as dc
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from datahandling.graph_creation import smiles_to_pygdata
from evaluation.mlp_rf import evaluate_mlp_classification, evaluate_rf_classification
from evaluation.knn_bace import resolve_checkpoint_path, resolve_torch_device, infer_graph_featurization
from model.config import ModelConfig
from model.gnn_model import GNNModel

HIV_DATA_DIR = "data/MoleculeNet_HIV_custom"
FP_RADIUS = 2
FP_NBITS = 2048


def load_hiv_splits(data_dir: str, splitter: str = "scaffold"):
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
        for smiles, label in zip(ids, labels):
            if not np.isfinite(label):
                continue
            y = int(label)
            if y not in (0, 1):
                continue
            rows.append((str(smiles), y))
        rows_by_split[split_name] = rows
        n_pos = sum(1 for _, y in rows if y == 1)
        stats[f"n_{split_name}"] = len(rows)
        stats[f"n_{split_name}_pos"] = n_pos
        stats[f"n_{split_name}_pos_frac"] = n_pos / len(rows) if rows else 0.0
    return rows_by_split, stats


def build_embedding_features(rows, model, device, explicit_hydrogens, encode_hydrogen_count,
                              use_extended_features=False, scale_eccentricity=False):
    features, labels = [], []
    invalid = 0
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
                invalid += 1
                continue
            data = data.to(device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            emb = model.get_embeddings(data.x, data.edge_index, data.edge_attr, batch).squeeze(0)
            features.append(emb.cpu().numpy())
            labels.append(int(target))
    return np.asarray(features), np.asarray(labels), invalid


def build_fingerprint_features(rows, radius=FP_RADIUS, nbits=FP_NBITS):
    features, labels = [], []
    invalid = 0
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    for smiles, target in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid += 1
            continue
        bitvect = gen.GetFingerprint(mol)
        arr = np.zeros((nbits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bitvect, arr)
        features.append(arr)
        labels.append(int(target))
    return np.asarray(features), np.asarray(labels), invalid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="KHOP_GAT3HEADS_TEST_60EP")
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
    print(f"Device: {device}")

    checkpoint_path = None
    if args.checkpoint_name:
        checkpoint_path = str(Path(f"models/{args.model_name}/checkpoints") / args.checkpoint_name)
    checkpoint_path = resolve_checkpoint_path(args.model_name, checkpoint_path)
    print(f"Model: {args.model_name}  Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint["config"])
    explicit_h, encode_h = infer_graph_featurization(config)
    use_extended = bool(getattr(config, "use_extended_features", False))
    scale_ecc = bool(getattr(config, "scale_eccentricity", False))
    checkpoint_epoch = checkpoint.get("epoch")
    print(f"Checkpoint epoch: {checkpoint_epoch}")

    model = GNNModel.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nLoading HIV (scaffold split)...")
    rows_by_split, stats = load_hiv_splits(HIV_DATA_DIR, splitter="scaffold")
    print(f"Split sizes (train/val/test): {stats['n_train']}/{stats['n_val']}/{stats['n_test']}")
    print(f"Positive fraction (train/val/test): "
          f"{stats['n_train_pos_frac']:.4f}/{stats['n_val_pos_frac']:.4f}/{stats['n_test_pos_frac']:.4f}")

    print("\nBuilding embeddings...")
    emb = {}
    for split in ("train", "val", "test"):
        X, y, inv = build_embedding_features(
            rows_by_split[split], model, device, explicit_h, encode_h, use_extended, scale_ecc
        )
        emb[split] = (X, y)
        print(f"  {split}: {len(y)} usable, {inv} invalid SMILES")

    print("\nBuilding Morgan fingerprints...")
    fp = {}
    for split in ("train", "val", "test"):
        X, y, inv = build_fingerprint_features(rows_by_split[split])
        fp[split] = (X, y)
        print(f"  {split}: {len(y)} usable, {inv} invalid SMILES")

    print("\nTraining/evaluating MLP + RF on embeddings and fingerprints...")
    mlp_emb = evaluate_mlp_classification(*emb["train"], *emb["val"], *emb["test"], device)
    rf_emb = evaluate_rf_classification(*emb["train"], *emb["val"], *emb["test"], seed=args.seed)
    mlp_fp = evaluate_mlp_classification(*fp["train"], *fp["val"], *fp["test"], device)
    rf_fp = evaluate_rf_classification(*fp["train"], *fp["val"], *fp["test"], seed=args.seed)

    print(f"\n=== HIV results for {args.model_name} (epoch {checkpoint_epoch}) ===")
    for label, result in (("MLP(emb)", mlp_emb), ("RF(emb)", rf_emb), ("MLP(fp)", mlp_fp), ("RF(fp)", rf_fp)):
        m = result["test_metrics"]
        print(f"  {label}: ROC-AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}"
              + (f"  BAcc={m.get('balanced_accuracy'):.4f}" if 'balanced_accuracy' in m else ""))


if __name__ == "__main__":
    main()

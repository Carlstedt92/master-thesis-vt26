"""Phase 1 of a two-phase MLP/RF eval: GPU-only embedding extraction.

eval_many_models_mlp_rf.py interleaves GPU work (per-molecule embedding
extraction via model.get_embeddings()) with heavy CPU-only work (Morgan
fingerprint generation, sklearn RF fitting) inside the same job -- the
long CPU-bound stretches between brief GPU calls make aggregate GPU
utilization low across the whole job's wall-clock time, which is suspected
to have triggered Berzelius's auto-cancellation of low-utilization jobs
(job 17376620, a 6-model sweep, was CANCELLED after ~64 min with no error,
matching the earlier finetuning-job cancellation's signature exactly).

This phase does ONLY the part that strictly needs the loaded encoder model:
building embedding features (train/val/test) for each (model, dataset, seed)
and saving them to disk. Phase 2 (eval_phase2_fit_probes.py) loads these,
builds fingerprints (once per dataset+seed, not per model -- they don't
depend on the model at all), and fits the MLP/RF probes, entirely without
needing the GPU for anything but a small, dense MLP training loop.
"""

import argparse
from pathlib import Path

import numpy as np

from evaluation.knn_bace import (
    build_embedding_features as build_bace_embedding_features,
    load_bace_admet_benchmark_splits,
    load_bace_splits_from_deepchem,
)
from evaluation.knn_lipo import (
    build_embedding_features as build_lipo_embedding_features,
    load_lipo_splits_from_deepchem,
    resolve_checkpoint_path,
    resolve_torch_device,
    infer_graph_featurization,
)
from evaluation.knn_tox21 import build_embedding_features as build_tox21_embedding_features, load_tox21_splits_from_deepchem
from evaluation.tdc_datasets import (
    build_embedding_features as build_tdc_embedding_features,
    load_tdc_admet_benchmark_splits,
    load_tdc_splits_from_tdc,
)
from model.config import ModelConfig
from model.gnn_model import GNNModel

import torch


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint["config"])
    explicit_h, encode_h = infer_graph_featurization(config)
    use_extended_features = bool(getattr(config, "use_extended_features", False))
    scale_eccentricity = bool(getattr(config, "scale_eccentricity", False))
    model = GNNModel.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, explicit_h, encode_h, use_extended_features, scale_eccentricity


_TDC_DATASETS = ("bbb_martins", "herg", "ames")


def extract_and_save(model_name, checkpoint_path, checkpoint_name, device, dataset, seed,
                      lipo_data_dir, lipo_splitter, bace_data_dir, bace_splitter,
                      tox21_data_dir, tox21_splitter, tdc_data_dirs, tdc_splitters, cache_dir):
    resolved_checkpoint = checkpoint_path
    if not resolved_checkpoint and checkpoint_name:
        resolved_checkpoint = str(Path(f"models/{model_name}/checkpoints") / checkpoint_name)
    resolved_checkpoint = resolve_checkpoint_path(model_name, resolved_checkpoint)

    model, config, explicit_h, encode_h, use_ext, scale_ecc = load_model(resolved_checkpoint, device)

    if dataset == "lipo":
        rows_by_split, _ = load_lipo_splits_from_deepchem(lipo_data_dir, lipo_splitter, split_seed=seed)
        arrays = {}
        for split in ("train", "val", "test"):
            X, y, invalid = build_lipo_embedding_features(rows_by_split[split], model, device, explicit_h, encode_h, use_ext, scale_ecc)
            arrays[f"{split}_X"], arrays[f"{split}_y"] = X, y
    elif dataset == "bace":
        if bace_splitter == "admet_benchmark":
            rows_by_split, _ = load_bace_admet_benchmark_splits(bace_data_dir, split_seed=seed)
        else:
            rows_by_split, _ = load_bace_splits_from_deepchem(bace_data_dir, bace_splitter, split_seed=seed)
        arrays = {}
        for split in ("train", "val", "test"):
            X, y, invalid = build_bace_embedding_features(rows_by_split[split], model, device, explicit_h, encode_h, use_ext, scale_ecc)
            arrays[f"{split}_X"], arrays[f"{split}_y"] = X, y
    elif dataset == "tox21":
        data_by_split, _ = load_tox21_splits_from_deepchem(tox21_data_dir, tox21_splitter, split_seed=seed)
        arrays = {}
        for split in ("train", "val", "test"):
            smiles = data_by_split[split]["smiles"]
            labels = data_by_split[split]["labels"]
            X, kept, invalid = build_tox21_embedding_features(smiles, model, device, explicit_h, encode_h, use_ext, scale_ecc)
            arrays[f"{split}_X"], arrays[f"{split}_y"] = X, labels[kept]
    elif dataset in _TDC_DATASETS:
        if tdc_splitters[dataset] == "admet_benchmark":
            rows_by_split, _ = load_tdc_admet_benchmark_splits(dataset, tdc_data_dirs[dataset], split_seed=seed)
        else:
            rows_by_split, _ = load_tdc_splits_from_tdc(dataset, tdc_data_dirs[dataset], tdc_splitters[dataset], split_seed=seed)
        arrays = {}
        for split in ("train", "val", "test"):
            X, y, invalid = build_tdc_embedding_features(rows_by_split[split], model, device, explicit_h, encode_h, use_ext, scale_ecc)
            arrays[f"{split}_X"], arrays[f"{split}_y"] = X, y
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    seed_suffix = f"seed{seed}" if seed is not None else "seedNone"
    out_path = Path(cache_dir) / model_name / f"{dataset}_{seed_suffix}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    print(f"  Saved {out_path} ({', '.join(f'{k}={v.shape}' for k, v in arrays.items())})")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: GPU-only embedding extraction for the MLP/RF eval.")
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--random-split-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--datasets", type=str, default="lipo,bace,tox21")
    parser.add_argument("--lipo-data-dir", type=str, default="data/MoleculeNet_LIPO_custom")
    parser.add_argument("--lipo-splitter", type=str, default="random")
    parser.add_argument("--bace-data-dir", type=str, default="data/MoleculeNet_BACE_custom")
    parser.add_argument("--bace-splitter", type=str, default="admet_benchmark")
    parser.add_argument("--tox21-data-dir", type=str, default="data/MoleculeNet_Tox21_custom")
    parser.add_argument("--tox21-splitter", type=str, default="random")
    parser.add_argument("--bbb-martins-data-dir", type=str, default="data/TDC_BBB_Martins_custom")
    parser.add_argument("--bbb-martins-splitter", type=str, default="admet_benchmark")
    parser.add_argument("--herg-data-dir", type=str, default="data/TDC_hERG_custom")
    parser.add_argument("--herg-splitter", type=str, default="admet_benchmark")
    parser.add_argument("--ames-data-dir", type=str, default="data/TDC_AMES_custom")
    parser.add_argument("--ames-splitter", type=str, default="admet_benchmark")
    parser.add_argument("--cache-dir", type=str, default="embedding_cache")
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
    print(f"Device: {device}")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.random_split_seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    splitters = {
        "lipo": args.lipo_splitter, "bace": args.bace_splitter, "tox21": args.tox21_splitter,
        "bbb_martins": args.bbb_martins_splitter, "herg": args.herg_splitter, "ames": args.ames_splitter,
    }
    tdc_data_dirs = {"bbb_martins": args.bbb_martins_data_dir, "herg": args.herg_data_dir, "ames": args.ames_data_dir}
    tdc_splitters = {"bbb_martins": args.bbb_martins_splitter, "herg": args.herg_splitter, "ames": args.ames_splitter}

    for model_name in model_names:
        print(f"\n=== Extracting embeddings for {model_name} ===")
        for dataset in datasets:
            splitter = splitters[dataset]
            # admet_benchmark ALWAYS uses TDC's official 5 leaderboard seeds (1-5), regardless of
            # --random-split-seeds -- the fixed test set is identical across all of them, only
            # train/valid gets reshuffled per seed, and TDC's own evaluate_many() requires exactly
            # these 5 runs for a leaderboard-comparable result (see evaluation/tdc_datasets.py).
            if splitter == "admet_benchmark":
                dataset_seeds = [1, 2, 3, 4, 5]
            elif splitter == "random":
                dataset_seeds = seeds
            else:
                dataset_seeds = [None]
            for seed in dataset_seeds:
                print(f"  {dataset} seed={seed}")
                extract_and_save(
                    model_name, args.checkpoint_path, args.checkpoint_name, device, dataset, seed,
                    args.lipo_data_dir, args.lipo_splitter, args.bace_data_dir, args.bace_splitter,
                    args.tox21_data_dir, args.tox21_splitter, tdc_data_dirs, tdc_splitters, args.cache_dir,
                )

    print("\nPhase 1 complete. Run eval_phase2_fit_probes.py next.")


if __name__ == "__main__":
    main()

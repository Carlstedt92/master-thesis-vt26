"""Phase 1 of a two-phase finetuning eval: GPU-only encoder finetuning.

Split into two phases because the original combined finetune_eval_many_models.py
mixed GPU-heavy finetuning with CPU-only KNN/linear-probe/RF hyperparameter
sweeps in the same job -- the GPU sat idle for extended stretches during the
sweep phases, which likely triggered Berzelius's auto-cancellation of
low-GPU-utilization jobs (job 17363293 was killed by SIGTERM after 1:11:33,
not a timeout -- confirmed via sacct).

This phase does ONLY the GPU-dense work (finetune encoder + task head on
train data, validation-selected best epoch) and saves each finetuned
checkpoint to disk. Phase 2 (finetune_phase2_eval.py) loads those checkpoints
and does the CPU-heavy MLP/RF evaluation separately, matching the same
methodology used everywhere else this session (evaluation/mlp_rf.py -- not
KNN/linear-probe, per explicit instruction to keep this comparable to the
frozen-embedding results already produced).
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from datahandling.graph_creation import smiles_to_pygdata
from evaluation.knn_lipo import infer_graph_featurization
from evaluation.knn_bace import load_bace_admet_benchmark_splits
from evaluation.tdc_datasets import load_tdc_admet_benchmark_splits
from finetune_eval_many_models import (
    resolve_checkpoint_path,
    load_model_for_finetuning,
    finetune_model,
    get_dataset_splits,
    make_graph_loader,
)
from model.gnn_model import ProjectionHead

TDC_DATASETS = ("bbb_martins", "herg", "ames")


def finetune_and_save(model_name, checkpoint_path, checkpoint_name, device, dataset, seed,
                       finetune_epochs, batch_size, lr, weight_decay,
                       lipo_data_dir, tox21_data_dir, bace_data_dir, tdc_data_dirs, out_dir):
    # resolve_checkpoint_path's own default priority (best_online_eval_model.pth
    # first) is the OPPOSITE of the best_model.pth (SSL-val-loss-selected)
    # convention used everywhere else this session -- checkpoint_name must be
    # passed explicitly (default "best_model.pth" in main()) to avoid silently
    # picking the wrong one.
    resolved_checkpoint = resolve_checkpoint_path(model_name, checkpoint_path, checkpoint_name)

    if dataset == "tox21":
        from evaluation.knn_tox21 import load_tox21_splits_from_deepchem

        splits, _ = load_tox21_splits_from_deepchem(tox21_data_dir, "random", split_seed=seed)
        model, config = load_model_for_finetuning(resolved_checkpoint, device, "tox21")
        explicit_h, encode_h = infer_graph_featurization(config)

        labels_train = splits["train"]["labels"]
        labels_val = splits["val"]["labels"]
        num_tasks = labels_train.shape[1]

        def build_multi(smiles_rows, labels_arr):
            data_list = []
            for i, smiles in enumerate(smiles_rows):
                data = smiles_to_pygdata(str(smiles), explicit_hydrogens=explicit_h, encode_hydrogen_count=encode_h)
                if data is None or data.num_nodes == 0:
                    continue
                data.y = torch.tensor(labels_arr[i], dtype=torch.float32)
                data_list.append(data)
            if not data_list:
                raise RuntimeError("No valid Tox21 graphs after conversion.")
            return DataLoader(data_list, batch_size=batch_size, shuffle=True)

        train_loader = build_multi(splits["train"]["smiles"], labels_train)
        val_loader = build_multi(splits["val"]["smiles"], labels_val)
        test_loader = build_multi(splits["test"]["smiles"], splits["test"]["labels"])

        # Multi-task linear head -- matches finetune_eval_many_models.py's own approach.
        model.head = nn.Linear(config.hidden_dim, num_tasks).to(device)
        for p in model.parameters():
            p.requires_grad = True

        result = finetune_model(model, train_loader, val_loader, device, "tox21", finetune_epochs, lr, weight_decay, test_loader=test_loader)
    elif dataset in TDC_DATASETS:
        rows = load_tdc_admet_benchmark_splits(dataset, tdc_data_dirs[dataset], split_seed=seed)[0]
        model, config = load_model_for_finetuning(resolved_checkpoint, device, dataset)
        explicit_h, encode_h = infer_graph_featurization(config)

        train_loader, _, _, _ = make_graph_loader(rows["train"], explicit_h, encode_h, batch_size)
        val_loader, _, _, _ = make_graph_loader(rows["val"], explicit_h, encode_h, batch_size)
        test_loader, _, _, _ = make_graph_loader(rows["test"], explicit_h, encode_h, batch_size)

        result = finetune_model(model, train_loader, val_loader, device, dataset, finetune_epochs, lr, weight_decay, test_loader=test_loader)
    elif dataset == "bace":
        # Same admet_benchmark protocol as the TDC datasets -- DeepChem's own ScaffoldSplitter
        # never actually uses its seed argument (verified by reading the source), so the old
        # get_dataset_splits(..., "bace", ...) path below always returned the identical split
        # regardless of seed. load_bace_admet_benchmark_splits fixes that the same way
        # evaluation/tdc_datasets.py already does for BBB_Martins/hERG/AMES: fixed test set,
        # seeded scaffold reshuffle of train/valid via TDC's own (seed-using) splitter.
        rows = load_bace_admet_benchmark_splits(bace_data_dir, split_seed=seed)[0]
        model, config = load_model_for_finetuning(resolved_checkpoint, device, dataset)
        explicit_h, encode_h = infer_graph_featurization(config)

        train_loader, _, _, _ = make_graph_loader(rows["train"], explicit_h, encode_h, batch_size)
        val_loader, _, _, _ = make_graph_loader(rows["val"], explicit_h, encode_h, batch_size)
        test_loader, _, _, _ = make_graph_loader(rows["test"], explicit_h, encode_h, batch_size)

        result = finetune_model(model, train_loader, val_loader, device, dataset, finetune_epochs, lr, weight_decay, test_loader=test_loader)
    else:
        rows = get_dataset_splits(dataset, seed, lipo_data_dir, tox21_data_dir, bace_data_dir)
        model, config = load_model_for_finetuning(resolved_checkpoint, device, dataset)
        explicit_h, encode_h = infer_graph_featurization(config)

        train_loader, _, _, _ = make_graph_loader(rows["train"], explicit_h, encode_h, batch_size)
        val_loader, _, _, _ = make_graph_loader(rows["val"], explicit_h, encode_h, batch_size)
        test_loader, _, _, _ = make_graph_loader(rows["test"], explicit_h, encode_h, batch_size)

        result = finetune_model(model, train_loader, val_loader, device, dataset, finetune_epochs, lr, weight_decay, test_loader=test_loader)

    # result["test_metrics"] above already captured the REAL finetuned head's end-to-end test
    # performance (evaluate_finetuned_head_on_test, called by finetune_model before returning) --
    # that's the actual point of finetuning, and used to be silently thrown away here entirely.
    # What follows is Phase 2's own, separate question ("did finetuning change the encoder's
    # general representation quality"): it re-extracts embeddings and fits a fresh probe, which
    # only ever needs the encoder, never model.head. Phase 2 loads checkpoints via
    # GNNModel.from_config(config) followed by a STRICT model.load_state_dict() -- the same
    # shared code path used for the frozen-embedding evals everywhere else this session, which
    # must stay strict there. config.head_type is left as "dino" (unmodified from the original
    # SSL checkpoint) so from_config builds a ProjectionHead -- but the finetuning task head
    # (RegressionHead / ClassificationHead / plain nn.Linear for tox21) doesn't structurally
    # match a ProjectionHead's state dict at all, so it's swapped for throwaway, correctly-shaped
    # ProjectionHead weights here -- fine for Phase 2's purposes since result["test_metrics"]
    # already preserved the trained head's own answer above, before this swap.
    encoder_state = {k: v for k, v in model.state_dict().items() if k.startswith("encoder.")}
    dummy_head = ProjectionHead(
        input_dim=config.hidden_dim,
        hidden_dim=config.projection_hidden_dim,
        output_dim=config.projection_output_dim,
        num_layers=config.projection_layers,
        bottleneck_dim=config.projection_bottleneck_dim,
    )
    dummy_head_state = {f"head.{k}": v for k, v in dummy_head.state_dict().items()}
    save_state_dict = {**encoder_state, **dummy_head_state}

    out_path = Path(out_dir) / f"{dataset}_seed{seed}.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": save_state_dict,
            "config": asdict(config),
            "dataset": dataset,
            "seed": seed,
            "finetune_result": result,
            "source_checkpoint": resolved_checkpoint,
        },
        out_path,
    )
    test_summary = ""
    if "test_metrics" in result:
        tm = result["test_metrics"]
        if "roc_auc" in tm:
            test_summary = f", test_roc_auc={tm['roc_auc']:.4f}, test_f1={tm['f1']:.4f}, test_mcc={tm['mcc']:.4f}"
        elif "roc_auc_mean_tasks" in tm:
            test_summary = f", test_roc_auc={tm['roc_auc_mean_tasks']:.4f}, test_mcc={tm['mcc_mean_tasks']:.4f}"
        elif "r2" in tm:
            test_summary = f", test_r2={tm['r2']:.4f}, test_rmse={tm['rmse']:.4f}"
    print(f"  Saved finetuned checkpoint: {out_path}  (best_epoch={result['best_epoch']}, "
          f"best_val={result['best_val_loss']:.4f}{test_summary})")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: GPU-only encoder finetuning, saves checkpoints per (model, dataset, seed).")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names (mirrors eval_many_models_mlp_rf.py's --models).")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pth",
                         help="Checkpoint filename under models/<model>/checkpoints/. Defaults to best_model.pth "
                              "(SSL-val-loss-selected), matching the convention used everywhere else this session.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-weight-decay", type=float, default=1e-5)
    parser.add_argument("--random-split-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--datasets", type=str, default="lipo,bace,tox21")
    parser.add_argument("--lipo-data-dir", type=str, default="data/MoleculeNet_LIPO_custom")
    parser.add_argument("--tox21-data-dir", type=str, default="data/MoleculeNet_Tox21_custom")
    parser.add_argument("--bace-data-dir", type=str, default="data/MoleculeNet_BACE_custom")
    parser.add_argument("--bbb-martins-data-dir", type=str, default="data/TDC_BBB_Martins_custom")
    parser.add_argument("--herg-data-dir", type=str, default="data/TDC_hERG_custom")
    parser.add_argument("--ames-data-dir", type=str, default="data/TDC_AMES_custom")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.random_split_seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    tdc_data_dirs = {"bbb_martins": args.bbb_martins_data_dir, "herg": args.herg_data_dir, "ames": args.ames_data_dir}

    for model_name in model_names:
        print(f"\n{'=' * 80}\nModel: {model_name}\n{'=' * 80}")
        out_dir = Path(f"models/{model_name}/finetune_checkpoints")
        for dataset in datasets:
            # TDC datasets AND bace (now on the same admet_benchmark protocol -- see the bace
            # branch above) ALWAYS use the official 5 leaderboard seeds (1-5), regardless of
            # --random-split-seeds -- same reasoning as eval_phase1_extract_embeddings.py's
            # main(): the fixed test set is identical across them, only train/valid reshuffles,
            # and it's what makes the finetuned numbers comparable to the frozen-embedding ones.
            dataset_seeds = [1, 2, 3, 4, 5] if dataset in TDC_DATASETS or dataset == "bace" else seeds
            for seed in dataset_seeds:
                print(f"\n=== Finetuning {dataset} seed={seed} ===")
                finetune_and_save(
                    model_name, args.checkpoint_path, args.checkpoint_name, device, dataset, seed,
                    args.finetune_epochs, args.batch_size, args.finetune_lr, args.finetune_weight_decay,
                    args.lipo_data_dir, args.tox21_data_dir, args.bace_data_dir, tdc_data_dirs, out_dir,
                )

    print("\nPhase 1 complete. Run finetune_phase2_eval.py next to evaluate the saved checkpoints.")


if __name__ == "__main__":
    main()

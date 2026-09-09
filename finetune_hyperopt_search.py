"""Separate, deliberately isolated script: real hyperparameter optimization
(lr, weight_decay, epochs) of the end-to-end finetuning pipeline via
Hyperopt's TPE search, instead of the one fixed recipe (lr=1e-4, wd=1e-5,
epochs=30) used everywhere else in finetune_phase1_train.py.

Kept OUT of the existing finetuning pipeline on purpose. finetune_phase1_train.py
/ finetune_phase2_eval.py exist specifically to give every model the SAME
fixed, untuned recipe, so cross-model comparisons aren't confounded by unequal
tuning budgets -- that's the whole point of that pipeline. This script answers
a different question ("how good can finetuning get for ONE specific (model,
dataset) pair if you actually tune it") and should never be used to produce
numbers that get compared against the fixed-recipe results as if they were
the same experiment.

Methodologically important: the search NEVER touches test while searching --
the Hyperopt objective is the validation metric only (same selection metric
finetune_model() already uses: val ROC-AUC for classification, val loss for
regression), averaged across one or more seeds' train/valid splits
(--search-seeds, defaults to the first 3 official seeds) so a single trial's
score isn't just noise from one split -- this matters most on small datasets
(e.g. hERG's ~650 compounds) where a lone validation split can make one
hyperparameter set look better than it really is. Once the search finds a best (lr, weight_decay, epochs), that
FIXED hyperparameter set is run once across the dataset's official seeds
(matching every other finetuning result in this project) to get a real,
comparable multi-seed mean/std test metric via the same
evaluate_finetuned_head_on_test() used by the standard pipeline -- test is
touched exactly once per seed, at the very end, never during search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from torch_geometric.loader import DataLoader

from datahandling.graph_creation import smiles_to_pygdata
from evaluation.knn_bace import load_bace_admet_benchmark_splits
from evaluation.knn_lipo import infer_graph_featurization
from evaluation.tdc_datasets import load_tdc_admet_benchmark_splits
from finetune_eval_many_models import (
    resolve_checkpoint_path,
    load_model_for_finetuning,
    finetune_model,
    get_dataset_splits,
    make_graph_loader,
)

TDC_DATASETS = ("bbb_martins", "herg", "ames")
ADMET_BENCHMARK_SEEDS = (1, 2, 3, 4, 5)  # bace + the 3 TDC datasets
RANDOM_SPLIT_SEEDS = (0, 1, 2, 3, 4)  # lipo, tox21


class FlexibleHead(nn.Module):
    """Configurable-depth MLP head -- unlike RegressionHead/ClassificationHead
    (model/gnn_model.py), which are both hardcoded to exactly one hidden
    layer, this lets num_layers and hidden_dim be searched alongside the
    optimizer hyperparameters. Deliberately defined here rather than added to
    model/gnn_model.py: that file's heads are used by the standardized
    fixed-recipe finetuning pipeline everywhere else in the project, and
    changing them would confound that comparison -- the exact thing this
    whole script exists to avoid doing to the rest of the pipeline.
    num_layers=1 with the same hidden_dim reduces to exactly
    RegressionHead/ClassificationHead's architecture, so the search space
    strictly contains the old fixed head as one of its choices."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def _official_seeds(dataset: str) -> tuple[int, ...]:
    if dataset in TDC_DATASETS or dataset == "bace":
        return ADMET_BENCHMARK_SEEDS
    return RANDOM_SPLIT_SEEDS


def build_model_and_loaders(dataset, seed, resolved_checkpoint, device, batch_size,
                             lipo_data_dir, tox21_data_dir, bace_data_dir, tdc_data_dirs,
                             need_test: bool, head_hidden_dim: int | None = None, head_num_layers: int | None = None,
                             dropout: float | None = None):
    """Fresh model (reloaded from the original SSL checkpoint -- never carries over
    weights from a previous trial) + train/val/(test) loaders for one (dataset, seed).
    Mirrors finetune_phase1_train.py's own per-dataset branching.

    head_hidden_dim/head_num_layers, when both given, REPLACE whatever head
    load_model_for_finetuning attached (its fixed single-hidden-layer default)
    with a FlexibleHead of the requested shape -- this is what lets head
    architecture itself be a search dimension, not just a fixed choice.

    dropout, when given, overrides both the pretrained GAT encoder's dropout
    rate (model.encoder.dropout is a plain float read at forward time via
    F.dropout(..., p=self.dropout), not a learned weight, so it's safe to
    change after loading pretrained weights) and the FlexibleHead's own
    inter-layer dropout."""
    if dataset == "tox21":
        from evaluation.knn_tox21 import load_tox21_splits_from_deepchem

        splits, _ = load_tox21_splits_from_deepchem(tox21_data_dir, "random", split_seed=seed)
        model, config = load_model_for_finetuning(resolved_checkpoint, device, "tox21")
        explicit_h, encode_h = infer_graph_featurization(config)

        labels_train = splits["train"]["labels"]
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
        val_loader = build_multi(splits["val"]["smiles"], splits["val"]["labels"])
        test_loader = build_multi(splits["test"]["smiles"], splits["test"]["labels"]) if need_test else None

        # Fixed single-linear-layer default -- overridden below by the FlexibleHead
        # block when head_hidden_dim/head_num_layers are given.
        model.head = nn.Linear(config.hidden_dim, num_tasks).to(device)
        for p in model.parameters():
            p.requires_grad = True
        output_dim = num_tasks
    elif dataset in TDC_DATASETS:
        rows = load_tdc_admet_benchmark_splits(dataset, tdc_data_dirs[dataset], split_seed=seed)[0]
        model, config = load_model_for_finetuning(resolved_checkpoint, device, dataset)
        explicit_h, encode_h = infer_graph_featurization(config)
        train_loader, _, _, _ = make_graph_loader(rows["train"], explicit_h, encode_h, batch_size)
        val_loader, _, _, _ = make_graph_loader(rows["val"], explicit_h, encode_h, batch_size)
        test_loader, _, _, _ = make_graph_loader(rows["test"], explicit_h, encode_h, batch_size) if need_test else (None, None, None, None)
        output_dim = 1
    elif dataset == "bace":
        rows = load_bace_admet_benchmark_splits(bace_data_dir, split_seed=seed)[0]
        model, config = load_model_for_finetuning(resolved_checkpoint, device, dataset)
        explicit_h, encode_h = infer_graph_featurization(config)
        train_loader, _, _, _ = make_graph_loader(rows["train"], explicit_h, encode_h, batch_size)
        val_loader, _, _, _ = make_graph_loader(rows["val"], explicit_h, encode_h, batch_size)
        test_loader, _, _, _ = make_graph_loader(rows["test"], explicit_h, encode_h, batch_size) if need_test else (None, None, None, None)
        output_dim = 1
    else:
        rows = get_dataset_splits(dataset, seed, lipo_data_dir, tox21_data_dir, bace_data_dir)
        model, config = load_model_for_finetuning(resolved_checkpoint, device, dataset)
        explicit_h, encode_h = infer_graph_featurization(config)
        train_loader, _, _, _ = make_graph_loader(rows["train"], explicit_h, encode_h, batch_size)
        val_loader, _, _, _ = make_graph_loader(rows["val"], explicit_h, encode_h, batch_size)
        test_loader, _, _, _ = make_graph_loader(rows["test"], explicit_h, encode_h, batch_size) if need_test else (None, None, None, None)
        output_dim = 1

    if head_hidden_dim is not None and head_num_layers is not None:
        model.head = FlexibleHead(config.hidden_dim, head_hidden_dim, head_num_layers, output_dim,
                                   dropout=dropout or 0.0).to(device)
        for p in model.parameters():
            p.requires_grad = True

    if dropout is not None:
        model.encoder.dropout = dropout

    return model, train_loader, val_loader, test_loader


def run_search(dataset, resolved_checkpoint, device, batch_size, search_seeds, max_evals,
                lr_bounds, wd_bounds, epoch_choices, head_hidden_choices, head_layers_choices,
                dropout_bounds, scheduler_choices,
                lipo_data_dir, tox21_data_dir, bace_data_dir, tdc_data_dirs):
    print(f"\n=== Hyperopt search: dataset={dataset}  search_seeds={search_seeds}  max_evals={max_evals} ===")
    # Loaders for each search seed are built once and reused across all trials -- only
    # the model needs reloading each trial (fresh SSL-pretrained weights), the data
    # itself doesn't change. Averaging the objective across multiple seeds (instead of
    # just one) means a trial's score reflects genuine hyperparameter quality rather
    # than one split's idiosyncrasies -- more important the smaller the dataset is.
    seed_loaders = {}
    for seed in search_seeds:
        _, train_loader, val_loader, _ = build_model_and_loaders(
            dataset, seed, resolved_checkpoint, device, batch_size,
            lipo_data_dir, tox21_data_dir, bace_data_dir, tdc_data_dirs, need_test=False,
        )
        seed_loaders[seed] = (train_loader, val_loader)

    trial_log = []

    def objective(params):
        per_seed_val = {}
        per_seed_best_epoch = {}
        for seed, (train_loader, val_loader) in seed_loaders.items():
            model, _, _, _ = build_model_and_loaders(
                dataset, seed, resolved_checkpoint, device, batch_size,
                lipo_data_dir, tox21_data_dir, bace_data_dir, tdc_data_dirs, need_test=False,
                head_hidden_dim=int(params["head_hidden_dim"]), head_num_layers=int(params["head_num_layers"]),
                dropout=float(params["dropout"]),
            )
            result = finetune_model(
                model, train_loader, val_loader, device, dataset,
                int(params["epochs"]), params["lr"], params["weight_decay"],
                scheduler_type=params["scheduler_type"],
            )
            # misleading key name in finetune_model -- this is the actual selection
            # metric: val LOSS for lipo (lower better), val ROC-AUC for everything
            # else (higher better). See finetune_eval_many_models.py's finetune_model().
            per_seed_val[seed] = result["best_val_loss"]
            per_seed_best_epoch[seed] = result["best_epoch"]
        val_metric = float(np.mean(list(per_seed_val.values())))
        loss = val_metric if dataset == "lipo" else -val_metric
        trial_log.append({
            "params": {"lr": float(params["lr"]), "weight_decay": float(params["weight_decay"]),
                       "epochs": int(params["epochs"]), "head_hidden_dim": int(params["head_hidden_dim"]),
                       "head_num_layers": int(params["head_num_layers"]), "dropout": float(params["dropout"]),
                       "scheduler_type": params["scheduler_type"]},
            "val_metric": val_metric,
            "per_seed_val_metric": {str(s): float(v) for s, v in per_seed_val.items()},
            "per_seed_best_epoch": {str(s): int(e) for s, e in per_seed_best_epoch.items()},
        })
        seed_str = " ".join(f"s{s}={v:.4f}" for s, v in per_seed_val.items())
        print(f"  trial: lr={params['lr']:.2e} wd={params['weight_decay']:.2e} epochs={int(params['epochs'])}"
              f" head_hidden={int(params['head_hidden_dim'])} head_layers={int(params['head_num_layers'])}"
              f" dropout={params['dropout']:.3f} scheduler={params['scheduler_type']}"
              f"  -> val_metric={val_metric:.4f} ({seed_str})")
        return {"loss": float(loss), "status": STATUS_OK}

    space = {
        "lr": hp.loguniform("lr", np.log(lr_bounds[0]), np.log(lr_bounds[1])),
        "weight_decay": hp.loguniform("weight_decay", np.log(wd_bounds[0]), np.log(wd_bounds[1])),
        "epochs": hp.choice("epochs", epoch_choices),
        "head_hidden_dim": hp.choice("head_hidden_dim", head_hidden_choices),
        "head_num_layers": hp.choice("head_num_layers", head_layers_choices),
        "dropout": hp.uniform("dropout", dropout_bounds[0], dropout_bounds[1]),
        "scheduler_type": hp.choice("scheduler_type", scheduler_choices),
    }

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials, show_progressbar=False)
    best_params = {
        "lr": float(best["lr"]), "weight_decay": float(best["weight_decay"]),
        "epochs": int(epoch_choices[best["epochs"]]),
        "head_hidden_dim": int(head_hidden_choices[best["head_hidden_dim"]]),
        "head_num_layers": int(head_layers_choices[best["head_num_layers"]]),
        "dropout": float(best["dropout"]),
        "scheduler_type": scheduler_choices[best["scheduler_type"]],
    }
    print(f"  Best found: {best_params}")
    return best_params, trial_log


def evaluate_best_across_seeds(dataset, resolved_checkpoint, device, batch_size, best_params,
                                lipo_data_dir, tox21_data_dir, bace_data_dir, tdc_data_dirs):
    seeds = _official_seeds(dataset)
    print(f"\n=== Evaluating best hyperparameters across {len(seeds)} official seeds: {seeds} ===")
    per_seed_metrics = []
    for seed in seeds:
        model, train_loader, val_loader, test_loader = build_model_and_loaders(
            dataset, seed, resolved_checkpoint, device, batch_size,
            lipo_data_dir, tox21_data_dir, bace_data_dir, tdc_data_dirs, need_test=True,
            head_hidden_dim=best_params["head_hidden_dim"], head_num_layers=best_params["head_num_layers"],
            dropout=best_params["dropout"],
        )
        result = finetune_model(
            model, train_loader, val_loader, device, dataset,
            best_params["epochs"], best_params["lr"], best_params["weight_decay"],
            test_loader=test_loader, scheduler_type=best_params["scheduler_type"],
        )
        print(f"  seed={seed}  test_metrics={result['test_metrics']}")
        per_seed_metrics.append({"seed": seed, "best_epoch": result["best_epoch"], "test_metrics": result["test_metrics"]})
    return per_seed_metrics


def _aggregate(per_seed_metrics, metric_keys):
    agg = {}
    for key in metric_keys:
        vals = [m["test_metrics"][key] for m in per_seed_metrics if key in m["test_metrics"]]
        if vals:
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def main():
    parser = argparse.ArgumentParser(description="Hyperopt search over finetuning hyperparameters for one (model, dataset) pair.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pth")
    parser.add_argument("--dataset", type=str, required=True,
                         choices=["lipo", "bace", "tox21", "bbb_martins", "herg", "ames"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-evals", type=int, default=20)
    parser.add_argument("--search-seeds", type=str, default=None,
                         help="Comma-separated seeds to average the validation metric over during search "
                              "(e.g. '1,2,3'). Defaults to the first 3 official seeds for the dataset. "
                              "More seeds means a more robust search signal at proportionally higher cost.")
    parser.add_argument("--lr-min", type=float, default=1e-6,
                         help="Widened from the original 1e-5 floor after the hERG/AMES runs both found "
                              "their best lr hugging that edge, suggesting the search wanted to go lower.")
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--wd-min", type=float, default=1e-8,
                         help="Widened from the original 1e-6 floor for the same edge-hugging reason as lr-min.")
    parser.add_argument("--wd-max", type=float, default=1e-2)
    parser.add_argument("--epoch-choices", type=str, default="15,30,50,75",
                         help="Added 75 -- lower lr values may need more epochs to converge.")
    parser.add_argument("--head-hidden-choices", type=str, default="128,256,512,1024",
                         help="Candidate head hidden_dim values -- searched alongside the optimizer hyperparameters.")
    parser.add_argument("--head-layers-choices", type=str, default="1,2,3",
                         help="Candidate head depths (num hidden layers). 1 matches the old fixed "
                              "RegressionHead/ClassificationHead architecture; this search can find deeper.")
    parser.add_argument("--dropout-min", type=float, default=0.0)
    parser.add_argument("--dropout-max", type=float, default=0.5,
                         help="Overrides both the pretrained GAT encoder's dropout rate and the "
                              "FlexibleHead's inter-layer dropout (same value for both).")
    parser.add_argument("--scheduler-choices", type=str, default="constant,cosine",
                         help="Candidate LR schedules. 'constant' matches the old fixed-recipe behavior; "
                              "'cosine' anneals lr to 0 over the trial's epoch budget.")
    parser.add_argument("--lipo-data-dir", type=str, default="data/MoleculeNet_LIPO_custom")
    parser.add_argument("--tox21-data-dir", type=str, default="data/MoleculeNet_Tox21_custom")
    parser.add_argument("--bace-data-dir", type=str, default="data/MoleculeNet_BACE_custom")
    parser.add_argument("--bbb-martins-data-dir", type=str, default="data/TDC_BBB_Martins_custom")
    parser.add_argument("--herg-data-dir", type=str, default="data/TDC_hERG_custom")
    parser.add_argument("--ames-data-dir", type=str, default="data/TDC_AMES_custom")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    resolved_checkpoint = resolve_checkpoint_path(args.model, args.checkpoint_path, args.checkpoint_name)
    tdc_data_dirs = {"bbb_martins": args.bbb_martins_data_dir, "herg": args.herg_data_dir, "ames": args.ames_data_dir}
    epoch_choices = [int(e.strip()) for e in args.epoch_choices.split(",") if e.strip()]
    head_hidden_choices = [int(h.strip()) for h in args.head_hidden_choices.split(",") if h.strip()]
    head_layers_choices = [int(n.strip()) for n in args.head_layers_choices.split(",") if n.strip()]
    scheduler_choices = [s.strip() for s in args.scheduler_choices.split(",") if s.strip()]
    if args.search_seeds:
        search_seeds = tuple(int(s.strip()) for s in args.search_seeds.split(",") if s.strip())
    else:
        search_seeds = _official_seeds(args.dataset)[:3]

    best_params, trial_log = run_search(
        args.dataset, resolved_checkpoint, device, args.batch_size, search_seeds, args.max_evals,
        (args.lr_min, args.lr_max), (args.wd_min, args.wd_max), epoch_choices, head_hidden_choices, head_layers_choices,
        (args.dropout_min, args.dropout_max), scheduler_choices,
        args.lipo_data_dir, args.tox21_data_dir, args.bace_data_dir, tdc_data_dirs,
    )

    per_seed_metrics = evaluate_best_across_seeds(
        args.dataset, resolved_checkpoint, device, args.batch_size, best_params,
        args.lipo_data_dir, args.tox21_data_dir, args.bace_data_dir, tdc_data_dirs,
    )

    if args.dataset == "lipo":
        metric_keys = ("r2", "rmse", "mae")
    elif args.dataset == "tox21":
        metric_keys = ("roc_auc_mean_tasks", "f1_mean_tasks", "mcc_mean_tasks", "balanced_accuracy_mean_tasks")
    else:
        metric_keys = ("roc_auc", "f1", "mcc", "balanced_accuracy")
    summary = _aggregate(per_seed_metrics, metric_keys)

    print(f"\n--- Summary: {args.model} / {args.dataset} (hyperopt-tuned finetuning) ---")
    print(f"Best hyperparameters: {best_params}")
    for k, v in summary.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    output_path = args.output or f"models/{args.model}/hyperopt_finetune_{args.dataset}_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "dataset": args.dataset,
            "search_seeds": list(search_seeds),
            "max_evals": args.max_evals,
            "best_params": best_params,
            "trial_log": trial_log,
            "per_seed_test_metrics": per_seed_metrics,
            "summary": summary,
        }, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"✓ Saved {output_path}")


if __name__ == "__main__":
    main()

"""Phase 2 of the two-phase finetuning eval: loads the checkpoints saved by
finetune_phase1_train.py and evaluates them with the exact same MLP/RF
methodology used everywhere else this session (eval_many_models_mlp_rf.py) --
not KNN/linear-probe, so results are directly comparable to the frozen-
embedding numbers already produced for every other model.

Structure deliberately mirrors eval_many_models_mlp_rf.py's own main()
(multi-model loop, per-model durable JSON in models/<name>/, a combined
--output file, one comparison plot across all models) -- the only addition
is that each model's checkpoint is the finetuned one from Phase 1 instead of
the raw SSL checkpoint. eval_lipo/eval_bace/eval_tox21/generate_comparison_plot
are reused unchanged from eval_many_models_mlp_rf.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_many_models_mlp_rf import eval_lipo, eval_bace, eval_tox21, eval_tdc_classification, generate_dataset_comparison_plots, _aggregate_seeds
from evaluation.knn_lipo import resolve_torch_device

TDC_DATASETS = ("bbb_martins", "herg", "ames")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: evaluate finetuned checkpoints from finetune_phase1_train.py via MLP/RF.")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names (must match --models used in phase 1).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--random-split-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--datasets", type=str, default="lipo,bace,tox21")
    parser.add_argument("--lipo-data-dir", type=str, default="data/MoleculeNet_LIPO_custom")
    parser.add_argument("--tox21-data-dir", type=str, default="data/MoleculeNet_Tox21_custom")
    parser.add_argument("--bace-data-dir", type=str, default="data/MoleculeNet_BACE_custom")
    parser.add_argument("--bace-splitter", type=str, default="admet_benchmark")
    parser.add_argument("--bbb-martins-data-dir", type=str, default="data/TDC_BBB_Martins_custom")
    parser.add_argument("--herg-data-dir", type=str, default="data/TDC_hERG_custom")
    parser.add_argument("--ames-data-dir", type=str, default="data/TDC_AMES_custom")
    parser.add_argument("--allow-partial-results", action="store_true")
    parser.add_argument("--output", type=str, default="finetune_eval_many_models_mlp_rf_results.json")
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
    print(f"Device: {device}")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.random_split_seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    tdc_data_dirs = {"bbb_martins": args.bbb_martins_data_dir, "herg": args.herg_data_dir, "ames": args.ames_data_dir}

    all_results = {}
    for model_name in model_names:
        print(f"\n=== Evaluating finetuned {model_name} ===")
        checkpoint_dir = Path(f"models/{model_name}/finetune_checkpoints")

        model_results = {"lipo": [], "tox21": [], "bace": [], "bbb_martins": [], "herg": [], "ames": []}
        failures = []
        finetune_details = {"lipo": [], "tox21": [], "bace": [], "bbb_martins": [], "herg": [], "ames": []}

        for dataset in datasets:
            # Must match the exact seed set finetune_phase1_train.py actually ran with -- TDC
            # datasets and bace (same admet_benchmark protocol) always use the official 1-5,
            # everything else uses --random-split-seeds.
            dataset_seeds = [1, 2, 3, 4, 5] if dataset in TDC_DATASETS or dataset == "bace" else seeds
            for seed in dataset_seeds:
                ckpt_path = checkpoint_dir / f"{dataset}_seed{seed}.pth"
                if not ckpt_path.exists():
                    msg = f"{dataset} seed={seed}: missing finetuned checkpoint {ckpt_path} (did phase 1 run for this?)"
                    print(f"[SKIPPED] {msg}")
                    failures.append(msg)
                    continue
                try:
                    if dataset == "lipo":
                        result = eval_lipo(model_name, str(ckpt_path), device, seed, args.lipo_data_dir, "random")
                    elif dataset == "bace":
                        result = eval_bace(model_name, str(ckpt_path), device, seed, args.bace_data_dir, args.bace_splitter)
                    elif dataset == "tox21":
                        result = eval_tox21(model_name, str(ckpt_path), device, seed, args.tox21_data_dir, "random")
                    elif dataset in TDC_DATASETS:
                        result = eval_tdc_classification(dataset, model_name, str(ckpt_path), device, seed, tdc_data_dirs[dataset])
                    else:
                        continue
                    model_results[dataset].append(result)

                    ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    finetune_details[dataset].append({"seed": seed, "finetune_result": ckpt_meta.get("finetune_result")})
                    del ckpt_meta
                except Exception as exc:
                    msg = f"[{dataset.upper()}][seed={seed}] {model_name}: {exc}"
                    print(f"[FAILED] {msg}")
                    failures.append(msg)

        summary = {}
        if model_results["lipo"]:
            for method in ("mlp", "rf"):
                for feat in ("embeddings", "fingerprints"):
                    mean, std = _aggregate_seeds(model_results["lipo"], (feat, method, "test_metrics", "r2"))
                    summary[f"lipo_{feat}_{method}_r2"] = {"mean": mean, "std": std}
        if model_results["bace"]:
            for method in ("mlp", "rf"):
                for feat in ("embeddings", "fingerprints"):
                    mean, std = _aggregate_seeds(model_results["bace"], (feat, method, "test_metrics", "roc_auc"))
                    summary[f"bace_{feat}_{method}_roc_auc"] = {"mean": mean, "std": std}
        if model_results["tox21"]:
            for method in ("mlp", "rf"):
                for feat in ("embeddings", "fingerprints"):
                    mean, std = _aggregate_seeds(model_results["tox21"], (feat, method, "test_metrics", "roc_auc_mean_tasks"))
                    summary[f"tox21_{feat}_{method}_roc_auc"] = {"mean": mean, "std": std}
        for tdc_ds in TDC_DATASETS:
            if model_results[tdc_ds]:
                for method in ("mlp", "rf"):
                    for feat in ("embeddings", "fingerprints"):
                        for metric in ("roc_auc", "f1", "mcc"):
                            mean, std = _aggregate_seeds(model_results[tdc_ds], (feat, method, "test_metrics", metric))
                            summary[f"{tdc_ds}_{feat}_{method}_{metric}"] = {"mean": mean, "std": std}

        # The REAL end-to-end result: the actual finetuned task head's own test performance
        # (evaluate_finetuned_head_on_test, captured by finetune_phase1_train.py before the head
        # was discarded for Phase 2's re-probing) -- distinct "_finetuned_head_" key prefix so it
        # never collides with the re-probed "_embeddings_{mlp,rf}_" numbers above. Entries from
        # checkpoints saved before this was added won't have "test_metrics" and are skipped.
        for ds in ("lipo", "bace") + TDC_DATASETS:
            entries = [d for d in finetune_details[ds] if (d.get("finetune_result") or {}).get("test_metrics")]
            if not entries:
                continue
            metrics = ("r2", "rmse", "mae") if ds == "lipo" else ("roc_auc", "f1", "mcc", "balanced_accuracy")
            for metric in metrics:
                mean, std = _aggregate_seeds(entries, ("finetune_result", "test_metrics", metric))
                summary[f"{ds}_finetuned_head_{metric}"] = {"mean": mean, "std": std}
        tox21_entries = [d for d in finetune_details["tox21"] if (d.get("finetune_result") or {}).get("test_metrics")]
        if tox21_entries:
            for metric in ("roc_auc", "f1", "mcc", "balanced_accuracy"):
                mean, std = _aggregate_seeds(tox21_entries, ("finetune_result", "test_metrics", f"{metric}_mean_tasks"))
                summary[f"tox21_finetuned_head_{metric}"] = {"mean": mean, "std": std}

        model_result = {
            "summary": summary,
            "raw": model_results,
            "failures": failures,
            "finetune_details": finetune_details,
        }
        all_results[model_name] = model_result

        print(f"\n--- Finetuned {model_name} summary ---")
        for key, v in summary.items():
            print(f"  {key}: {v['mean']:.4f} ± {v['std']:.4f}")

        # Saved under a DISTINCT filename from the frozen-embedding results
        # (eval_mlp_rf_results.json) in the same model directory -- this is a
        # different evaluation condition (finetuned encoder), not a replacement.
        model_output_dir = Path(f"models/{model_name}")
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_output_path = model_output_dir / "finetune_eval_mlp_rf_results.json"
        with open(model_output_path, "w") as f:
            json.dump(model_result, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"✓ Saved {model_name}'s finetuned results to {model_output_path}")

        if failures and not args.allow_partial_results:
            raise RuntimeError(
                f"Evaluation completed with {len(failures)} failures for {model_name}. "
                "Re-run with --allow-partial-results to keep partial outputs without failing."
            )

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nSaved combined results to {args.output}")

    plotted_models = [m for m in model_names if all_results[m]["summary"]]
    if plotted_models:
        # filename_prefix="finetune_eval" writes directly to finetune_eval_{dataset}_vs_ecfp.png --
        # deliberately NOT the old clobber-then-rename trick, which would briefly overwrite (and,
        # since rename doesn't restore it, permanently lose) the frozen-eval plots already saved
        # in the same models/<name>/ directory.
        saved_plot_paths = generate_dataset_comparison_plots(all_results, plotted_models, filename_prefix="finetune_eval")
        for dataset_key, paths in saved_plot_paths.items():
            for path in paths:
                print(f"✓ Saved {dataset_key} comparison plot to {path}")
    else:
        print("⚠ Skipping plot generation -- no model produced a non-empty summary.")


if __name__ == "__main__":
    main()

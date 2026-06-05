"""Run multiple KNN downstream evaluations for multiple SSL model folders.

This runs LIPO, Tox21, and BACE evaluators for each model.

Example Usage: 
    python eval_many_models.py --models model1,model2,model3 --k_values 5,10,15

"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.knn_bace import run_knn_eval_bace
from evaluation.knn_lipo import run_knn_eval_lipo
from evaluation.knn_tox21 import run_knn_eval_tox21
from evaluation.linear_probe_eval import run_linear_probe
from plotting.knn_summary import (
    update_random_seed_summary,
    finalize_random_seed_summary,
    save_random_seed_artifacts,
)
from plotting.linear_probe_summary import (
    update_linear_probe_summary,
    finalize_linear_probe_summary,
    save_linear_probe_artifacts,
)


def parse_models(models_csv: str):
    return [name.strip() for name in models_csv.split(",") if name.strip()]


def parse_int_list(values_csv: str):
    return [int(value.strip()) for value in values_csv.split(",") if value.strip()]


def resolve_model_checkpoint_path(model_name: str, checkpoint_path: str | None, checkpoint_name: str | None) -> str | None:
    """Resolve either an explicit checkpoint path or a checkpoint filename under the model folder."""
    if checkpoint_path:
        return checkpoint_path

    if checkpoint_name:
        return str(Path(f"models/{model_name}/checkpoints") / checkpoint_name)

    return None


def main():
    parser = argparse.ArgumentParser(description="Run KNN evals (LIPO, Tox21, BACE) for multiple SSL models.")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated model names, e.g. GDZ_5000Epochs,GDZ_GAT_KHOP",
    )

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path to use for all evals. If omitted, each evaluator resolves its default checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Optional checkpoint filename under models/<model>/checkpoints/, e.g. best_online_eval_model.pth.",
    )

    parser.add_argument("--lipo-data-dir", type=str, default="data/MoleculeNet_LIPO_custom")
    parser.add_argument("--lipo-splitter", type=str, default="random")

    parser.add_argument("--tox21-data-dir", type=str, default="data/MoleculeNet_Tox21_custom")
    parser.add_argument("--tox21-splitter", type=str, default="random")

    parser.add_argument("--bace-data-dir", type=str, default="data/MoleculeNet_BACE_custom")
    parser.add_argument("--bace-splitter", type=str, default="scaffold")
    parser.add_argument(
        "--random-split-seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds used only when a dataset splitter is random.",
    )
    parser.add_argument(
        "--allow-partial-results",
        action="store_true",
        help="Do not fail the run if some dataset/seed evaluations fail.",
    )

    args = parser.parse_args()

    model_names = parse_models(args.models)
    if not model_names:
        raise ValueError("No model names provided via --models")

    random_split_seeds = parse_int_list(args.random_split_seeds)
    if (
        args.lipo_splitter == "random"
        or args.tox21_splitter == "random"
        or args.bace_splitter == "random"
        or args.dataset in {"all", "bace"}
    ) and not random_split_seeds:
        raise ValueError("At least one --random-split-seeds value is required for random split datasets.")

    for model_name in model_names:
        print(f"\n=== Evaluating {model_name} ===")
        random_seed_summary = {}
        linear_probe_seed_summary = {}
        model_failures = []
        resolved_checkpoint = resolve_model_checkpoint_path(model_name, args.checkpoint_path, args.checkpoint_name)

        if args.lipo_splitter == "random":
            print(f"\n--- LIPO (random seeds: {random_split_seeds}) ---")
            for seed in random_split_seeds:
                try:
                    print(f"\n[LIPO] Running seed={seed}")
                    result = run_knn_eval_lipo(
                        ssl_model_name=model_name,
                        checkpoint_path=resolved_checkpoint,
                        lipo_data_dir=args.lipo_data_dir,
                        lipo_splitter=args.lipo_splitter,
                        split_seed=seed,
                        device_preference=args.device,
                        save_plot=False,
                    )
                    update_random_seed_summary(random_seed_summary, result)
                except Exception as exc:
                    print(f"[FAILED][LIPO][seed={seed}] {model_name}: {exc}")
                    model_failures.append(f"LIPO-KNN seed={seed}: {exc}")
                # Also run linear probe for this seed and aggregate
                try:
                    lp_result, lp_path = run_linear_probe(
                        model_name=model_name,
                        dataset="lipo",
                        checkpoint=resolved_checkpoint,
                        device_pref=args.device,
                        split_seed=seed,
                        lipo_data_dir=args.lipo_data_dir,
                        lipo_splitter=args.lipo_splitter,
                    )
                    update_linear_probe_summary(linear_probe_seed_summary, lp_result)
                except Exception as exc:
                    print(f"[FAILED][LIPO-LP][seed={seed}] {model_name}: {exc}")
                    model_failures.append(f"LIPO-LP seed={seed}: {exc}")
        else:
            try:
                print("\n--- LIPO ---")
                result = run_knn_eval_lipo(
                    ssl_model_name=model_name,
                    checkpoint_path=resolved_checkpoint,
                    lipo_data_dir=args.lipo_data_dir,
                    lipo_splitter=args.lipo_splitter,
                    split_seed=None,
                    device_preference=args.device,
                    save_plot=False,
                )
                update_random_seed_summary(random_seed_summary, result)
            except Exception as exc:
                print(f"[FAILED][LIPO] {model_name}: {exc}")
                model_failures.append(f"LIPO-KNN: {exc}")
            # Run linear probe once for non-random LIPO
            try:
                lp_result, lp_path = run_linear_probe(
                    model_name=model_name,
                    dataset="lipo",
                    checkpoint=resolved_checkpoint,
                    device_pref=args.device,
                    split_seed=None,
                    lipo_data_dir=args.lipo_data_dir,
                    lipo_splitter=args.lipo_splitter,
                )
                update_linear_probe_summary(linear_probe_seed_summary, lp_result)
            except Exception as exc:
                print(f"[FAILED][LIPO-LP] {model_name}: {exc}")
                model_failures.append(f"LIPO-LP: {exc}")

        if args.tox21_splitter == "random":
            print(f"\n--- Tox21 (random seeds: {random_split_seeds}) ---")
            for seed in random_split_seeds:
                try:
                    print(f"\n[Tox21] Running seed={seed}")
                    result = run_knn_eval_tox21(
                        ssl_model_name=model_name,
                        checkpoint_path=resolved_checkpoint,
                        tox21_data_dir=args.tox21_data_dir,
                        tox21_splitter=args.tox21_splitter,
                        split_seed=seed,
                        device_preference=args.device,
                        save_plot=False,
                    )
                    update_random_seed_summary(random_seed_summary, result)
                except Exception as exc:
                    print(f"[FAILED][Tox21][seed={seed}] {model_name}: {exc}")
                    model_failures.append(f"Tox21-KNN seed={seed}: {exc}")

                try:
                    lp_result, lp_path = run_linear_probe(
                        model_name=model_name,
                        dataset="tox21",
                        checkpoint=resolved_checkpoint,
                        device_pref=args.device,
                        split_seed=seed,
                        tox21_data_dir=args.tox21_data_dir,
                        tox21_splitter=args.tox21_splitter,
                    )
                    update_linear_probe_summary(linear_probe_seed_summary, lp_result)
                except Exception as exc:
                    print(f"[FAILED][Tox21-LP][seed={seed}] {model_name}: {exc}")
                    model_failures.append(f"Tox21-LP seed={seed}: {exc}")
        else:
            try:
                print("\n--- Tox21 ---")
                result = run_knn_eval_tox21(
                    ssl_model_name=model_name,
                    checkpoint_path=resolved_checkpoint,
                    tox21_data_dir=args.tox21_data_dir,
                    tox21_splitter=args.tox21_splitter,
                    split_seed=None,
                    device_preference=args.device,
                    save_plot=False,
                )
                update_random_seed_summary(random_seed_summary, result)
            except Exception as exc:
                print(f"[FAILED][Tox21] {model_name}: {exc}")
                model_failures.append(f"Tox21-KNN: {exc}")

            try:
                lp_result, lp_path = run_linear_probe(
                    model_name=model_name,
                    dataset="tox21",
                    checkpoint=resolved_checkpoint,
                    device_pref=args.device,
                    split_seed=None,
                    tox21_data_dir=args.tox21_data_dir,
                    tox21_splitter=args.tox21_splitter,
                )
                update_linear_probe_summary(linear_probe_seed_summary, lp_result)
            except Exception as exc:
                print(f"[FAILED][Tox21-LP] {model_name}: {exc}")
                model_failures.append(f"Tox21-LP: {exc}")

        print(f"\n--- BACE ({args.bace_splitter}, seeds: {random_split_seeds}) ---")
        for seed in random_split_seeds:
            try:
                print(f"\n[BACE] Running seed={seed}")
                result = run_knn_eval_bace(
                    ssl_model_name=model_name,
                    checkpoint_path=resolved_checkpoint,
                    bace_data_dir=args.bace_data_dir,
                    bace_splitter=args.bace_splitter,
                    split_seed=seed,
                    device_preference=args.device,
                    save_plot=False,
                )
                update_random_seed_summary(random_seed_summary, result)
            except Exception as exc:
                print(f"[FAILED][BACE][seed={seed}] {model_name}: {exc}")
                model_failures.append(f"BACE-KNN seed={seed}: {exc}")

            try:
                lp_result, lp_path = run_linear_probe(
                    model_name=model_name,
                    dataset="bace",
                    checkpoint=resolved_checkpoint,
                    device_pref=args.device,
                    split_seed=seed,
                    bace_data_dir=args.bace_data_dir,
                    bace_splitter=args.bace_splitter,
                )
                update_linear_probe_summary(linear_probe_seed_summary, lp_result)
            except Exception as exc:
                print(f"[FAILED][BACE-LP][seed={seed}] {model_name}: {exc}")
                model_failures.append(f"BACE-LP seed={seed}: {exc}")

        finalized_summary = finalize_random_seed_summary(random_seed_summary)
        finalized_lp_summary = finalize_linear_probe_summary(linear_probe_seed_summary)
        if finalized_summary:
            save_random_seed_artifacts(model_name, finalized_summary)
        if finalized_lp_summary:
            save_linear_probe_artifacts(model_name, finalized_lp_summary)

        if model_failures:
            print(f"\nEncountered {len(model_failures)} failed evaluations for {model_name}:")
            for item in model_failures:
                print(f"  - {item}")
            if not args.allow_partial_results:
                raise RuntimeError(
                    "Evaluation completed with failures. "
                    "Re-run with --allow-partial-results to keep partial outputs without failing."
                )


if __name__ == "__main__":
    main()

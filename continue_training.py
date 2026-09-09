"""Resume DINO training from an existing checkpoint.

Example:
  uv run python continue_training.py --model 512_DIM_GINE --checkpoint-epoch 30
"""

import argparse
import json
from pathlib import Path

import json5
import pandas as pd
import torch

from model.config import ModelConfig
from plotting.loss_plot import load_loss_data, plot_ssl_and_online_knn, plot_train_val_loss_curves
from training.dino_training import dino_train
from utils.seed import set_seed


def load_model_config(model_name: str, config_path: str | None) -> ModelConfig:
    if config_path:
        path = Path(config_path)
    else:
        model_config_json = Path(f"models/{model_name}/config.json")
        path = model_config_json if model_config_json.exists() else Path("configs/default.json5")

    with open(path, "r", encoding="utf-8") as handle:
        if path.suffix == ".json5":
            payload = json5.load(handle)
        else:
            payload = json.load(handle)

    config = ModelConfig.from_dict(payload)
    config.name = model_name
    return config


def resolve_checkpoint_path(model_name: str, args: argparse.Namespace) -> Path:
    checkpoint_dir = Path(f"models/{model_name}/checkpoints")

    if args.checkpoint_path:
        return Path(args.checkpoint_path)

    if args.checkpoint_name:
        return checkpoint_dir / args.checkpoint_name

    if args.checkpoint_epoch is not None:
        return checkpoint_dir / f"checkpoint_epoch_{args.checkpoint_epoch}.pth"

    raise ValueError("One of --checkpoint-path, --checkpoint-name, or --checkpoint-epoch is required.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume training from a saved checkpoint.")
    parser.add_argument("--model", required=True, help="Model name under models/<name>/")

    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint-path", type=str, default=None)
    ckpt_group.add_argument("--checkpoint-name", type=str, default=None)
    ckpt_group.add_argument("--checkpoint-epoch", type=int, default=None)

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config override. Defaults to models/<model>/config.json if it exists.",
    )
    parser.add_argument(
        "--target-epochs",
        type=int,
        default=None,
        help="Total target epochs after resume (same meaning as config.num_epochs).",
    )
    parser.add_argument(
        "--extra-epochs",
        type=int,
        default=None,
        help="Train this many additional epochs from the checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.target_epochs is not None and args.extra_epochs is not None:
        raise ValueError("Use either --target-epochs or --extra-epochs, not both.")

    checkpoint_path = resolve_checkpoint_path(args.model, args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resume_epoch = int(checkpoint.get("epoch", -1)) + 1

    config = load_model_config(args.model, args.config)

    if args.extra_epochs is not None:
        if args.extra_epochs <= 0:
            raise ValueError("--extra-epochs must be > 0")
        config.num_epochs = resume_epoch + args.extra_epochs
    elif args.target_epochs is not None:
        config.num_epochs = int(args.target_epochs)

    if config.num_epochs <= resume_epoch:
        raise ValueError(
            f"Target num_epochs ({config.num_epochs}) must be greater than resumed epoch ({resume_epoch})."
        )

    print(f"Resuming model: {args.model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Resumed epoch index: {resume_epoch}")
    print(f"Training until epoch index: {config.num_epochs - 1}")

    set_seed(config.seed)
    dino_train(config, resume_checkpoint_path=str(checkpoint_path))

    # Plot results. Named with the final epoch reached (not a fixed filename)
    # so repeated/successive resumes each get their own plot instead of
    # overwriting the original run's loss_curves_ssl_knn.png or each other's.
    loss_history_path = f"models/{args.model}/loss_history.json"
    loss_history = load_loss_data(loss_history_path)

    final_epoch = config.num_epochs
    if isinstance(loss_history, dict) and loss_history.get("DINO_Loss"):
        final_epoch = loss_history["DINO_Loss"][-1].get("epoch", final_epoch)
    plot_basename = f"loss_curves_ssl_knn_to_epoch_{final_epoch}"

    online_eval_enabled = bool(getattr(config, "online_eval_enabled", False))
    online_eval_datasets = str(getattr(config, "online_eval_datasets", "lipo")).split(",")[0].strip()

    if online_eval_enabled and isinstance(loss_history, dict) and "Evaluation_Loss" in loss_history:
        try:
            plot_path = f"models/{args.model}/{plot_basename}.png"
            plot_ssl_and_online_knn(
                loss_history_path,
                plot_path,
                model_name=args.model,
                dataset=online_eval_datasets,
            )
            print(f"✓ Dual-axis plot saved: {plot_path}")
        except (ValueError, KeyError) as e:
            print(f"⚠ Could not generate dual-axis plot: {e}")
            if isinstance(loss_history, dict) and "DINO_Loss" in loss_history:
                loss_data = pd.DataFrame(loss_history["DINO_Loss"])
                plot_path = f"models/{args.model}/loss_curves_to_epoch_{final_epoch}.png"
                plot_train_val_loss_curves(loss_data, plot_path, model_name=args.model)
                print(f"✓ Standard plot saved: {plot_path}")
    else:
        if isinstance(loss_history, dict) and "DINO_Loss" in loss_history:
            loss_data = pd.DataFrame(loss_history["DINO_Loss"])
        else:
            loss_data = loss_history if isinstance(loss_history, pd.DataFrame) else pd.DataFrame(loss_history)
        plot_path = f"models/{args.model}/loss_curves_to_epoch_{final_epoch}.png"
        plot_train_val_loss_curves(loss_data, plot_path, model_name=args.model)
        print(f"✓ Standard plot saved: {plot_path}")


if __name__ == "__main__":
    main()

""" Main Entry point for training and evaluation"""

import argparse
import json
from pathlib import Path

import torch
import pandas as pd

from model.config import ModelConfig
from plotting.loss_plot import load_loss_data, plot_train_val_loss_curves, plot_ssl_and_online_knn
from training.dino_training import dino_train
from utils.seed import set_seed


def build_default_config() -> ModelConfig:
    return ModelConfig(
        # Model identity and architecture
        name="GDZ_GINE_MASKING",  # Model identifier (used for save paths/metadata)
        head_type="dino",  # SSL head type; keep as "dino" for DINO pretraining
        encoder_type="GINE",  # Backbone encoder: "GINE" or "GAT"

        # Data source
        data_path="data/zinc/zinc_data",  # Directory of .smi files or path to single CSV

        # Reproducibility and runtime
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=16,  # DataLoader workers

        # Multi-view augmentation setup
        local_views=4,  # Number of local views per graph
        local_augmentation_mode="masking",  # "k_hop" or "masking"
        k_hops=2,  # k-hop size when local_augmentation_mode="k_hop"
        node_mask_ratio=0.15,  # Node masking ratio when mode="masking"
        feature_mask_ratio=0.15,  # Feature masking ratio when mode="masking"

        # Input feature dimensions (dataset/dataloader dependent)
        num_features=24,
        edge_features=12,

        # Encoder size/depth
        hidden_dim=128,
        num_layers=3,
        dropout=0.0,
        epsilon=0.0,  # Used by GINE

        # Projection head used for SSL targets
        projection_hidden_dim=256,
        projection_output_dim=128,
        projection_layers=2,

        # Training length and batching
        num_epochs=200,
        batch_size=1024,  # Effective views/step = batch_size * (2 global + local_views)

        # Optimizer / schedules
        auto_scale_lr=True,  # Enable DINO-style linear LR scaling
        lr_scale_base=2.5e-4,
        lr_scale_reference_batch_size=256,
        learning_rate=1e-3,  # Ignored when auto_scale_lr=True
        weight_decay=0.04,  # Backward-compatibility field
        weight_decay_start=0.04,
        weight_decay_end=0.4,

        # DINO loss dynamics
        teacher_temp=0.04,
        teacher_temp_final=0.07,
        teacher_temp_warmup_epochs=30,
        student_temp=0.1,
        teacher_momentum=0.996,
        center_momentum=0.9,

        # Learning-rate schedule
        warmup_epochs=10,
        final_learning_rate=1e-5,

        # Online downstream tracking (kNN-only)
        online_eval_enabled=True,
        online_eval_every_n_epochs=1,
        online_eval_datasets="lipo",
        online_eval_fixed_k=5,
        online_eval_top_k_checkpoints=5,
    )


def load_config(config_path: str | None) -> ModelConfig:
    if not config_path:
        return build_default_config()

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return ModelConfig.from_dict(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINO model with optional JSON config.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file. If omitted, uses defaults in train.py.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    if args.config:
        print(f"Using config file: {args.config}")

    set_seed(config.seed)
    if config.head_type == "dino":
        dino_train(config)
        
        # Plot results
        loss_history_path = f"models/{config.name}/loss_history.json"
        loss_history = load_loss_data(loss_history_path)
        
        # If online eval was enabled, plot dual-axis (SSL loss + online kNN metric)
        online_eval_enabled = bool(getattr(config, "online_eval_enabled", False))
        online_eval_datasets = str(getattr(config, "online_eval_datasets", "lipo")).split(",")[0].strip()
        
        if online_eval_enabled and isinstance(loss_history, dict) and "Evaluation_Loss" in loss_history:
            try:
                plot_ssl_and_online_knn(
                    loss_history_path,
                    f"models/{config.name}/loss_curves_ssl_knn.png",
                    model_name=config.name,
                    dataset=online_eval_datasets
                )
                print(f"✓ Dual-axis plot saved: models/{config.name}/loss_curves_ssl_knn.png")
            except (ValueError, KeyError) as e:
                # Fall back to simple plot if online eval data is missing
                print(f"⚠ Could not generate dual-axis plot: {e}")
                if isinstance(loss_history, dict) and "DINO_Loss" in loss_history:
                    loss_data = pd.DataFrame(loss_history["DINO_Loss"])
                    plot_train_val_loss_curves(loss_data, f"models/{config.name}/loss_curves.png", model_name=config.name)
        else:
            # Plot standard train/val loss curves for non-online-eval or old runs
            if isinstance(loss_history, dict) and "DINO_Loss" in loss_history:
                loss_data = pd.DataFrame(loss_history["DINO_Loss"])
            else:
                loss_data = loss_history if isinstance(loss_history, pd.DataFrame) else pd.DataFrame(loss_history)
            plot_train_val_loss_curves(loss_data, f"models/{config.name}/loss_curves.png", model_name=config.name)
            print(f"✓ Standard plot saved: models/{config.name}/loss_curves.png")

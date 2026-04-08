""" Main Entry point for training and evaluation"""

from training.dino_training import dino_train
from model.config import ModelConfig
import torch
import pandas as pd
from plotting.loss_plot import load_loss_data, plot_train_val_loss_curves, plot_ssl_and_linear_probe
from utils.seed import set_seed
from datahandling.dataset_creation import SmilesCsvDataset
from torch.utils.data import DataLoader

config = ModelConfig(
    name="GDZ_GINE_MASKING", # Model identifier (used for saving checkpoints and metadata)
    head_type="dino", # Options: "dino", "regression"
    encoder_type = "GINE", # Options: "GINE", "GAT"
    # Single file mode: provide path to CSV file
    #data_path="data/delaney-processed.csv", # Path to dataset CSV file
    # Multi-file mode: provide path to directory containing .smi files
    data_path="data/zinc/zinc_data",  # Use ZINC dataset (156 files, 1.35M molecules)
    seed = 42, # Random seed for reproducibility
    device= "cuda" if torch.cuda.is_available() else "cpu", # Device to train on (cuda or cpu)
    num_workers=16, # Number of worker processes for data loading (0 = main process)
    local_views=4, # Number of local augmented views per graph (default: 4)
    local_augmentation_mode="masking", # "k_hop" or "masking" for local augmentation
    k_hops=2, # Number of hops for local subgraph extraction (default: num_layers)
    node_mask_ratio=0.15, # Fraction of nodes to mask in masking mode
    feature_mask_ratio=0.15, # Fraction of node features to mask in masking mode
    num_features=24, # Dont change this, its determined by the dataset and dataloader
    edge_features=12, # Dont change this, its determined by the dataset and dataloader
    hidden_dim=128, # GINE hidden dimension
    num_layers=3, # Number of GINE layers
    dropout=0.0, # Dropout rate for GINE
    epsilon=0.0, # Epsilon for GINE
    projection_hidden_dim=256, # Hidden dimension for DINO projection head
    projection_output_dim=128, # Output dimension for DINO projection head
    projection_layers=2, # Number of layers in DINO projection head
    num_epochs=200, # Number of training epochs
    batch_size=128, # Number of graphs per batch (before augmentation) Total views per batch = batch_size * (2 global + 4 local) = batch_size * 6
    auto_scale_lr=True, # Use linear LR scaling from effective batch size
    lr_scale_base=2.5e-4, # Paper base LR used in scaling rule
    lr_scale_reference_batch_size=256, # Reference batch size used in scaling rule
    learning_rate=1e-3, # Ignored when auto_scale_lr=True
    weight_decay=0.04, # Kept for backward compatibility
    weight_decay_start=0.04, # Cosine schedule start for weight decay
    weight_decay_end=0.4, # Cosine schedule end for weight decay
    teacher_temp=0.04, # Temperature for teacher in DINO
    teacher_temp_final=0.07, # Final teacher temperature after warmup
    teacher_temp_warmup_epochs=30, # Warmup epochs for teacher temperature
    student_temp=0.1, # Temperature for student in DINO
    teacher_momentum=0.996, # Momentum for updating teacher parameters in DINO
    center_momentum=0.9, # Momentum for updating teacher center in DINO
    warmup_epochs=10, # Number of epochs for learning rate warmup
    final_learning_rate=1e-5, # Final learning rate after cosine decay
    online_eval_enabled=True, # Evaluate downstream quality during SSL training
    online_eval_every_n_epochs=1, # Run every epoch (can be increased later)
    online_eval_datasets="lipo", # Downstream dataset to track during SSL
    online_eval_fixed_k=5, # Fixed kNN for speed
    online_eval_top_k_checkpoints=5, # Keep top-5 checkpoints by online eval score
    online_eval_linear_probe_enabled=False, # Log linear probe metrics each online eval epoch
    online_eval_linear_probe_alphas="1.0", # Keep linear probe cheap (single alpha/C)
)


if __name__ == "__main__":
    set_seed(config.seed)
    if config.head_type == "dino":
        dino_train(config)
        
        # Plot results
        loss_history_path = f"models/{config.name}/loss_history.json"
        loss_history = load_loss_data(loss_history_path)
        
        # If online eval was enabled, plot dual-axis (SSL loss + linear probe)
        online_eval_enabled = bool(getattr(config, "online_eval_enabled", False))
        online_eval_datasets = str(getattr(config, "online_eval_datasets", "lipo")).split(",")[0].strip()
        
        if online_eval_enabled and isinstance(loss_history, dict) and "Evaluation_Loss" in loss_history:
            try:
                plot_ssl_and_linear_probe(
                    loss_history_path,
                    f"models/{config.name}/loss_curves_ssl_linearprobe.png",
                    model_name=config.name,
                    dataset=online_eval_datasets
                )
                print(f"✓ Dual-axis plot saved: models/{config.name}/loss_curves_ssl_linearprobe.png")
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

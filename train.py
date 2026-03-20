""" Main Entry point for training and evaluation"""

from training.dino_training import dino_train
from model.config import ModelConfig
import torch
import pandas as pd
from plotting.loss_plot import load_loss_data, plot_train_val_loss_curves
from utils.seed import set_seed
from datahandling.dataset_creation import SmilesCsvDataset
from torch.utils.data import DataLoader

config = ModelConfig(
    name="GDZ_2Layer", # Model identifier (used for saving checkpoints and metadata)
    head_type="dino", # Options: "dino", "regression"
    # Single file mode: provide path to CSV file
    #data_path="data/delaney-processed.csv", # Path to dataset CSV file
    # Multi-file mode: provide path to directory containing .smi files
    data_path="data/zinc/zinc_data",  # Use ZINC dataset (156 files, 1.35M molecules)
    seed = 42, # Random seed for reproducibility
    device="cuda" if torch.cuda.is_available() else "cpu", # Device to train on (cuda or cpu)
    num_workers=16, # Number of worker processes for data loading (0 = main process)
    local_views=4, # Number of local augmented views per graph (default: 4)
    k_hops=2, # Number of hops for local subgraph extraction (default: num_layers)
    num_features=20, # Dont change this, its determined by the dataset and dataloader
    edge_features=6, # Dont change this, its determined by the dataset and dataloader
    hidden_dim=128, # GINE hidden dimension
    num_layers=3, # Number of GINE layers
    dropout=0.0, # Dropout rate for GINE
    epsilon=0.0, # Epsilon for GINE
    projection_hidden_dim=256, # Hidden dimension for DINO projection head
    projection_output_dim=128, # Output dimension for DINO projection head
    projection_layers=2, # Number of layers in DINO projection head
    num_epochs=100, # Number of training epochs
    batch_size=1024, # Number of graphs per batch (before augmentation) Total views per batch = batch_size * (2 global + 4 local) = batch_size * 6
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
    final_learning_rate=1e-5 # Final learning rate after cosine decay   
)


if __name__ == "__main__":
    set_seed(config.seed)
    if config.head_type == "dino":
        dino_train(config)
        # Load loss history and extract DINO loss data
        loss_history = load_loss_data(f"models/{config.name}/loss_history.json")
        # Handle both old (list) and new (dict with DINO_Loss key) formats
        if isinstance(loss_history, dict) and "DINO_Loss" in loss_history:
            loss_data = pd.DataFrame(loss_history["DINO_Loss"])
        else:
            # Fallback for old format
            loss_data = loss_history if isinstance(loss_history, pd.DataFrame) else pd.DataFrame(loss_history)
        
        plot_train_val_loss_curves(loss_data, f"models/{config.name}/loss_curves.png", model_name=config.name)

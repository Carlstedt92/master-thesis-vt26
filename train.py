""" Main Entry point for training and evaluation"""

from training.dino_training import dino_train
from model.config import ModelConfig
import torch
from plotting.loss_plot import load_loss_data, plot_train_val_loss_curves
from plotting.t_sne_embeddings import plot_tsne_embeddings
from utils.seed import set_seed
from datahandling.dataset_creation import SmilesCsvDataset
from torch.utils.data import DataLoader

config = ModelConfig(
    name="GINE_DINO", # Model identifier (used for saving checkpoints and metadata)
    head_type="dino", # Options: "dino", "regression"
    data_path="data/delaney-processed.csv", # Path to dataset CSV file
    seed = 42, # Random seed for reproducibility
    device="cuda" if torch.cuda.is_available() else "cpu", # Device to train on (cuda or cpu)
    num_features=20, # Dont change this, its determined by the dataset and dataloader
    edge_features=6, # Dont change this, its determined by the dataset and dataloader
    hidden_dim=128, # GINE hidden dimension
    num_layers=3, # Number of GINE layers
    dropout=0.5, # Dropout rate for GINE
    epsilon=0, # Epsilon for GINE
    projection_hidden_dim=2048, # Hidden dimension for DINO projection head
    projection_output_dim=256, # Output dimension for DINO projection head
    projection_layers=3, # Number of layers in DINO projection head
    num_epochs=40, # Number of training epochs
    batch_size=32, # Number of graphs per batch (before augmentation) Total views per batch = batch_size * (2 global + 4 local) = batch_size * 6
    learning_rate=1e-3, # Learning rate for optimizer
    weight_decay=1e-5, # Weight decay for optimizer
    teacher_temp=0.04, # Temperature for teacher in DINO
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
        loss_data = load_loss_data(f"models/{config.name}/loss_history.json")
        plot_train_val_loss_curves(loss_data, f"models/{config.name}/loss_curves.png", model_name=config.name)
        plot_tsne_embeddings(
            checkpoint_path=f"models/{config.name}/checkpoints/best_model.pth",
            output_path=f"models/{config.name}/tsne_embeddings.png",
            target = "measured log solubility in mols per litre",
            task = "regression",
            smiles_col = "smiles"
        )

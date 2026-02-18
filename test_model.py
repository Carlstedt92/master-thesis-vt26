"""Script for testing the trained model on the test set and evaluating its performance."""

import torch
from model.gine_model import GINEModel
from datahandling.dataset_creation import SmilesCsvDataset
from datahandling.graph_creation import smiles_to_pygdata
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from utils.seed import set_seed
from plotting.loss_plot import plot_train_val_loss_curves
import os
from model.config import ModelConfig
from training.train_manager import TrainingManager
import matplotlib.pyplot as plt

def load_model(model_path, device):
    """Load a trained model from a checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_state_dict = checkpoint['model_state_dict']
    
    # Initialize the model architecture (must match the training config)
    model = GINEModel(
        num_features=checkpoint['config']['num_features'],
        edge_features=checkpoint['config']['edge_features'],
        hidden_dim=checkpoint['config']['hidden_dim'],
        num_layers=checkpoint['config']['num_layers'],
        dropout=checkpoint['config']['dropout'],
        epsilon=checkpoint['config']['epsilon'],
        projection_hidden_dim=checkpoint['config']['projection_hidden_dim'],
        projection_output_dim=checkpoint['config']['projection_output_dim'],
        projection_layers=checkpoint['config']['projection_layers'],
        head_type="regression"  # Load with regression head for downstream evaluation
    ).to(device)
    
    model_state = model.state_dict()
    encoder_only = {k: v for k, v in model_state_dict.items() if k.startswith('encoder')}
    model_state.update(encoder_only)  # Load encoder weights, ignore projection head
    model.load_state_dict(model_state)
    return model, checkpoint

def data_split(dataset, seed):
    """Split dataset into train/val/test sets."""
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)

def create_dataloaders(dataset, batch_size, seed):
    """Create DataLoaders for train/val/test sets."""
    train_dataset, val_dataset, test_dataset = data_split(dataset, seed)
    generator = torch.Generator().manual_seed(seed)
    
    # Use num_workers=0 for deterministic behavior
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=generator,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    return train_loader, val_loader, test_loader

def train_val_loop(model, train_loader, val_loader, optimizer, device, freeze_epochs, manager):
    """Train the model and validate after each epoch."""
    num_epochs = manager.config.num_epochs
    for epoch in range(num_epochs):
        if epoch == freeze_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=manager.config.learning_rate)
            print(f"✓ Encoder unfrozen at epoch {epoch + 1} with LR={manager.config.learning_rate}")
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = torch.nn.functional.mse_loss(outputs, batch.y.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                batch_loss = torch.nn.functional.mse_loss(outputs, batch.y.float())
                val_loss += batch_loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches else 0.0
        manager.record_loss(epoch, avg_loss, val_loss=avg_val_loss)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.6f}, "
            f"Val Loss: {avg_val_loss:.6f}"
        )
    manager.save_loss_history()
    manager.save_metadata()

def evaluate(model, test_loader, device, manager):
    """Evaluate the model on the test set."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test R^2: {r2:.6f}")
    if manager.loss_history:
        manager.loss_history[-1]["test_loss"] = float(mse)
        manager.save_loss_history()
        manager.save_metadata()

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    # Load data for finetuning and testing
    data_path = "data/delaney-processed.csv"
    dataset = SmilesCsvDataset(data_path, smiles_col="smiles", target="measured log solubility in mols per litre", task="regression")

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size=64, seed=SEED)
    print(f"✓ DataLoaders created: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    # Load the trained model
    model_name = "GINE_DINO" # Change this to the name of your trained model
    finetune_name = f"{model_name}_regression"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f"models/{model_name}/checkpoints/best_model.pth"  # Path to the best checkpoint
    model, checkpoint = load_model(model_path, device)
    print(f"✓ Model loaded successfully from {model_path}")

    # Set up training manager for finetuning
    freeze_epochs = 101  # Number of epochs to keep encoder frozen during finetuning
    finetune_lr = 1e-4
    for p in model.encoder.parameters():
        p.requires_grad = False  # Freeze encoder parameters for warmup
    print(f"✓ Encoder parameters frozen for {freeze_epochs} epochs")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    finetune_config = ModelConfig.from_dict(checkpoint["config"])
    finetune_config.name = finetune_name
    finetune_config.head_type = "regression"
    finetune_config.num_epochs = 40
    finetune_config.batch_size = 64
    finetune_config.learning_rate = 1e-3
    manager = TrainingManager(finetune_config)

    # Create randomly initialized GINE model for comparison
    random_model = GINEModel(
        num_features=checkpoint['config']['num_features'],
        edge_features=checkpoint['config']['edge_features'],
        hidden_dim=checkpoint['config']['hidden_dim'],
        num_layers=checkpoint['config']['num_layers'],
        dropout=checkpoint['config']['dropout'],
        epsilon=checkpoint['config']['epsilon'],
        projection_hidden_dim=checkpoint['config']['projection_hidden_dim'],
        projection_output_dim=checkpoint['config']['projection_output_dim'],
        projection_layers=checkpoint['config']['projection_layers'],
        head_type="regression"
    ).to(device)
    random_optimizer = torch.optim.Adam(random_model.parameters(), lr=1e-3)
    random_config = ModelConfig(
        name="GINE_random_regression",
        head_type="regression",
        num_features=checkpoint['config']['num_features'],
        edge_features=checkpoint['config']['edge_features'],
        hidden_dim=checkpoint['config']['hidden_dim'],
        num_layers=checkpoint['config']['num_layers'],
        dropout=checkpoint['config']['dropout'],
        epsilon=checkpoint['config']['epsilon'],
        projection_hidden_dim=checkpoint['config']['projection_hidden_dim'],
        projection_output_dim=checkpoint['config']['projection_output_dim'],
        projection_layers=checkpoint['config']['projection_layers'],
        num_epochs=40,
        batch_size=64,
        learning_rate=1e-3
    )
    random_manager = TrainingManager(random_config)
 
    # Train the regression head on the training set and track loss history
    train_val_loop(model, train_loader, val_loader, optimizer, device, freeze_epochs, manager)
    train_val_loop(random_model, train_loader, val_loader, random_optimizer, device, 0, random_manager)
    # Evaluate the model on the test set
    evaluate(model, test_loader, device, manager)
    evaluate(random_model, test_loader, device, random_manager)
    # Plot training loss curves
    loss_data = pd.DataFrame(manager.loss_history)
    output_path = f"models/{finetune_name}/loss_curves.png"
    plot_train_val_loss_curves(loss_data, output_path, model_name=finetune_name)

    random_loss_data = pd.DataFrame(random_manager.loss_history)
    random_output_path = f"models/{random_config.name}/loss_curves.png"
    plot_train_val_loss_curves(random_loss_data, random_output_path, model_name=random_config.name)

    print(f"✓ Training loss curves saved to {output_path}")
    # Plot loss curves for both models in the same figure for comparison
    fig, ax = plt.subplots(2,1,figsize=(10, 6))
    ax[0].plot(loss_data['epoch'], loss_data['train_loss'], label=f'{finetune_name} Train Loss', marker='o')
    ax[0].plot(loss_data['epoch'], loss_data['val_loss'], label=f'{finetune_name} Val Loss', marker='o')
    ax[0].set_title(f'Training and Validation Loss Curves for {finetune_name}')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(random_loss_data['epoch'], random_loss_data['train_loss'], label=f'{random_config.name} Train Loss', marker='o')
    ax[1].plot(random_loss_data['epoch'], random_loss_data['val_loss'], label=f'{random_config.name} Val Loss', marker='o')
    ax[1].set_title(f'Training and Validation Loss Curves for {random_config.name}')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()

    plt.legend()
    comparison_output_path = f"models/{finetune_name}_vs_{random_config.name}_comparison_loss_curves.png"
    plt.savefig(comparison_output_path)
    plt.close()
    print(f"✓ Comparison loss curves saved to {comparison_output_path}")

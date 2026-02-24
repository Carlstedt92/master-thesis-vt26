"""Generate t-SNE visualization of GINE model embeddings."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.data import Batch
from model.gine_model import GINEModel
from model.config import ModelConfig
from torch.utils.data import DataLoader
from datahandling.dataset_creation import SmilesCsvDataset


def plot_tsne_embeddings(checkpoint_path: str, output_path: str = "tsne_embeddings.png", 
                        num_samples: int = 500, batch_size: int = 32, 
                        random_state: int = 42, target: str = None, task: str = "regression", 
                        smiles_col: str = "smiles", color_label: str = None):
    """
    Generate and plot t-SNE visualization of GINE embeddings from a trained model.
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to the model checkpoint (.pth file)
    output_path : str, default="tsne_embeddings.png"
        Path to save the generated t-SNE plot
    num_samples : int, default=500
        Number of samples to use for t-SNE (limits computation time)
    batch_size : int, default=32
        Batch size for data loading
    random_state : int, default=42
        Random seed for reproducibility
    target : str, optional
        Column name in CSV to use as target values for coloring points.
        If None, all points will be colored uniformly.
    task : str, default="regression"
        Task type ('regression' or 'classification') - affects target data type loading
    smiles_col : str, default="smiles"
        Column name containing SMILES strings
    color_label : str, optional
        Label for the colorbar. If None and target is provided, uses target column name
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Load config from checkpoint
    config_dict = checkpoint["config"]
    config = ModelConfig(**config_dict)
    
    # Initialize model from config
    model = GINEModel(
        num_features=config.num_features,
        edge_features=config.edge_features,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        epsilon=config.epsilon,
        projection_hidden_dim=config.projection_hidden_dim,
        projection_output_dim=config.projection_output_dim,
        projection_layers=config.projection_layers,
        head_type=config.head_type
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(config.device)
    
    # Create dataset with targets
    dataset = SmilesCsvDataset(
        config.data_path, 
        target=target, 
        smiles_col=smiles_col,
        task=task
    )
    
    # Create dataloader with PyG Batch collate
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: Batch.from_data_list(batch)
    )
    
    # Extract embeddings and targets
    embeddings = []
    targets = []
    num_embeddings = 0
    has_targets = target is not None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(config.device)
            batch_embeddings = model.get_embeddings(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            embeddings.append(batch_embeddings.cpu().numpy())
            if has_targets and batch.y is not None:
                targets.append(batch.y.cpu().numpy().flatten())
            num_embeddings += batch_embeddings.shape[0]
            
            if num_embeddings >= num_samples:
                break
    
    # Concatenate and trim to specified number of samples
    embeddings = np.concatenate(embeddings)[:num_samples]
    if has_targets and targets:
        targets = np.concatenate(targets)[:num_samples]
    else:
        targets = None
    
    print(f"✓ Extracted {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    
    # Apply t-SNE
    print("Computing t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=random_state, n_jobs=-1, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot with targets as colors
    plt.figure(figsize=(12, 8))
    
    if targets is not None:
        scatter = plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=targets, 
            cmap='viridis', 
            alpha=0.6, 
            s=20
        )
        label = color_label if color_label else target
        plt.colorbar(scatter, label=label)
        plt.title(f"t-SNE of GINE Embeddings (colored by {label})")
    else:
        plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            alpha=0.6, 
            s=20,
            color='blue'
        )
        plt.title("t-SNE of GINE Embeddings")
    
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"✓ t-SNE plot saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Example usage with target coloring
    model_name = "GINE_DINO"
    checkpoint_path = f"models/{model_name}/checkpoints/best_model.pth"
    output_path = f"models/{model_name}/tsne_embeddings.png"
    
    # Generate t-SNE colored by solubility
    plot_tsne_embeddings(
        checkpoint_path, 
        output_path,
        target="measured log solubility in mols per litre",
        color_label="Log Solubility (mol/L)",
        task="regression"
    )
    
    # Example: Generate t-SNE without coloring (just uncomment)
    # plot_tsne_embeddings(
    #     checkpoint_path,
    #     f"models/{model_name}/tsne_embeddings_plain.png"
    # )
"""Generate t-SNE visualization of GINE model embeddings."""

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
                        random_state: int = 42, target=None, task=None, smiles_col = "smiles"):
    """
    Generate and plot t-SNE visualization of GINE embeddings from a trained model.
    
    Parameters:
    checkpoint_path (str): Path to the model checkpoint (.pth file)
    output_path (str): Path to save the generated t-SNE plot
    num_samples (int): Number of samples to use for t-SNE (limits computation time)
    batch_size (int): Batch size for data loading
    random_state (int): Random seed for reproducibility
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
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(config.device)
            batch_embeddings = model.get_embeddings(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            embeddings.append(batch_embeddings.cpu().numpy())
            targets.append(batch.y.cpu().numpy().flatten())
            num_embeddings += batch_embeddings.shape[0]
            
            if num_embeddings >= num_samples:
                break
    
    # Concatenate and trim to specified number of samples
    embeddings = np.concatenate(embeddings)[:num_samples]
    targets = np.concatenate(targets)[:num_samples]
    
    print(f"✓ Extracted {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    
    # Apply t-SNE
    print("Computing t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=random_state, n_jobs=-1, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot with targets as colors
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=targets, 
        cmap='viridis', 
        alpha=0.6, 
        s=20
    )
    plt.colorbar(scatter, label='Log Solubility (mol/L)')
    plt.title("t-SNE of GINE Embeddings (colored by solubility)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"✓ t-SNE plot saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Example usage
    model_name = "GINE_DINO"
    checkpoint_path = f"models/{model_name}/checkpoints/best_model.pth"
    output_path = f"models/{model_name}/tsne_embeddings.png"
    plot_tsne_embeddings(checkpoint_path, output_path)
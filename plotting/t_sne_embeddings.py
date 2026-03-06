"""Utilities for extracting and plotting t-SNE embeddings."""

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from model.gine_model import GINEModel
from model.config import ModelConfig


def load_model_from_checkpoint(checkpoint_path: str, device: str | None = None):
    """Load a GINE model from checkpoint and return model + config."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(checkpoint["config"])

    model = GINEModel.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    if device is None:
        device = config.device

    model = model.to(device)
    model.eval()
    return model, config


def extract_embeddings(
    model,
    dataset,
    num_samples: int = 500,
    batch_size: int = 32,
    device: str = "cpu",
    include_targets: bool = True,
):
    """Extract graph embeddings from a provided dataset using a provided model."""
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: Batch.from_data_list(batch),
    )

    embeddings = []
    targets = []
    num_embeddings = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_embeddings = model.get_embeddings(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            embeddings.append(batch_embeddings.cpu().numpy())

            if include_targets and hasattr(batch, "y") and batch.y is not None:
                targets.append(batch.y.cpu().numpy().flatten())

            num_embeddings += batch_embeddings.shape[0]
            if num_embeddings >= num_samples:
                break

    embeddings = np.concatenate(embeddings)[:num_samples]
    if include_targets and targets:
        targets = np.concatenate(targets)[:num_samples]
    else:
        targets = None

    print(f"✓ Extracted {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    return embeddings, targets


def plot_tsne_from_embeddings(
    embeddings,
    output_path: str,
    targets=None,
    task: str = "regression",
    random_state: int = 42,
    perplexity: float = 30,
    title: str = "t-SNE of GINE Embeddings",
    color_label: str | None = None,
):
    """Plot t-SNE from pre-computed embeddings and optional targets for coloring."""
    print("Computing t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=random_state, n_jobs=-1, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    if targets is not None:
        if task == "classification":
            class_values = np.unique(targets.astype(int))
            cmap_name = "tab10" if len(class_values) <= 10 else "tab20"
            cmap = plt.get_cmap(cmap_name)

            for class_index, class_value in enumerate(class_values):
                mask = targets.astype(int) == class_value
                plt.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    alpha=0.7,
                    s=20,
                    color=cmap(class_index),
                    label=f"Class {class_value}",
                )

            legend_title = color_label if color_label is not None else "Class"
            plt.legend(title=legend_title)
            plt.title(f"{title} (classification)")
        else:
            scatter = plt.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=targets,
                cmap="viridis",
                alpha=0.6,
                s=20,
            )
            if color_label is None:
                color_label = "Target"
            plt.colorbar(scatter, label=color_label)
            plt.title(f"{title} (colored by {color_label})")
    else:
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.6,
            s=20,
            color="blue",
        )
        plt.title(title)

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✓ t-SNE plot saved: {output_path}")
    plt.close()
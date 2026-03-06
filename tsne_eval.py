"""Standalone script for t-SNE embedding extraction and plotting."""

import torch
from datahandling.dataset_creation import SmilesCsvDataset
from datahandling.graph_creation import smiles_to_pygdata
from torch_geometric.datasets import MoleculeNet
from plotting.t_sne_embeddings import (
    load_model_from_checkpoint,
    extract_embeddings,
    plot_tsne_from_embeddings,
)


if __name__ == "__main__":
    checkpoint_path = "models/GINE_DINO/checkpoints/best_model.pth"
    output_path = "models/GINE_DINO/tsne_embeddings_hiv.png"

    # Create dataset - choose one of the examples below or create your own:
    
    # Example 1: CSV dataset (regression)
    # dataset = SmilesCsvDataset(
    #     data_path="data/delaney-processed.csv",
    #     target="measured log solubility in mols per litre",
    #     smiles_col="smiles",
    #     task="regression",
    # )
    # target = "measured log solubility in mols per litre"
    # task = "regression"
    # color_label = "Log Solubility (mol/L)"

    # Example 2: MoleculeNet HIV dataset (classification)
    dataset = MoleculeNet(
        root="data/MoleculeNet_HIV_custom",
        name="HIV",
        from_smiles=smiles_to_pygdata,
    )
    target = "y"
    task = "classification"
    color_label = "HIV Class"

    # t-SNE parameters
    num_samples = 500
    batch_size = 32
    random_state = 42
    perplexity = 30.0
    device = None

    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path, device=device)
    device = device if device is not None else config.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Extract embeddings
    embeddings, targets = extract_embeddings(
        model=model,
        dataset=dataset,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
        include_targets=target is not None,
    )

    # Plot t-SNE
    model_name = config.name if hasattr(config, "name") else "Model"
    title = f"t-SNE of GINE Embeddings - {model_name}"

    plot_tsne_from_embeddings(
        embeddings=embeddings,
        output_path=output_path,
        targets=targets,
        task=task,
        random_state=random_state,
        perplexity=perplexity,
        title=title,
        color_label=color_label if color_label else target,
    )

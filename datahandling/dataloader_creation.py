"""Create DataLoader for PyG Dataset with graph augmentations."""

from torch_geometric.data import Data, Batch
from .graph_augmentation import GraphAugmentation
from .dataset_creation import SmilesCsvDataset, MultiFileSmilesDataset
import torch
from torch.utils.data import DataLoader
from typing import List
import os


def collate_fn(batch: List[Data]):
    """Apply augmentation to each graph and flatten into a single batch."""
    augmenter = GraphAugmentation(local_views=4)
    augmented = [augmenter(data) for data in batch]
    flat: List[Data] = [view for views in augmented for view in views]
    return Batch.from_data_list(flat)

def create_dataloader(csv_path: str, batch_size: int = 32,
                      shuffle: bool = True, seed: int | None = None) -> DataLoader:
    dataset = SmilesCsvDataset(csv_path, smiles_col="smiles")
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=generator,
        num_workers=0  # Ensure deterministic behavior
    )
    return loader


def create_dataloader_auto(data_path: str, batch_size: int = 32,
                           shuffle: bool = True, seed: int | None = None,
                           num_workers: int = 0) -> DataLoader:
    """Auto-detect whether data_path is a file or directory and create appropriate dataloader.
    
    Args:
        data_path: Path to either a CSV file or directory containing .smi files
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
    
    Returns:
        DataLoader that works with either single file or multi-file datasets
    """
    if os.path.isdir(data_path):
        # Multi-file mode: directory containing .smi files
        print(f"✓ Detected directory mode: loading from {data_path}")
        return create_multifile_dataloader(
            data_dir=data_path,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            num_workers=num_workers
        )
    elif os.path.isfile(data_path):
        # Single file mode: CSV file
        print(f"✓ Detected single file mode: loading from {data_path}")
        return create_dataloader(
            csv_path=data_path,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed
        )
    else:
        raise ValueError(f"data_path must be either a file or directory, got: {data_path}")


def create_multifile_dataloader(data_dir: str, batch_size: int = 32,
                                 shuffle: bool = True, seed: int | None = None,
                                 pattern: str = "*.smi", num_workers: int = 0) -> DataLoader:
    """Create a DataLoader for multiple SMILES files in a directory.
    
    Args:
        data_dir: Directory containing .smi files
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        pattern: Glob pattern to match files (default: *.smi)
        num_workers: Number of worker processes for data loading (default: 0)
    
    Returns:
        DataLoader that lazily loads from multiple files
    """
    dataset = MultiFileSmilesDataset(data_dir, smiles_col="smiles", pattern=pattern)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=generator,
        num_workers=num_workers,
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    return loader

# Example usage:
if __name__ == "__main__":
    # Single file example
    loader = create_dataloader("data/delaney-processed.csv", batch_size=16)
    for batch in loader:
        print("Single file batch:", batch)
        break
    
    # Multi-file example
    multi_loader = create_multifile_dataloader("data/zinc/zinc_data", batch_size=16)
    for batch in multi_loader:
        print("Multi-file batch:", batch)
        break
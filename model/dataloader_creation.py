"""Create DataLoader for PyG Dataset with graph augmentations."""

from torch_geometric.data import Data, Batch
from .graph_augmentation import GraphAugmentation
from .dataset_creation import SmilesCsvDataset
from torch.utils.data import DataLoader
from typing import List


def collate_fn(batch: List[Data]):
    """Apply augmentation to each graph and flatten into a single batch."""
    augmenter = GraphAugmentation(local_views=4)
    augmented = [augmenter(data) for data in batch]
    flat: List[Data] = [view for views in augmented for view in views]
    return Batch.from_data_list(flat)

def create_dataloader(csv_path: str, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    dataset = SmilesCsvDataset(csv_path, smiles_col="smiles")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return loader

# Example usage:
if __name__ == "__main__":
    loader = create_dataloader("data/delaney-processed.csv", batch_size=16)
    for batch in loader:
        print(batch)
        break
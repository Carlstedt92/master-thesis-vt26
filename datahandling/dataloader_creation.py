"""Create DataLoader for PyG Dataset with graph augmentations."""

from torch_geometric.data import Data, Batch
from .graph_augmentation import GraphAugmentation
from .dataset_creation import SmilesCsvDataset, MultiFileSmilesDataset
import torch
from torch.utils.data import DataLoader
from typing import List, Optional
import os
import time


class DataLoaderCreator:
    """Create DataLoaders using values from a stored config object."""
    
    def __init__(self, config):
        """Initialize with configuration object.
        
        Args:
            config: Configuration object with augmentation parameters (e.g., num_layers for k_hops)
        """
        self.config = config
    
    def _get_collate_fn(self):
        """Build collate function using augmentation settings from config."""
        def _normalize_dtypes(data: Data) -> Data:
            """Enforce stable tensor dtypes expected by PyG batching and model code."""
            if hasattr(data, 'x') and data.x is not None:
                data.x = data.x.float()
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = data.edge_attr.float()
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                data.edge_index = data.edge_index.long()
            if hasattr(data, 'graph_idx') and data.graph_idx is not None:
                data.graph_idx = data.graph_idx.long()
            if 'view' in data and data['view'] is not None:
                data['view'] = data['view'].long()
            return data

        def collate_fn(batch: List[Optional[Data]]):
            """Apply augmentation to each graph and flatten into a single batch."""
            valid_batch = [data for data in batch if data is not None]
            if not valid_batch:
                return None

            k_hops = getattr(self.config, 'k_hops', 2)
            local_views = getattr(self.config, 'local_views', 4)
            augmenter = GraphAugmentation(local_views=local_views, k_hops=k_hops)
            augmented = [augmenter(data) for data in valid_batch]
            flat: List[Data] = [_normalize_dtypes(view_data) for views in augmented for view_data in views]
            result = Batch.from_data_list(flat)
            return result
        return collate_fn

    def _build_generator(self):
        if self.config.seed is None:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        return generator
    
    def create_dataloader(self) -> DataLoader:
        """Create DataLoader for a single CSV file using stored config.
        
        Reads csv_path, batch_size, and seed from config.
        
        Returns:
            DataLoader instance
        """
        dataset = SmilesCsvDataset(self.config.data_path, smiles_col="smiles")
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._get_collate_fn(),
            generator=self._build_generator(),
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0
        )

    def create_dataloader_auto(self) -> DataLoader:
        """Auto-detect whether data_path is a file or directory and create appropriate dataloader.
        
        Reads data_path, batch_size, seed, and num_workers from stored config.

        Returns:
            DataLoader that works with either single file or multi-file datasets
        """
        data_path = self.config.data_path
        if os.path.isdir(data_path):
            # Multi-file mode: directory containing .smi files
            print(f"✓ Detected directory mode: loading from {data_path}")
            return self.create_multifile_dataloader()
        elif os.path.isfile(data_path):
            # Single file mode: CSV file
            print(f"✓ Detected single file mode: loading from {data_path}")
            return self.create_dataloader()
        else:
            raise ValueError(f"data_path must be either a file or directory, got: {data_path}")

    def create_multifile_dataloader(self, pattern: str = "*.smi") -> DataLoader:
        """Create a DataLoader for multiple SMILES files in a directory.
        
        Reads data_path, batch_size, seed, and num_workers from stored config.
        
        Args:
            pattern: Glob pattern to match files (default: *.smi)
        
        Returns:
            DataLoader that lazily loads from multiple files
        """
        dataset = MultiFileSmilesDataset(self.config.data_path, smiles_col="smiles", pattern=pattern)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._get_collate_fn(),
            generator=self._build_generator(),
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0  # Keep workers alive between epochs
        )
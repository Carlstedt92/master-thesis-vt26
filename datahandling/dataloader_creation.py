"""Create DataLoader for PyG Dataset with graph augmentations."""

from torch_geometric.data import Data, Batch
from .graph_augmentation import GraphAugmentation
from .dataset_creation import SmilesCsvDataset, MultiFileSmilesDataset, PrecomputedGraphDataset
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
        k_hops = getattr(self.config, 'k_hops', 2)
        local_views = getattr(self.config, 'local_views', 4)
        local_augmentation_mode = getattr(self.config, 'local_augmentation_mode', 'k_hop')
        node_mask_ratio = getattr(self.config, 'node_mask_ratio', 0.15)
        feature_mask_ratio = getattr(self.config, 'feature_mask_ratio', 0.15)
        augmenter = GraphAugmentation(
            local_views=local_views,
            k_hops=k_hops,
            local_augmentation_mode=local_augmentation_mode,
            node_mask_ratio=node_mask_ratio,
            feature_mask_ratio=feature_mask_ratio,
        )

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
            profile_enabled = self._loader_debug() or bool(getattr(self.config, "profile_timing", False))
            profile = {}
            collate_start = time.time() if profile_enabled else None

            filter_start = time.time() if profile_enabled else None
            valid_batch = [data for data in batch if data is not None]
            if profile_enabled:
                profile["filter_invalid"] = time.time() - filter_start
            if not valid_batch:
                return None

            aug_start = time.time() if profile_enabled else None
            augmented = [augmenter(data) for data in valid_batch]
            if profile_enabled:
                profile["augmentation"] = time.time() - aug_start

            normalize_start = time.time() if profile_enabled else None
            flat: List[Data] = [_normalize_dtypes(view_data) for views in augmented for view_data in views]
            if profile_enabled:
                profile["normalize_flatten"] = time.time() - normalize_start

            batch_start = time.time() if profile_enabled else None
            result = Batch.from_data_list(flat)
            if profile_enabled:
                profile["batch_from_data_list"] = time.time() - batch_start
                profile["collate_total"] = time.time() - collate_start
                result.profile_timing = profile
            return result
        return collate_fn

    def _build_generator(self):
        if self.config.seed is None:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        return generator

    def _cache_in_memory(self) -> bool:
        return bool(getattr(self.config, "cache_data_in_memory", False))

    def _precomputed_cache_in_memory(self) -> bool:
        return bool(
            getattr(
                self.config,
                "precomputed_cache_in_memory",
                self._cache_in_memory(),
            )
        )

    def _precomputed_max_cached_shards(self) -> int:
        return int(getattr(self.config, "precomputed_max_cached_shards", 4))

    def _loader_debug(self) -> bool:
        return bool(getattr(self.config, "loader_debug", False))
    
    def create_dataloader(self) -> DataLoader:
        """Create DataLoader for a single CSV file using stored config.
        
        Reads csv_path, batch_size, and seed from config.
        
        Returns:
            DataLoader instance
        """
        dataset = SmilesCsvDataset(
            self.config.data_path,
            smiles_col="smiles",
            cache_in_memory=self._cache_in_memory(),
        )
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
        """Create dataloader based on config mode.
        
        If ``use_precomputed`` is enabled, reads precomputed graph shards.
        Otherwise auto-detects whether ``data_path`` is a file or directory.

        Returns:
            DataLoader that works with either single file or multi-file datasets
        """
        if bool(getattr(self.config, "use_precomputed", False)):
            precomputed_path = str(getattr(self.config, "precomputed_data_path", "")).strip()
            if not precomputed_path:
                raise ValueError(
                    "use_precomputed=True but precomputed_data_path is empty in config"
                )
            print(f"✓ Precomputed mode enabled: loading from {precomputed_path}")
            return self.create_precomputed_dataloader(precomputed_path)

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
        dataset = MultiFileSmilesDataset(
            self.config.data_path,
            smiles_col="smiles",
            pattern=pattern,
            cache_in_memory=self._cache_in_memory(),
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._get_collate_fn(),
            generator=self._build_generator(),
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0  # Keep workers alive between epochs
        )

    def create_precomputed_dataloader(self, precomputed_path: str, pattern: str = "shard_*.pt") -> DataLoader:
        """Create DataLoader for precomputed PyG graph shards.

        Args:
            precomputed_path: Directory containing shard files and optional metadata.json
            pattern: Glob pattern for shard files

        Returns:
            DataLoader that reads precomputed base graphs and applies online augmentation
        """
        if self._loader_debug():
            print(f"[DataLoaderCreator] Building PrecomputedGraphDataset from {precomputed_path}")
        build_start = time.time()
        dataset = PrecomputedGraphDataset(
            precomputed_path,
            pattern=pattern,
            cache_in_memory=self._precomputed_cache_in_memory(),
            max_cached_shards=self._precomputed_max_cached_shards(),
            debug=self._loader_debug(),
        )
        if self._loader_debug():
            print(f"[DataLoaderCreator] Dataset built in {time.time() - build_start:.2f}s")
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._get_collate_fn(),
            generator=self._build_generator(),
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0,
        )
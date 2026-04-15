"""Create DataLoader for PyG Dataset with graph augmentations."""

from torch_geometric.data import Data, Batch
from .graph_augmentation import GraphAugmentation
from .dataset_creation import SmilesCsvDataset, MultiFileSmilesDataset, PrecomputedGraphDataset
import torch
from torch.utils.data import DataLoader
from typing import List, Optional
import os
import time
import random
import random


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

        if local_augmentation_mode == "masking" and self._use_batched_mask_collate():
            return self._build_batched_mask_collate_fn()

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

    def _use_batched_mask_collate(self) -> bool:
        return bool(getattr(self.config, "use_batched_mask_collate", False))

    def _explicit_hydrogens(self) -> bool:
        return bool(getattr(self.config, "explicit_hydrogens", True))

    def _encode_hydrogen_count(self) -> bool:
        return bool(getattr(self.config, "encode_hydrogen_count", False))

    def _concat_view_batches(self, view_batches: List[Batch]) -> Batch:
        """Concatenate full-batch view copies into one final PyG Batch."""
        if not view_batches:
            return None

        combined = Batch()
        x_parts = []
        edge_index_parts = []
        edge_attr_parts = []
        batch_parts = []
        graph_idx_parts = []
        view_parts = []

        node_offset = 0
        graphs_per_view = view_batches[0].num_graphs

        for view_offset, view_batch in enumerate(view_batches):
            x_parts.append(view_batch.x)

            if view_batch.edge_index is not None:
                edge_index_parts.append(view_batch.edge_index + node_offset)
            if view_batch.edge_attr is not None:
                edge_attr_parts.append(view_batch.edge_attr)

            batch_parts.append(view_batch.batch + (view_offset * graphs_per_view))
            graph_idx_parts.append(view_batch.graph_idx)
            view_parts.append(view_batch['view'])

            node_offset += view_batch.x.size(0)

        combined.x = torch.cat(x_parts, dim=0)
        combined.edge_index = torch.cat(edge_index_parts, dim=1) if edge_index_parts else None
        combined.edge_attr = torch.cat(edge_attr_parts, dim=0) if edge_attr_parts else None
        combined.batch = torch.cat(batch_parts, dim=0)
        combined.graph_idx = torch.cat(graph_idx_parts, dim=0)
        combined['view'] = torch.cat(view_parts, dim=0)
        combined._num_graphs = graphs_per_view * len(view_batches)
        return combined

    def _build_batched_mask_collate_fn(self):
        """Fast-path collate for masking mode that batches once then masks tensors."""
        local_views = getattr(self.config, 'local_views', 4)
        node_mask_ratio = getattr(self.config, 'node_mask_ratio', 0.15)
        feature_mask_ratio = getattr(self.config, 'feature_mask_ratio', 0.15)

        def _sample_count(total_items: int, ratio: float) -> int:
            if total_items <= 0 or ratio <= 0:
                return 0
            return min(total_items, max(1, int(round(total_items * ratio))))

        def _mask_batched_local_view(base_batch: Batch) -> Batch:
            masked = base_batch.clone()
            if masked.x is None:
                masked['view'] = torch.zeros(masked.num_graphs, dtype=torch.long)
                return masked

            x = masked.x.clone()
            ptr = masked.ptr
            num_graphs = masked.num_graphs
            num_features = x.size(1) if x.dim() > 1 else 0

            for graph_idx in range(num_graphs):
                start = int(ptr[graph_idx].item())
                end = int(ptr[graph_idx + 1].item())
                num_nodes = end - start

                num_node_mask = _sample_count(num_nodes, node_mask_ratio)
                if num_node_mask > 0:
                    mask_nodes = random.sample(range(start, end), num_node_mask)
                    x[mask_nodes] = 0

                num_feature_mask = _sample_count(num_features, feature_mask_ratio)
                if num_feature_mask > 0:
                    feature_mask_indices = random.sample(range(num_features), num_feature_mask)
                    x[start:end, feature_mask_indices] = 0

            masked.x = x
            masked['view'] = torch.zeros(masked.num_graphs, dtype=torch.long)
            return masked

        def collate_fn(batch: List[Optional[Data]]):
            profile_enabled = self._loader_debug() or bool(getattr(self.config, "profile_timing", False))
            profile = {}
            collate_start = time.time() if profile_enabled else None

            filter_start = time.time() if profile_enabled else None
            valid_batch = [data for data in batch if data is not None]
            if profile_enabled:
                profile["filter_invalid"] = time.time() - filter_start
            if not valid_batch:
                return None

            base_batch_start = time.time() if profile_enabled else None
            base_batch = Batch.from_data_list(valid_batch)
            if profile_enabled:
                profile["base_batch_from_data_list"] = time.time() - base_batch_start

            global_batches: List[Batch] = []
            global_start = time.time() if profile_enabled else None
            for _ in range(2):
                global_batch = base_batch.clone()
                global_batch['view'] = torch.ones(global_batch.num_graphs, dtype=torch.long)
                global_batches.append(global_batch)
            if profile_enabled:
                profile["global_clone"] = time.time() - global_start

            local_batches: List[Batch] = []
            local_start = time.time() if profile_enabled else None
            for _ in range(local_views):
                local_batches.append(_mask_batched_local_view(base_batch))
            if profile_enabled:
                profile["local_masking"] = time.time() - local_start

            combine_start = time.time() if profile_enabled else None
            result = self._concat_view_batches(global_batches + local_batches)
            if profile_enabled:
                profile["combine_views"] = time.time() - combine_start
                profile["collate_total"] = time.time() - collate_start
                result.profile_timing = profile
            return result

        return collate_fn
    
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
            explicit_hydrogens=self._explicit_hydrogens(),
            encode_hydrogen_count=self._encode_hydrogen_count(),
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
            explicit_hydrogens=self._explicit_hydrogens(),
            encode_hydrogen_count=self._encode_hydrogen_count(),
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
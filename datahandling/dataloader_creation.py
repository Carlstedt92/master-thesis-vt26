"""Create DataLoader for PyG Dataset with graph augmentations."""

from torch_geometric.data import Data, Batch
from .graph_augmentation import GraphAugmentation
from .dataset_creation import SmilesCsvDataset, MultiFileSmilesDataset, PrecomputedGraphDataset
from .shard_sampler import ShardAwareBatchSampler
import torch
from torch.utils.data import DataLoader, random_split
from typing import List, Optional, Tuple
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
        local_view_modes = getattr(self.config, 'local_view_modes', None)
        augmenter = GraphAugmentation(
            local_views=local_views,
            k_hops=k_hops,
            local_augmentation_mode=local_augmentation_mode,
            node_mask_ratio=node_mask_ratio,
            feature_mask_ratio=feature_mask_ratio,
            local_view_modes=local_view_modes,
        )

        if local_augmentation_mode == "masking" and self._use_batched_mask_collate():
            if bool(getattr(self.config, "use_node_reconstruction_loss", False)):
                raise ValueError(
                    "use_node_reconstruction_loss=True is not supported together with "
                    "local_augmentation_mode='masking' + use_batched_mask_collate=True: "
                    "that fast path masks an already-batched tensor directly and bypasses "
                    "GraphAugmentation entirely, so it never produces the node_reconstruction_mask "
                    "attribute the loss needs. Use 'functional_group_masking' (always routed "
                    "through GraphAugmentation regardless of this flag) or set "
                    "use_batched_mask_collate=False."
                )
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

    def _split_validation_dataset(self, dataset):
        validation_enabled = bool(getattr(self.config, "validation_enabled", False))
        validation_split = float(getattr(self.config, "validation_split", 0.0))

        if not validation_enabled or validation_split <= 0.0:
            return dataset, None

        total_items = len(dataset)
        if total_items < 2:
            return dataset, None

        val_size = int(round(total_items * validation_split))
        val_size = max(1, min(val_size, total_items - 1))
        train_size = total_items - val_size

        generator = self._build_generator()
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        return train_dataset, val_dataset

    def _build_loader(self, dataset, shuffle: bool) -> DataLoader:
        pin_memory = str(getattr(self.config, "device", "cpu")).startswith("cuda")
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "collate_fn": self._get_collate_fn(),
            "generator": self._build_generator(),
            "num_workers": self.config.num_workers,
            "persistent_workers": self.config.num_workers > 0,
            "pin_memory": pin_memory,
        }
        if self.config.num_workers > 0:
            loader_kwargs["prefetch_factor"] = int(getattr(self.config, "prefetch_factor", 4))
        return DataLoader(
            **loader_kwargs,
        )

    def _ddp_rank_world_size(self) -> Tuple[int, int]:
        """DDP rank/world_size for this process, set by dino_training.py before loader
        creation when running under torchrun. Defaults to single-process (0, 1)."""
        rank = int(getattr(self.config, "ddp_rank", 0))
        world_size = int(getattr(self.config, "ddp_world_size", 1))
        return rank, world_size

    def _build_precomputed_loader(
        self, dataset, shard_ids: List[int], shuffle: bool, partition_across_ranks: bool = True
    ) -> DataLoader:
        """Build a DataLoader over a PrecomputedGraphDataset using shard-local batching.

        partition_across_ranks=False keeps every DDP rank on the full shard_ids
        list (rank=0, world_size=1 for sampling purposes) -- used for the
        validation loader, since a small validation split can have fewer
        shards than ranks. Splitting it would leave some ranks with zero
        shards; even in eval() mode a DDP-wrapped forward() still does a
        buffer broadcast every call (independent of no_grad/eval state), so a
        rank that runs zero eval iterations while another runs dozens
        desyncs the collective call count and hangs the whole process group.
        Redundant (whole-set) validation compute is a fine trade for that.
        """
        pin_memory = str(getattr(self.config, "device", "cpu")).startswith("cuda")
        rank, world_size = self._ddp_rank_world_size() if partition_across_ranks else (0, 1)
        batch_sampler = ShardAwareBatchSampler(
            cumulative_sizes=dataset.cumulative_sizes,
            shard_ids=shard_ids,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            seed=self.config.seed,
            rank=rank,
            world_size=world_size,
        )
        loader_kwargs = {
            "dataset": dataset,
            "batch_sampler": batch_sampler,
            "collate_fn": self._get_collate_fn(),
            "num_workers": self.config.num_workers,
            "persistent_workers": self.config.num_workers > 0,
            "pin_memory": pin_memory,
        }
        if self.config.num_workers > 0:
            loader_kwargs["prefetch_factor"] = int(getattr(self.config, "prefetch_factor", 4))
        return DataLoader(**loader_kwargs)

    def _split_shards_for_validation(
        self, num_shards: int, validation_split: float, seed: Optional[int]
    ) -> Tuple[List[int], List[int]]:
        """Reserve whole shards for validation instead of splitting by item.

        Keeps every shard's index range fully contiguous to either the train
        or validation sampler, which the shard-aware batch sampler relies on.
        """
        if num_shards < 2 or validation_split <= 0.0:
            return list(range(num_shards)), []

        val_shard_count = max(1, min(num_shards - 1, int(round(num_shards * validation_split))))
        shard_ids = list(range(num_shards))
        rng = random.Random(seed) if seed is not None else random.Random()
        rng.shuffle(shard_ids)
        val_shard_ids = sorted(shard_ids[:val_shard_count])
        train_shard_ids = sorted(shard_ids[val_shard_count:])
        return train_shard_ids, val_shard_ids

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
        return self._build_loader(dataset, shuffle=True)

    def create_train_val_dataloaders(self):
        """Create train and validation dataloaders for a single CSV file."""
        dataset = SmilesCsvDataset(
            self.config.data_path,
            smiles_col="smiles",
            cache_in_memory=self._cache_in_memory(),
            explicit_hydrogens=self._explicit_hydrogens(),
            encode_hydrogen_count=self._encode_hydrogen_count(),
        )
        train_dataset, val_dataset = self._split_validation_dataset(dataset)
        train_loader = self._build_loader(train_dataset, shuffle=True)
        val_loader = self._build_loader(val_dataset, shuffle=False) if val_dataset is not None else None
        return train_loader, val_loader

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
        return self._build_loader(dataset, shuffle=True)

    def create_multifile_train_val_dataloaders(self, pattern: str = "*.smi"):
        """Create train and validation loaders for multiple SMILES files."""
        dataset = MultiFileSmilesDataset(
            self.config.data_path,
            smiles_col="smiles",
            pattern=pattern,
            cache_in_memory=self._cache_in_memory(),
            explicit_hydrogens=self._explicit_hydrogens(),
            encode_hydrogen_count=self._encode_hydrogen_count(),
        )
        train_dataset, val_dataset = self._split_validation_dataset(dataset)
        train_loader = self._build_loader(train_dataset, shuffle=True)
        val_loader = self._build_loader(val_dataset, shuffle=False) if val_dataset is not None else None
        return train_loader, val_loader

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
        shard_ids = list(range(len(dataset.shard_paths)))
        return self._build_precomputed_loader(dataset, shard_ids, shuffle=True)

    def create_precomputed_train_val_dataloaders(self, precomputed_path: str, pattern: str = "shard_*.pt"):
        """Create train and validation loaders for precomputed graph shards.

        Validation is a whole-shard split (not a per-item random_split) so
        every shard's index range stays contiguous for the shard-aware sampler.
        """
        dataset = PrecomputedGraphDataset(
            precomputed_path,
            pattern=pattern,
            cache_in_memory=self._precomputed_cache_in_memory(),
            max_cached_shards=self._precomputed_max_cached_shards(),
            debug=self._loader_debug(),
        )
        num_shards = len(dataset.shard_paths)
        validation_enabled = bool(getattr(self.config, "validation_enabled", False))
        validation_split = float(getattr(self.config, "validation_split", 0.0))

        if not validation_enabled or validation_split <= 0.0:
            train_shard_ids = list(range(num_shards))
            val_shard_ids: List[int] = []
        else:
            train_shard_ids, val_shard_ids = self._split_shards_for_validation(
                num_shards, validation_split, getattr(self.config, "seed", None)
            )

        train_loader = self._build_precomputed_loader(dataset, train_shard_ids, shuffle=True)
        # Partitioned across ranks like the train loader (partition_across_ranks
        # defaults to True) -- eval_step bypasses the DDP wrapper entirely (see
        # DINOGraphSSL._forward_student_eval), so there's no per-batch collective
        # tied to validation anymore, and each rank's slice is aggregated via a
        # single _ddp_sum reduction after the loop (dino_training.py). This is
        # what actually lets validation benefit from multi-GPU parallelism
        # instead of running single-rank while the rest of the GPUs sit idle.
        val_loader = (
            self._build_precomputed_loader(dataset, val_shard_ids, shuffle=False)
            if val_shard_ids
            else None
        )
        return train_loader, val_loader

    def create_train_val_dataloaders_auto(self):
        """Create train and validation dataloaders based on config mode."""
        if bool(getattr(self.config, "use_precomputed", False)):
            precomputed_path = str(getattr(self.config, "precomputed_data_path", "")).strip()
            if not precomputed_path:
                raise ValueError(
                    "use_precomputed=True but precomputed_data_path is empty in config"
                )
            print(f"✓ Precomputed mode enabled: loading from {precomputed_path}")
            return self.create_precomputed_train_val_dataloaders(precomputed_path)

        data_path = self.config.data_path
        if os.path.isdir(data_path):
            print(f"✓ Detected directory mode: loading from {data_path}")
            return self.create_multifile_train_val_dataloaders()
        elif os.path.isfile(data_path):
            print(f"✓ Detected single file mode: loading from {data_path}")
            return self.create_train_val_dataloaders()
        else:
            raise ValueError(f"data_path must be either a file or directory, got: {data_path}")
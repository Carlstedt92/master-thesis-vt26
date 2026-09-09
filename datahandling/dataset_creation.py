"""Create datasets for on-the-fly SMILES parsing or precomputed PyG graph loading."""
from torch_geometric.data import Data
from torch.utils.data import Dataset
import csv
import torch
from .graph_creation import smiles_to_pygdata
from typing import List, Sequence, Union, Tuple, Optional
import os
import glob
import bisect
import json
from collections import OrderedDict
import time


def _load_torch_data(path: str):
    """Load trusted local data objects across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support the weights_only argument.
        return torch.load(path, map_location="cpu")

class SmilesCsvDataset(Dataset):
    """Lazy dataset: keep SMILES on disk, build graphs on demand."""

    def __init__(self, csv_path: str, smiles_col: str = "smiles",
                 target: Union[str, Sequence[str], None] = None,
                 task: str = "regression",
                 cache_in_memory: bool = False,
                 explicit_hydrogens: bool = True,
                 encode_hydrogen_count: bool = False) -> None:
        self.csv_path = csv_path
        self.smiles_col = smiles_col
        self.target = target
        self.task = task
        self.cache_in_memory = cache_in_memory
        self.explicit_hydrogens = explicit_hydrogens
        self.encode_hydrogen_count = encode_hydrogen_count
        self._index = self._build_index()
        self._fieldnames = self._get_fieldnames()
        self._rows_cache: Optional[List[dict]] = None
        if self.cache_in_memory:
            self._rows_cache = self._load_rows_into_memory()

    def _load_rows_into_memory(self) -> List[dict]:
        """Read all rows once so __getitem__ can avoid reopening files."""
        rows: List[dict] = []
        with open(self.csv_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
        print(f"Initialized SmilesCsvDataset RAM cache with {len(rows)} rows")
        return rows

    def _get_fieldnames(self) -> List[str]:
        """Read CSV header to get field names."""
        with open(self.csv_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            return reader.fieldnames

    def _build_index(self) -> List[int]:
        """Store file offsets for each data row (after header)."""
        offsets: List[int] = []
        with open(self.csv_path, "r", newline="") as handle:
            header = handle.readline()
            if not header:
                return []
            # Store offset of first data row
            offsets.append(handle.tell())
            while handle.readline():
                offsets.append(handle.tell())
        # Remove the last offset (it's past the last row)
        return offsets[:-1] if offsets else []

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Optional[Data]:
        if self._rows_cache is not None:
            row = self._rows_cache[idx]
        else:
            with open(self.csv_path, "r", newline="") as handle:
                handle.seek(self._index[idx])
                reader = csv.DictReader(handle, fieldnames=self._fieldnames)
                row = next(reader)
        data = smiles_to_pygdata(
            row[self.smiles_col],
            explicit_hydrogens=self.explicit_hydrogens,
            encode_hydrogen_count=self.encode_hydrogen_count,
        )
        if data is None:
            return None

        if self.target:
            is_regression = self.task == "regression"
            dtype = torch.float if is_regression else torch.long
            if isinstance(self.target, (list, tuple)):
                target_values = [float(row[name]) for name in self.target]
                if dtype is torch.long:
                    target_values = [int(value) for value in target_values]
                data.y = torch.tensor([target_values], dtype=dtype)
            else:
                target_value = float(row[self.target])
                if dtype is torch.long:
                    target_value = int(target_value)
                if is_regression:
                    data.y = torch.tensor([[target_value]], dtype=dtype)
                else:
                    data.y = torch.tensor([target_value], dtype=dtype)
        data.graph_idx = torch.tensor([idx])  # Tensor for proper PyG batching
        return data


class MultiFileSmilesDataset(Dataset):
    """Lazy dataset for multiple .smi files: keeps SMILES on disk, builds graphs on demand.
    
    Designed for large-scale datasets split across multiple files (e.g., ZINC database).
    Each file is assumed to have a header line and whitespace-separated columns.
    """

    def __init__(self, data_dir: str, smiles_col: str = "smiles",
                 pattern: str = "*.smi",
                 cache_in_memory: bool = False,
                 explicit_hydrogens: bool = True,
                 encode_hydrogen_count: bool = False) -> None:
        """
        Args:
            data_dir: Directory containing .smi files
            smiles_col: Name of the column containing SMILES strings
            pattern: Glob pattern to match files (default: *.smi)
        """
        self.data_dir = data_dir
        self.smiles_col = smiles_col
        self.pattern = pattern
        self.cache_in_memory = cache_in_memory
        self.explicit_hydrogens = explicit_hydrogens
        self.encode_hydrogen_count = encode_hydrogen_count
        self._rows_cache: Optional[List[dict]] = [] if cache_in_memory else None
        
        # Discover all matching files
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not self.file_paths:
            raise ValueError(f"No files found matching pattern {pattern} in {data_dir}")
        
        # Build index: list of (file_path, offset, fieldnames)
        self._index: List[Tuple[str, int, List[str]]] = []
        self._build_index()
        
        print(f"Initialized MultiFileSmilesDataset with {len(self.file_paths)} files, "
              f"{len(self._index)} total samples")

    def _build_index(self) -> None:
        """Build index mapping global indices to (file_path, offset, fieldnames)."""
        for file_path in self.file_paths:
            with open(file_path, "r") as handle:
                # Read header
                header = handle.readline().strip()
                if not header:
                    continue
                
                # Parse fieldnames (whitespace-delimited)
                fieldnames = header.split()
                
                # Store offset of each data row
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    # Only add if line is not empty
                    if line.strip():
                        if self._rows_cache is not None:
                            values = line.split()
                            self._rows_cache.append(dict(zip(fieldnames, values)))
                        else:
                            self._index.append((file_path, offset, fieldnames))

        if self._rows_cache is not None:
            # Keep length/iteration path consistent with the offset-based version.
            self._index = []
            print(f"Initialized MultiFileSmilesDataset RAM cache with {len(self._rows_cache)} rows")

    def __len__(self) -> int:
        if self._rows_cache is not None:
            return len(self._rows_cache)
        return len(self._index)

    def __getitem__(self, idx: int) -> Optional[Data]:
        """Load a single graph by reading the appropriate line from the appropriate file."""
        if self._rows_cache is not None:
            row = self._rows_cache[idx]
        else:
            file_path, offset, fieldnames = self._index[idx]

            with open(file_path, "r") as handle:
                handle.seek(offset)
                line = handle.readline().strip()

            values = line.split()
            row = dict(zip(fieldnames, values))
        smiles = row[self.smiles_col]
        data = smiles_to_pygdata(
            smiles,
            explicit_hydrogens=self.explicit_hydrogens,
            encode_hydrogen_count=self.encode_hydrogen_count,
        )
        if data is None:
            return None

        # Add global index for tracking
        data.graph_idx = torch.tensor([idx])

        return data


class PrecomputedGraphDataset(Dataset):
    """Dataset that reads precomputed PyG graphs from sharded ``.pt`` files.

    Each shard is expected to contain a list[Data]. A ``metadata.json`` file is optional;
    if present, it can provide shard sizes for faster startup.
    """

    def __init__(
        self,
        data_dir: str,
        pattern: str = "shard_*.pt",
        cache_in_memory: bool = False,
        max_cached_shards: int = 1,
        debug: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.pattern = pattern
        self.cache_in_memory = cache_in_memory
        self.max_cached_shards = max(1, int(max_cached_shards))
        self.debug = debug
        self._shard_access_count = 0
        self._shard_cache_hits = 0
        self._shard_cache_misses = 0

        init_start = time.time()
        self._debug(
            f"Init start | data_dir={data_dir} pattern={pattern} "
            f"cache_in_memory={cache_in_memory} max_cached_shards={self.max_cached_shards}"
        )

        self.shard_paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not self.shard_paths:
            raise ValueError(f"No precomputed shards found matching {pattern} in {data_dir}")
        self._debug(f"Discovered {len(self.shard_paths)} shard files")

        self.shard_sizes = self._load_or_infer_shard_sizes()
        self.cumulative_sizes: List[int] = []
        running_total = 0
        for size in self.shard_sizes:
            running_total += size
            self.cumulative_sizes.append(running_total)

        self.total_graphs = running_total
        if self.total_graphs == 0:
            raise ValueError(f"Precomputed dataset is empty in {data_dir}")

        # Per-worker LRU shard cache for on-demand loading.
        self._shard_cache: "OrderedDict[int, List[Data]]" = OrderedDict()
        self._all_graphs_cache: Optional[List[Data]] = None

        if self.cache_in_memory:
            self._debug("Starting full in-memory preload of all shards")
            self._all_graphs_cache = self._load_all_shards_into_memory()
            self._debug("Finished full in-memory preload")

        print(
            f"Initialized PrecomputedGraphDataset with {len(self.shard_paths)} shards, "
            f"{self.total_graphs} total graphs"
        )
        self._debug(f"Init complete in {time.time() - init_start:.2f}s")

    def _debug(self, message: str) -> None:
        if self.debug:
            now = time.strftime("%H:%M:%S")
            print(f"[PrecomputedGraphDataset {now}] {message}")

    def _load_all_shards_into_memory(self) -> List[Data]:
        """Load every graph once to avoid repeated shard file reads."""
        all_graphs: List[Data] = []
        preload_start = time.time()
        for shard_idx, shard_path in enumerate(self.shard_paths):
            shard_start = time.time()
            shard = _load_torch_data(shard_path)
            all_graphs.extend(shard)
            self._debug(
                f"Preloaded shard {shard_idx + 1}/{len(self.shard_paths)} "
                f"({len(shard)} graphs) in {time.time() - shard_start:.2f}s"
            )
        print(f"Initialized PrecomputedGraphDataset RAM cache with {len(all_graphs)} graphs")
        self._debug(f"Full preload total time: {time.time() - preload_start:.2f}s")
        return all_graphs

    def _load_or_infer_shard_sizes(self) -> List[int]:
        size_start = time.time()
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            shard_sizes = metadata.get("shard_sizes")
            if isinstance(shard_sizes, list) and len(shard_sizes) == len(self.shard_paths):
                self._debug(
                    f"Loaded shard sizes from metadata.json in {time.time() - size_start:.2f}s"
                )
                return [int(size) for size in shard_sizes]

        shard_sizes: List[int] = []
        self._debug("Inferring shard sizes by reading all shard files")
        for shard_path in self.shard_paths:
            shard = _load_torch_data(shard_path)
            shard_sizes.append(len(shard))
        self._debug(f"Inferred shard sizes in {time.time() - size_start:.2f}s")
        return shard_sizes

    def __len__(self) -> int:
        return self.total_graphs

    def _load_shard(self, shard_idx: int) -> List[Data]:
        self._shard_access_count += 1
        if shard_idx in self._shard_cache:
            self._shard_cache_hits += 1
            shard = self._shard_cache.pop(shard_idx)
            self._shard_cache[shard_idx] = shard
            if self.debug and self._shard_access_count <= 20:
                self._debug(
                    f"Cache HIT shard={shard_idx} | hits={self._shard_cache_hits} "
                    f"misses={self._shard_cache_misses}"
                )
            return shard

        self._shard_cache_misses += 1
        load_start = time.time()
        shard = _load_torch_data(self.shard_paths[shard_idx])
        self._shard_cache[shard_idx] = shard

        evicted = None
        while len(self._shard_cache) > self.max_cached_shards:
            evicted, _ = self._shard_cache.popitem(last=False)

        if self.debug and (self._shard_access_count <= 20 or self._shard_access_count % 100 == 0):
            self._debug(
                f"Cache MISS shard={shard_idx} load_time={time.time() - load_start:.2f}s "
                f"cache_size={len(self._shard_cache)} evicted={evicted} "
                f"hits={self._shard_cache_hits} misses={self._shard_cache_misses}"
            )

        return shard

    def __getitem__(self, idx: int) -> Data:
        if idx < 0 or idx >= self.total_graphs:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_graphs}")

        if self._all_graphs_cache is not None:
            data = self._all_graphs_cache[idx].clone()
            data.graph_idx = torch.tensor([idx], dtype=torch.long)
            return data

        shard_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        prev_cum = self.cumulative_sizes[shard_idx - 1] if shard_idx > 0 else 0
        local_idx = idx - prev_cum

        shard = self._load_shard(shard_idx)
        data = shard[local_idx].clone()
        data.graph_idx = torch.tensor([idx], dtype=torch.long)
        return data
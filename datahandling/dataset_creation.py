"""Create a PyG Dataset from a CSV of SMILES strings, with lazy loading and graph augmentations."""
from torch_geometric.data import Data
from torch.utils.data import Dataset
import csv
import torch
from .graph_creation import smiles_to_pygdata
from typing import List, Sequence, Union, Tuple, Optional
import os
import glob

class SmilesCsvDataset(Dataset):
    """Lazy dataset: keep SMILES on disk, build graphs on demand."""

    def __init__(self, csv_path: str, smiles_col: str = "smiles",
                 target: Union[str, Sequence[str], None] = None,
                 task: str = "regression") -> None:
        self.csv_path = csv_path
        self.smiles_col = smiles_col
        self.target = target
        self.task = task
        self._index = self._build_index()
        self._fieldnames = self._get_fieldnames()

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
        with open(self.csv_path, "r", newline="") as handle:
            handle.seek(self._index[idx])
            reader = csv.DictReader(handle, fieldnames=self._fieldnames)
            row = next(reader)
        data = smiles_to_pygdata(row[self.smiles_col])
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
                 pattern: str = "*.smi") -> None:
        """
        Args:
            data_dir: Directory containing .smi files
            smiles_col: Name of the column containing SMILES strings
            pattern: Glob pattern to match files (default: *.smi)
        """
        self.data_dir = data_dir
        self.smiles_col = smiles_col
        self.pattern = pattern
        
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
                        self._index.append((file_path, offset, fieldnames))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Optional[Data]:
        """Load a single graph by reading the appropriate line from the appropriate file."""
        file_path, offset, fieldnames = self._index[idx]

        with open(file_path, "r") as handle:
            handle.seek(offset)
            line = handle.readline().strip()

        values = line.split()
        row = dict(zip(fieldnames, values))
        smiles = row[self.smiles_col]
        data = smiles_to_pygdata(smiles)
        if data is None:
            return None

        # Add global index for tracking
        data.graph_idx = torch.tensor([idx])

        return data
"""Precompute molecular graphs from SMILES into sharded PyG Data files.

Usage examples:
  python precompute_graphs.py --input data/zinc/zinc_data --output data/zinc/precomputed_graphs
  python precompute_graphs.py --input data/delaney-processed.csv --output data/delaney/precomputed_graphs --smiles-col smiles
"""

import argparse
import csv
import glob
import json
import os
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from typing import Dict, Iterator, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from datahandling.graph_creation import smiles_to_pygdata


def _default_num_workers() -> int:
    # Respect the actual SLURM CPU allocation when running under sbatch/srun
    # rather than os.cpu_count(), which reports the node's full core count
    # even if the job only reserved a slice of it.
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    return max(1, os.cpu_count() or 1)


def _featurize_row(
    row: Dict[str, str],
    smiles_col: str,
    explicit_hydrogens: bool,
    encode_hydrogen_count: bool,
    use_extended_features: bool,
    scale_eccentricity: bool,
):
    """Top-level (picklable) worker for multiprocessing.Pool -- per-molecule
    featurization is fully independent, so this is embarrassingly parallel.

    Returns plain numpy arrays, not a PyG Data object with live torch
    tensors. torch installs custom multiprocessing reducers that share
    tensors via file-descriptor-backed shared memory instead of copying them
    through the pipe -- fine for a few large tensors, but with many workers
    each returning several small tensors per molecule across millions of
    molecules, it exhausts the OS's shared-memory/fd limit
    ("unable to mmap ... Cannot allocate memory"). Plain numpy arrays use
    ordinary pickling and sidestep that machinery entirely; the main process
    reconstructs the Data object (see _numpy_result_to_data).
    """
    smiles = row.get(smiles_col, "")
    if not smiles:
        return None
    data = smiles_to_pygdata(
        smiles,
        explicit_hydrogens=explicit_hydrogens,
        encode_hydrogen_count=encode_hydrogen_count,
        use_extended_features=use_extended_features,
        scale_eccentricity=scale_eccentricity,
    )
    if data is None:
        return None
    return (
        data.x.numpy(),
        data.edge_index.numpy(),
        data.edge_attr.numpy(),
        smiles,
    )


def _numpy_result_to_data(result) -> Data:
    x_np, edge_index_np, edge_attr_np, smiles = result
    return Data(
        x=torch.from_numpy(x_np),
        edge_index=torch.from_numpy(edge_index_np),
        edge_attr=torch.from_numpy(edge_attr_np),
        smiles=smiles,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute molecular graphs from SMILES and save sharded .pt files."
    )
    parser.add_argument("--input", required=True, help="Input CSV file or directory of .smi files")
    parser.add_argument("--output", required=True, help="Output directory for shard_*.pt files")
    parser.add_argument("--smiles-col", default="smiles", help="Column name containing SMILES")
    parser.add_argument("--pattern", default="*.smi", help="Glob for .smi files when input is a directory")
    parser.add_argument("--shard-size", type=int, default=50000, help="Graphs per shard file")
    parser.add_argument(
        "--explicit-hydrogens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include hydrogen atoms as explicit graph nodes (default: true)",
    )
    parser.add_argument(
        "--encode-hydrogen-count",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append total hydrogen count as an extra atom feature",
    )
    parser.add_argument(
        "--use-extended-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append Gasteiger partial charge and topological eccentricity (2 extra "
        "node feature dims). Produces 26-dim (or 27 with --encode-hydrogen-count) "
        "node features -- set num_features accordingly in any training config that "
        "uses these shards, and note it makes them incompatible with 24-feature checkpoints.",
    )
    parser.add_argument(
        "--scale-eccentricity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the fitted StandardScaler (fit_eccentricity_scaler.py) to the "
        "eccentricity feature. Only meaningful with --use-extended-features. "
        "Independent flag from --use-extended-features on purpose: existing "
        "shards/checkpoints built before this scaler existed expect raw "
        "eccentricity, so this must default to False.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard_*.pt files and metadata.json in output directory",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel worker processes for featurization (default: SLURM_CPUS_PER_TASK "
        "if running under sbatch/srun, else all detected CPU cores). Per-molecule "
        "featurization is fully independent, so this scales close to linearly.",
    )
    return parser.parse_args()


def iter_rows_from_csv(csv_path: str) -> Iterator[Dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def iter_rows_from_smi_dir(data_dir: str, pattern: str) -> Iterator[Dict[str, str]]:
    file_paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not file_paths:
        raise ValueError(f"No files found matching pattern {pattern} in {data_dir}")

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as handle:
            header = handle.readline().strip()
            if not header:
                continue
            fieldnames = header.split()
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                values = line.split()
                row = dict(zip(fieldnames, values))
                yield row


def write_shard(output_dir: str, shard_idx: int, shard_data: list) -> str:
    shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.pt")
    torch.save(shard_data, shard_path)
    return shard_path


def ensure_output_dir(output_dir: str, overwrite: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)
    existing_shards = sorted(glob.glob(os.path.join(output_dir, "shard_*.pt")))
    metadata_path = os.path.join(output_dir, "metadata.json")

    if existing_shards or os.path.isfile(metadata_path):
        if not overwrite:
            raise ValueError(
                f"Output directory {output_dir} already contains precomputed data. "
                "Use --overwrite to replace it."
            )
        for shard_path in existing_shards:
            os.remove(shard_path)
        if os.path.isfile(metadata_path):
            os.remove(metadata_path)


def main() -> None:
    args = parse_args()

    if args.shard_size <= 0:
        raise ValueError("--shard-size must be > 0")

    ensure_output_dir(args.output, args.overwrite)

    if os.path.isdir(args.input):
        row_iter = iter_rows_from_smi_dir(args.input, args.pattern)
        source_type = "smi_dir"
    elif os.path.isfile(args.input):
        row_iter = iter_rows_from_csv(args.input)
        source_type = "csv"
    else:
        raise ValueError(f"Input path must be a file or directory, got: {args.input}")

    shard_data = []
    shard_sizes = []
    shard_idx = 0

    total_rows = 0
    total_valid = 0
    total_invalid = 0

    num_workers = args.num_workers if args.num_workers is not None else _default_num_workers()
    print(f"Precomputing graphs with {num_workers} worker process(es)...")

    worker_fn = partial(
        _featurize_row,
        smiles_col=args.smiles_col,
        explicit_hydrogens=bool(args.explicit_hydrogens),
        encode_hydrogen_count=bool(args.encode_hydrogen_count),
        use_extended_features=bool(args.use_extended_features),
        scale_eccentricity=bool(args.scale_eccentricity),
    )

    # imap (not imap_unordered) keeps graph_idx assignment deterministic
    # across runs with the same input/worker count; chunksize amortizes
    # inter-process communication overhead across many small, fast tasks.
    with Pool(processes=num_workers) as pool:
        for result in pool.imap(worker_fn, row_iter, chunksize=256):
            total_rows += 1
            if result is None:
                total_invalid += 1
            else:
                data = _numpy_result_to_data(result)
                data.graph_idx = torch.tensor([total_valid], dtype=torch.long)
                shard_data.append(data)
                total_valid += 1

            if len(shard_data) >= args.shard_size:
                shard_path = write_shard(args.output, shard_idx, shard_data)
                shard_sizes.append(len(shard_data))
                print(f"  Wrote {shard_path} ({len(shard_data)} graphs)")
                shard_data = []
                shard_idx += 1

            if total_rows % 100000 == 0:
                print(f"  Processed {total_rows} rows | valid={total_valid} invalid={total_invalid}")

    if shard_data:
        shard_path = write_shard(args.output, shard_idx, shard_data)
        shard_sizes.append(len(shard_data))
        print(f"  Wrote {shard_path} ({len(shard_data)} graphs)")

    metadata = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_path": args.input,
        "source_type": source_type,
        "smiles_col": args.smiles_col,
        "pattern": args.pattern if source_type == "smi_dir" else None,
        "shard_size": args.shard_size,
        "explicit_hydrogens": bool(args.explicit_hydrogens),
        "encode_hydrogen_count": bool(args.encode_hydrogen_count),
        "use_extended_features": bool(args.use_extended_features),
        "scale_eccentricity": bool(args.scale_eccentricity),
        "num_shards": len(shard_sizes),
        "shard_sizes": shard_sizes,
        "total_rows": total_rows,
        "total_valid_graphs": total_valid,
        "total_invalid_rows": total_invalid,
    }

    metadata_path = os.path.join(args.output, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nDone.")
    print(f"  Total rows: {total_rows}")
    print(f"  Valid graphs: {total_valid}")
    print(f"  Invalid rows: {total_invalid}")
    print(f"  Shards written: {len(shard_sizes)}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

"""Plot overlapping atom-count distributions for multiple molecular datasets."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem


def gaussian_smooth(y: np.ndarray, sigma: float = 1.6) -> np.ndarray:
    """Apply 1D Gaussian smoothing using a normalized convolution kernel."""
    if sigma <= 0:
        return y
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def iter_smiles_from_smi_dir(shard_dir: Path):
    """Yield SMILES strings from all .smi files in a directory."""
    for smi_file in sorted(shard_dir.glob("*.smi")):
        with smi_file.open("r", encoding="utf-8") as f:
            _ = next(f, None)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield line.split()[0]


def iter_smiles_from_csv(csv_path: Path, smiles_col: str):
    """Yield SMILES strings from a CSV or CSV.GZ file."""
    opener = gzip.open if csv_path.suffix == ".gz" else open
    with opener(csv_path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or smiles_col not in reader.fieldnames:
            raise ValueError(
                f"SMILES column '{smiles_col}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            smiles = (row.get(smiles_col) or "").strip()
            if smiles:
                yield smiles


def count_atoms(smiles: str, include_hydrogens: bool = True) -> int | None:
    """Return atom count for a SMILES string, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if include_hydrogens:
        mol = Chem.AddHs(mol)
    return int(mol.GetNumAtoms())


def load_counts(input_path: Path, smiles_col: str, include_hydrogens: bool = True):
    """Load atom counts and basic stats from an input source."""
    if input_path.is_dir():
        smiles_iter = iter_smiles_from_smi_dir(input_path)
    else:
        suffixes = "".join(input_path.suffixes[-2:])
        if input_path.suffix == ".csv" or suffixes == ".csv.gz":
            smiles_iter = iter_smiles_from_csv(input_path, smiles_col)
        else:
            raise ValueError(
                f"Unsupported input file type: {input_path}. Expected directory, .csv, or .csv.gz"
            )

    counts = []
    total = 0
    invalid = 0
    for smiles in smiles_iter:
        total += 1
        atom_count = count_atoms(smiles, include_hydrogens=include_hydrogens)
        if atom_count is None:
            invalid += 1
            continue
        counts.append(atom_count)

    return counts, total, invalid


def main():
    datasets = [
        {
            "label": "ZINC half",
            "path": Path("data/zinc/zinc_half_data"),
            "smiles_col": "smiles",
            "color": "#1f77b4",
        },
        {
            "label": "BACE",
            "path": Path("data/MoleculeNet_BACE_custom/bace.csv"),
            "smiles_col": "mol",
            "color": "#d62728",
        },
        {
            "label": "LIPO",
            "path": Path("data/MoleculeNet_LIPO_custom/Lipophilicity.csv"),
            "smiles_col": "smiles",
            "color": "#2ca02c",
        },
        {
            "label": "Tox21",
            "path": Path("data/MoleculeNet_Tox21_custom/tox21.csv.gz"),
            "smiles_col": "smiles",
            "color": "#ff7f0e",
        },
    ]

    output_path = Path("data/atom_count_overlay_with_h.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    global_min = None
    global_max = None

    for dataset in datasets:
        counts, total, invalid = load_counts(
            dataset["path"],
            smiles_col=dataset["smiles_col"],
            include_hydrogens=True,
        )
        if not counts:
            raise ValueError(f"No valid molecules for dataset: {dataset['label']}")

        local_min = min(counts)
        local_max = max(counts)
        global_min = local_min if global_min is None else min(global_min, local_min)
        global_max = local_max if global_max is None else max(global_max, local_max)

        results.append(
            {
                "label": dataset["label"],
                "counts": counts,
                "total": total,
                "invalid": invalid,
                "color": dataset["color"],
            }
        )

    bins = np.arange(int(global_min), int(global_max) + 2)

    plt.figure(figsize=(12, 7))
    for result in results:
        density, edges = np.histogram(result["counts"], bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        smooth_density = gaussian_smooth(density, sigma=1.6)

        plt.plot(
            centers,
            smooth_density,
            color=result["color"],
            linewidth=2,
            label=f"{result['label']} (n={len(result['counts'])})",
        )
        plt.fill_between(centers, smooth_density, 0, color=result["color"], alpha=0.25)

    plt.title("Atom Count Distribution Overlap (with H)")
    plt.xlabel("Number of atoms per molecule")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved overlay plot: {output_path}")
    for result in results:
        print(
            f"{result['label']}: total={result['total']}, valid={len(result['counts'])}, invalid={result['invalid']}, "
            f"min={min(result['counts'])}, median={sorted(result['counts'])[len(result['counts']) // 2]}, "
            f"mean={sum(result['counts']) / len(result['counts']):.2f}, max={max(result['counts'])}"
        )


if __name__ == "__main__":
    main()

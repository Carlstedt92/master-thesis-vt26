"""Plot atom-count distribution for ZINC SMILES shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from rdkit import Chem


def iter_smiles(shard_dir: Path):
    """Yield SMILES strings from all .smi files in a directory."""
    for smi_file in sorted(shard_dir.glob("*.smi")):
        with smi_file.open("r", encoding="utf-8") as f:
            header = next(f, None)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format is: "<smiles> <zinc_id>"
                smiles = line.split()[0]
                yield smiles


def count_atoms(smiles: str, include_hydrogens: bool = False) -> int | None:
    """Return atom count for a SMILES string, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if include_hydrogens:
        mol = Chem.AddHs(mol)
    return int(mol.GetNumAtoms())


def summarize(values: list[int]) -> dict[str, float]:
    """Compute simple summary statistics."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        median = float(sorted_vals[mid])
    return {
        "count": float(n),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "mean": float(sum(sorted_vals) / n),
        "median": float(median),
    }


def plot_distribution(atom_counts: list[int], output_path: Path, title: str):
    """Create and save atom-count histogram."""
    if not atom_counts:
        raise ValueError("No valid molecules found to plot.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    bins = range(min(atom_counts), max(atom_counts) + 2)
    plt.figure(figsize=(10, 6))
    plt.hist(atom_counts, bins=bins, edgecolor="black", alpha=0.8)
    plt.title(title)
    plt.xlabel("Number of atoms per molecule")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot atom-count distribution for ZINC dataset.")
    parser.add_argument(
        "--zinc-dir",
        type=Path,
        default=Path("data/zinc/zinc_data"),
        help="Directory containing ZINC .smi shard files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/zinc/zinc_atom_count_distribution.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--include-hydrogens",
        action="store_true",
        help="If set, include explicit hydrogens in atom counts.",
    )
    args = parser.parse_args()

    if not args.zinc_dir.exists():
        raise FileNotFoundError(f"ZINC directory not found: {args.zinc_dir}")

    atom_counts: list[int] = []
    invalid = 0
    total = 0

    for smiles in iter_smiles(args.zinc_dir):
        total += 1
        count = count_atoms(smiles, include_hydrogens=args.include_hydrogens)
        if count is None:
            invalid += 1
            continue
        atom_counts.append(count)

    stats = summarize(atom_counts)
    title_suffix = "(with H)" if args.include_hydrogens else "(heavy atoms)"
    plot_distribution(atom_counts, args.output, f"ZINC Atom Count Distribution {title_suffix}")

    print(f"Processed molecules: {total}")
    print(f"Valid molecules: {len(atom_counts)}")
    print(f"Invalid molecules: {invalid}")
    print(
        "Atom counts - "
        f"min: {stats.get('min', 'n/a')}, "
        f"median: {stats.get('median', 'n/a')}, "
        f"mean: {stats.get('mean', 'n/a'):.2f}, "
        f"max: {stats.get('max', 'n/a')}"
    )
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()

"""
Measure how many ZINC training molecules contain a ketone and/or aldehyde
group, using the same SMARTS patterns as GraphAugmentation.FUNCTIONAL_GROUP_PATTERNS.
Samples broadly across shards (same stride approach as
measure_khop_fallback_rate.py, since single-shard sampling isn't
representative -- there's real shard-to-shard heterogeneity in this dataset).
"""

import glob
import torch
from rdkit import Chem
from datahandling.graph_augmentation import GraphAugmentation

SHARD_GLOB = "/proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended/*.pt"
MOLECULES_PER_SHARD = 3000
SHARD_STRIDE = 6

aldehyde_pattern = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
ketone_pattern = Chem.MolFromSmarts("[#6][CX3](=O)[#6]")

all_shard_paths = sorted(glob.glob(SHARD_GLOB))
sampled_paths = all_shard_paths[::SHARD_STRIDE]
print(f"Sampling {len(sampled_paths)} of {len(all_shard_paths)} shards, "
      f"{MOLECULES_PER_SHARD} molecules each (stride={SHARD_STRIDE}).")

total_molecules = 0
molecules_with_aldehyde = 0
molecules_with_ketone = 0
molecules_with_either = 0
total_aldehyde_matches = 0
total_ketone_matches = 0

for i, path in enumerate(sampled_paths):
    shard = torch.load(path, weights_only=False)
    sample = shard[:MOLECULES_PER_SHARD]
    for data in sample:
        smiles = getattr(data, "smiles", None)
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        total_molecules += 1
        ald_matches = mol.GetSubstructMatches(aldehyde_pattern)
        ket_matches = mol.GetSubstructMatches(ketone_pattern)
        has_ald = len(ald_matches) > 0
        has_ket = len(ket_matches) > 0
        if has_ald:
            molecules_with_aldehyde += 1
            total_aldehyde_matches += len(ald_matches)
        if has_ket:
            molecules_with_ketone += 1
            total_ketone_matches += len(ket_matches)
        if has_ald or has_ket:
            molecules_with_either += 1
    del shard
    print(f"  [{i + 1}/{len(sampled_paths)}] {path.split('/')[-1]}: running total {total_molecules} molecules, "
          f"aldehyde={molecules_with_aldehyde} ({100*molecules_with_aldehyde/total_molecules:.2f}%), "
          f"ketone={molecules_with_ketone} ({100*molecules_with_ketone/total_molecules:.2f}%)")

print(f"\n=== Overall (n={total_molecules}) ===")
print(f"Molecules with >=1 aldehyde: {molecules_with_aldehyde} ({100*molecules_with_aldehyde/total_molecules:.2f}%) "
      f"| total aldehyde matches (incl. multiples per molecule): {total_aldehyde_matches}")
print(f"Molecules with >=1 ketone:   {molecules_with_ketone} ({100*molecules_with_ketone/total_molecules:.2f}%) "
      f"| total ketone matches (incl. multiples per molecule): {total_ketone_matches}")
print(f"Molecules with either:       {molecules_with_either} ({100*molecules_with_either/total_molecules:.2f}%)")

"""
Supervisor feedback: the Aromatic_Ring SMARTS pattern (c1ccccc1) only covers
benzene, missing heteroaryl rings (pyridine, furan, thiophene, etc.) and
5-membered aromatic rings, which are common in drug-like molecules. Aliphatic
rings aren't covered either but are said to be rare.

This directly tests that: of the ~1% of ZINC molecules that hit the
functional_group_k_hop fallback (no SMARTS pattern matched at all), how many
are failing specifically because of a ring-coverage gap (they DO contain an
aromatic ring, just not a plain benzo one, or a non-aromatic ring) versus
being genuinely simple/acyclic molecules with nothing to detect.
"""

import glob
import torch
from rdkit import Chem
from datahandling.graph_augmentation import GraphAugmentation

SHARD_GLOB = "/proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended/*.pt"
MOLECULES_PER_SHARD = 3000
SHARD_STRIDE = 6

all_shard_paths = sorted(glob.glob(SHARD_GLOB))
sampled_paths = all_shard_paths[::SHARD_STRIDE]
print(f"Sampling {len(sampled_paths)} of {len(all_shard_paths)} shards, "
      f"{MOLECULES_PER_SHARD} molecules each (stride={SHARD_STRIDE}).")

augmenter = GraphAugmentation(local_views=2, k_hops=2, local_augmentation_mode="functional_group_k_hop")

total_molecules = 0
fallback_molecules = []  # store (smiles,) for classification

for i, path in enumerate(sampled_paths):
    shard = torch.load(path, weights_only=False)
    sample = shard[:MOLECULES_PER_SHARD]
    for data in sample:
        smiles = getattr(data, "smiles", None)
        if not smiles:
            continue
        total_molecules += 1
        groups = augmenter._functional_group_atom_sets(data)
        if not groups:
            fallback_molecules.append(smiles)
    del shard
    print(f"  [{i + 1}/{len(sampled_paths)}] processed, {total_molecules} molecules so far, "
          f"{len(fallback_molecules)} fallbacks so far")

print(f"\nTotal: {total_molecules} molecules, {len(fallback_molecules)} fallbacks "
      f"({100 * len(fallback_molecules) / total_molecules:.2f}%)")

# Classify each fallback molecule's ring content
has_aromatic_heteroatom_ring = 0  # aromatic ring containing >=1 non-carbon atom (heteroaryl)
has_aromatic_5ring = 0             # any 5-membered all-aromatic ring (carbocyclic or hetero)
has_any_aromatic_ring = 0          # any aromatic ring at all (should be 0 given Aromatic_Ring covers benzo, but 5-ring/hetero aromatic wouldn't be caught)
has_aliphatic_ring = 0             # any non-aromatic ring
has_any_ring = 0
acyclic_no_ring = 0

for smiles in fallback_molecules:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    if not atom_rings:
        acyclic_no_ring += 1
        continue
    has_any_ring += 1

    mol_has_aromatic = False
    mol_has_hetero_aromatic = False
    mol_has_aromatic_5ring = False
    mol_has_aliphatic_ring = False

    for ring in atom_rings:
        atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        ring_is_aromatic = all(a.GetIsAromatic() for a in atoms)
        if ring_is_aromatic:
            mol_has_aromatic = True
            if any(a.GetAtomicNum() != 6 for a in atoms):
                mol_has_hetero_aromatic = True
            if len(ring) == 5:
                mol_has_aromatic_5ring = True
        else:
            mol_has_aliphatic_ring = True

    if mol_has_aromatic:
        has_any_aromatic_ring += 1
    if mol_has_hetero_aromatic:
        has_aromatic_heteroatom_ring += 1
    if mol_has_aromatic_5ring:
        has_aromatic_5ring += 1
    if mol_has_aliphatic_ring:
        has_aliphatic_ring += 1

n = len(fallback_molecules)
print(f"\n=== Ring content of the {n} fallback molecules ===")
print(f"Acyclic (no ring at all): {acyclic_no_ring} ({100*acyclic_no_ring/n:.1f}%)")
print(f"Has any ring: {has_any_ring} ({100*has_any_ring/n:.1f}%)")
print(f"  - has aromatic ring (any kind, missed by benzo-only pattern): {has_any_aromatic_ring} ({100*has_any_aromatic_ring/n:.1f}%)")
print(f"    - of which heteroaromatic (contains non-C atom): {has_aromatic_heteroatom_ring} ({100*has_aromatic_heteroatom_ring/n:.1f}%)")
print(f"    - of which aromatic 5-membered ring: {has_aromatic_5ring} ({100*has_aromatic_5ring/n:.1f}%)")
print(f"  - has non-aromatic (aliphatic) ring: {has_aliphatic_ring} ({100*has_aliphatic_ring/n:.1f}%)")

print("\nExample fallback SMILES (first 15):")
for s in fallback_molecules[:15]:
    print(f"  {s}")

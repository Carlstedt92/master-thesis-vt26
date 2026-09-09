"""
Measure how often functional_group_k_hop's local-view construction falls
back to plain (random-atom) k_hop because no functional-group SMARTS pattern
matched the molecule -- across the whole 9.4M-molecule ZINC dataset, not
just one shard (a single-shard spot check showed real shard-to-shard
variance: 3.10% on shard_00000 vs 1.38% on shard_00050).
"""

import glob
import torch
from datahandling.graph_augmentation import GraphAugmentation

SHARD_GLOB = "/proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended/*.pt"
MOLECULES_PER_SHARD = 3000
SHARD_STRIDE = 6  # every 6th shard -> ~31 of 189 shards, spread across the full range

all_shard_paths = sorted(glob.glob(SHARD_GLOB))
sampled_paths = all_shard_paths[::SHARD_STRIDE]
print(f"Sampling {len(sampled_paths)} of {len(all_shard_paths)} shards, "
      f"{MOLECULES_PER_SHARD} molecules each (stride={SHARD_STRIDE}).")

augmenter = GraphAugmentation(local_views=2, k_hops=2, local_augmentation_mode="functional_group_k_hop")

per_shard_rates = []
total_no_group = 0
total_molecules = 0
for i, path in enumerate(sampled_paths):
    shard = torch.load(path, weights_only=False)
    sample = shard[:MOLECULES_PER_SHARD]
    no_group = sum(1 for d in sample if not augmenter._functional_group_atom_sets(d))
    rate = 100 * no_group / len(sample)
    per_shard_rates.append(rate)
    total_no_group += no_group
    total_molecules += len(sample)
    del shard
    print(f"  [{i + 1}/{len(sampled_paths)}] {path.split('/')[-1]}: {no_group}/{len(sample)} = {rate:.2f}%  "
          f"(running total: {100 * total_no_group / total_molecules:.2f}% over {total_molecules})")

import statistics
print(f"\n=== Overall ===")
print(f"Total sampled: {total_molecules} molecules across {len(sampled_paths)} shards")
print(f"Overall fallback rate: {100 * total_no_group / total_molecules:.2f}%")
print(f"Per-shard rate: mean={statistics.mean(per_shard_rates):.2f}% "
      f"std={statistics.stdev(per_shard_rates):.2f}% "
      f"min={min(per_shard_rates):.2f}% max={max(per_shard_rates):.2f}%")

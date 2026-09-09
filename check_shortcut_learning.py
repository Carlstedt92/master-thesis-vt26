"""
Verify the "shortcut learning" hypothesis for the extended-features (Gasteiger
charge + topological eccentricity) downstream-performance regression.

DINO's SSL loss is an instance-discrimination task: for every view (local or
global) produced from molecule G, the student's prediction must match the
teacher's prediction for a GLOBAL view of that SAME molecule G, and not match
any other molecule's view, within a batch. If two cheap scalar node features
survive local/masked/subgraph views well enough that their simple aggregate
statistics alone let you pick out the correct parent molecule among a batch
of candidates, the network could learn to solve the SSL task almost entirely
from those 2 numbers instead of from graph structure -- a shortcut that would
lower SSL loss while representations of the OTHER 24 features stay undertrained,
explaining the observed downstream regression.

This script builds real local/global views with the SAME augmentation code
path used in training (datahandling/graph_augmentation.py), computes simple
per-view aggregate statistics of ONLY the 2 new feature columns (mean + std
of Gasteiger charge, mean + std of eccentricity -- 4 numbers total, no graph
structure, none of the other 24 features), and measures how well a 1-NN
matcher using just those 4 numbers can identify which molecule's global view
a given local view came from, within a batch. Compared against chance and
against the same test using the pre-existing (base) feature set as a control.
"""

import random
import torch
import numpy as np

from datahandling.graph_augmentation import GraphAugmentation

SHARD_PATH = "/proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended/shard_00000.pt"
NUM_MOLECULES = 500
CHARGE_COL = 24
ECC_COL = 25
SEED = 0


def aggregate_new_features(x):
    charge = x[:, CHARGE_COL]
    ecc = x[:, ECC_COL]
    # Sum is included because global_pooling is "add" in these configs -- for
    # masking mode, masked-out nodes stay in the graph with zeroed features
    # (not removed), so a sum over all nodes is exactly what the network's
    # own graph-level readout would compute from raw input features, and also
    # implicitly encodes remaining-node count / molecule size.
    return np.array([
        charge.sum().item(), charge.mean().item(), charge.std().item() if charge.numel() > 1 else 0.0,
        ecc.sum().item(), ecc.mean().item(), ecc.std().item() if ecc.numel() > 1 else 0.0,
    ])


def match_accuracy(global_stats, local_stats_by_molecule, label):
    """For each local view, 1-NN match (Euclidean, over the 4 stats) against all
    molecules' global-view stats in the batch. Report top-1 / top-5 accuracy."""
    global_mat = np.stack(global_stats)  # [N_mol, 4]
    mu = global_mat.mean(axis=0)
    sigma = global_mat.std(axis=0) + 1e-8
    global_norm = (global_mat - mu) / sigma

    n_correct_top1 = 0
    n_correct_top5 = 0
    n_total = 0
    for mol_idx, local_list in local_stats_by_molecule.items():
        for local_stat in local_list:
            local_norm = (local_stat - mu) / sigma
            dists = np.linalg.norm(global_norm - local_norm[None, :], axis=1)
            order = np.argsort(dists)
            n_total += 1
            if order[0] == mol_idx:
                n_correct_top1 += 1
            if mol_idx in order[:5]:
                n_correct_top5 += 1

    top1 = n_correct_top1 / n_total
    top5 = n_correct_top5 / n_total
    chance_top1 = 1.0 / len(global_stats)
    chance_top5 = min(5, len(global_stats)) / len(global_stats)
    print(f"[{label}] N_molecules={len(global_stats)}  N_local_views={n_total}")
    print(f"  top-1 accuracy: {top1:.4f}  (chance: {chance_top1:.4f})")
    print(f"  top-5 accuracy: {top5:.4f}  (chance: {chance_top5:.4f})")
    return top1, top5


def node_count_only(x):
    # Isolates whether "how many (nonzero) nodes remain" alone -- pure view
    # size, no chemistry at all -- explains the matching accuracy.
    nonzero_mask = x.abs().sum(dim=1) > 0
    return np.array([nonzero_mask.sum().item()])


def run(mode, node_mask_ratio=0.05, k_hops=2):
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"\nLoading {NUM_MOLECULES} molecules from {SHARD_PATH} ...")
    shard = torch.load(SHARD_PATH, weights_only=False)
    molecules = shard[:NUM_MOLECULES]

    augmenter = GraphAugmentation(
        local_views=2,
        k_hops=k_hops,
        local_augmentation_mode=mode,
        node_mask_ratio=node_mask_ratio,
        feature_mask_ratio=0.0,
    )

    # Build a global pool of real (charge, ecc) node values to draw shuffled
    # replacements from, for the "destroy the chemistry, keep the structure"
    # control below.
    charge_pool = torch.cat([d.x[:, CHARGE_COL] for d in molecules])
    ecc_pool = torch.cat([d.x[:, ECC_COL] for d in molecules])
    pool_size = charge_pool.numel()

    def shuffled_aggregate(x):
        nonzero_mask = x.abs().sum(dim=1) > 0
        n = int(nonzero_mask.sum().item())
        if n == 0:
            return np.zeros(6)
        idx = torch.randint(0, pool_size, (n,))
        charge = charge_pool[idx]
        ecc = ecc_pool[idx]
        return np.array([
            charge.sum().item(), charge.mean().item(), charge.std().item() if n > 1 else 0.0,
            ecc.sum().item(), ecc.mean().item(), ecc.std().item() if n > 1 else 0.0,
        ])

    real_global, real_local = [], {}
    shuf_global, shuf_local = [], {}
    count_global, count_local = [], {}
    for mol_idx, data in enumerate(molecules):
        views = augmenter(data)
        global_views = [v for v in views if v["view"].item() == 1]
        local_views = [v for v in views if v["view"].item() == 0]
        g = global_views[0].x

        real_global.append(aggregate_new_features(g))
        real_local[mol_idx] = [aggregate_new_features(v.x) for v in local_views]

        shuf_global.append(shuffled_aggregate(g))
        shuf_local[mol_idx] = [shuffled_aggregate(v.x) for v in local_views]

        count_global.append(node_count_only(g))
        count_local[mol_idx] = [node_count_only(v.x) for v in local_views]

    print(f"\n=== Augmentation mode: {mode} (node_mask_ratio={node_mask_ratio}, k_hops={k_hops}) ===")
    match_accuracy(real_global, real_local, label=f"{mode}: REAL charge+eccentricity (6 stats incl. sum)")
    match_accuracy(shuf_global, shuf_local, label=f"{mode}: SHUFFLED charge+eccentricity control (same view sizes, random values)")
    match_accuracy(count_global, count_local, label=f"{mode}: node-count-only control (no chemistry at all)")


if __name__ == "__main__":
    # KHOP augmentation -- matches extended_features_test.json5 (the run that
    # showed the regression).
    run(mode="functional_group_k_hop", k_hops=2)

    # Masking augmentation -- matches masking60pct_extended_features_test.json5.
    run(mode="functional_group_masking", node_mask_ratio=0.60)

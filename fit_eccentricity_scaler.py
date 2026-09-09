"""
Fit an sklearn StandardScaler on the topological-eccentricity feature (node
feature column 25 in the 26-dim extended feature set), using the actual ZINC
SSL-training distribution -- per supervisor discussion, to test whether
eccentricity's raw scale (unbounded, grows with molecule size, unlike the
other 25 features which are all binary or small bounded integers/floats) is
contributing to the extended-features downstream regression independent of
the shortcut-learning explanation.

Fits on the EXISTING raw precomputed shards (no RDKit re-run needed -- the
raw eccentricity values are already there); saves the fitted scaler so
datahandling/graph_creation.py can load and apply it consistently to both
ZINC precompute and live SMILES featurization (online eval, knn eval,
LIPO/Tox21/BACE).
"""

import glob
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

SHARD_GLOB = "/proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended/*.pt"
ECC_COL = 25
OUTPUT_PATH = "datahandling/eccentricity_scaler.joblib"
# Shards are ~410MB each (104GB total across 189) -- a random subset of 15
# shards (~750k molecules, ~15-20M atoms) is more than enough to estimate a
# stable mean/std without loading the whole dataset into memory.
NUM_SAMPLE_SHARDS = 15
SEED = 0

all_shard_paths = sorted(glob.glob(SHARD_GLOB))
print(f"Found {len(all_shard_paths)} shards total.")

rng = np.random.RandomState(SEED)
shard_paths = list(rng.choice(all_shard_paths, size=min(NUM_SAMPLE_SHARDS, len(all_shard_paths)), replace=False))
print(f"Sampling {len(shard_paths)} shards for scaler fitting.")

all_ecc = []
total_atoms = 0
total_molecules = 0
for i, path in enumerate(shard_paths):
    shard = torch.load(path, weights_only=False)
    for data in shard:
        all_ecc.append(data.x[:, ECC_COL].numpy())
    total_atoms += sum(d.x.size(0) for d in shard)
    total_molecules += len(shard)
    del shard
    print(f"  loaded {i + 1}/{len(shard_paths)} sampled shards, {total_molecules} molecules, {total_atoms} atoms so far")

ecc_values = np.concatenate(all_ecc).reshape(-1, 1)
print(f"\nTotal atoms: {ecc_values.shape[0]}")
print(f"Raw eccentricity stats: min={ecc_values.min():.3f} max={ecc_values.max():.3f} "
      f"mean={ecc_values.mean():.5f} std={ecc_values.std():.5f}")

scaler = StandardScaler()
scaler.fit(ecc_values)
print(f"\nFitted StandardScaler: mean_={scaler.mean_[0]:.5f} scale_(std)={scaler.scale_[0]:.5f}")

joblib.dump(scaler, OUTPUT_PATH)
print(f"Saved scaler to {OUTPUT_PATH}")

# Sanity check the transform
transformed = scaler.transform(ecc_values[:1000])
print(f"\nSanity check on first 1000 values after transform: "
      f"mean={transformed.mean():.5f} std={transformed.std():.5f} "
      f"min={transformed.min():.3f} max={transformed.max():.3f}")

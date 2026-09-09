"""One-off diagnostic: compare embeddings from the KHOP model's epoch-35
(best_model.pth, pre-collapse) and epoch-200 (final_model.pth, fully
collapsed per the BatchNorm-gamma finding) checkpoints, on the same set of
molecules. Tests the "uniform rescaling, not informational loss" hypothesis:
if collapse just shrinks the encoder's output scale uniformly, corresponding
embeddings should be highly cosine-similar (scale-invariant) even though
their norms differ a lot.
"""

import numpy as np
import torch

from evaluation.knn_lipo import (
    build_embedding_features,
    infer_graph_featurization,
    load_lipo_splits_from_deepchem,
)
from model.config import ModelConfig
from model.gnn_model import GNNModel

MODEL_DIR = "models/KHOP_B1024_H1024_200EP_GAT_9M_ZINC"
CHECKPOINTS = {
    "epoch_35 (best_model.pth)": f"{MODEL_DIR}/checkpoints/best_model.pth",
    "epoch_200 (final_model.pth)": f"{MODEL_DIR}/checkpoints/final_model.pth",
}


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint["config"])
    explicit_h, encode_h = infer_graph_featurization(config)
    model = GNNModel.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, explicit_h, encode_h


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Reuse the LIPO test split as a convenient, already-available set of
    # diverse real molecules -- we don't need labels here, just SMILES.
    rows_by_split, _ = load_lipo_splits_from_deepchem("data/MoleculeNet_LIPO_custom", "random", split_seed=0)
    rows = rows_by_split["test"]
    print(f"Using {len(rows)} molecules from the LIPO test split")

    embeddings = {}
    for label, ckpt_path in CHECKPOINTS.items():
        model, explicit_h, encode_h = load_model(ckpt_path, device)
        X, _, invalid = build_embedding_features(rows, model, device, explicit_h, encode_h)
        embeddings[label] = X
        print(f"  {label}: {X.shape[0]} embeddings ({invalid} invalid SMILES skipped), "
              f"mean norm = {np.linalg.norm(X, axis=1).mean():.4f}")

    labels = list(CHECKPOINTS.keys())
    X_early, X_late = embeddings[labels[0]], embeddings[labels[1]]
    assert X_early.shape == X_late.shape, "molecule sets diverged between checkpoints -- unexpected"

    # Cosine similarity per molecule: scale-invariant, so a value near 1.0
    # means "same direction, different magnitude" -- exactly what uniform
    # rescaling predicts.
    dot = np.sum(X_early * X_late, axis=1)
    norm_early = np.linalg.norm(X_early, axis=1)
    norm_late = np.linalg.norm(X_late, axis=1)
    cosine_sim = dot / (norm_early * norm_late + 1e-12)

    # Per-molecule norm ratio: if collapse is a UNIFORM scale factor, this
    # ratio should be roughly constant across molecules (low std relative to
    # mean). If it varies a lot, the "collapse" is not simply a global scalar.
    norm_ratio = norm_late / (norm_early + 1e-12)

    print("\n=== Cosine similarity (epoch_35 embedding vs epoch_200 embedding, per molecule) ===")
    print(f"  mean   = {cosine_sim.mean():.4f}")
    print(f"  median = {np.median(cosine_sim):.4f}")
    print(f"  std    = {cosine_sim.std():.4f}")
    print(f"  min    = {cosine_sim.min():.4f}")
    print(f"  max    = {cosine_sim.max():.4f}")

    print("\n=== Norm ratio (||epoch_200|| / ||epoch_35||, per molecule) ===")
    print(f"  mean   = {norm_ratio.mean():.4f}")
    print(f"  median = {np.median(norm_ratio):.4f}")
    print(f"  std    = {norm_ratio.std():.4f}  (relative std = {norm_ratio.std()/norm_ratio.mean():.4f})")
    print(f"  min    = {norm_ratio.min():.4f}")
    print(f"  max    = {norm_ratio.max():.4f}")

    print("\n=== Interpretation ===")
    print("High mean cosine similarity (>0.9) + low relative std in norm ratio")
    print("would support 'uniform rescaling, information preserved'.")
    print("Low cosine similarity would mean the embedding SPACE itself")
    print("reorganized, not just shrank -- a different (more concerning) story.")


if __name__ == "__main__":
    main()

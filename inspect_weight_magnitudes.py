"""One-off diagnostic: track weight magnitude statistics across training
checkpoints, to check whether the weight-decay schedule is shrinking the
encoder's weights toward zero in a way that could explain the
encoder_embedding_std collapse seen in both the KHOP production run and the
teacher-temp-fix ablation.

Usage:
    uv run python inspect_weight_magnitudes.py
"""

import torch
from pathlib import Path

# BatchNorm running stats are buffers, not parameters -- weight decay (an
# optimizer-level effect applied only to .parameters()) can never touch them,
# so they're excluded to avoid diluting the signal.
EXCLUDE_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def tensor_stats(tensors):
    flat = torch.cat([t.flatten().float() for t in tensors])
    n = flat.numel()
    rms = flat.pow(2).mean().sqrt().item()
    mean_abs = flat.abs().mean().item()
    frac_near_zero = (flat.abs() < 1e-3).float().mean().item()
    return {"n": n, "rms": rms, "mean_abs": mean_abs, "frac_near_zero": frac_near_zero}


def summarize_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    encoder_tensors, head_tensors, all_tensors, bn_gamma_tensors = [], [], [], []
    for key, tensor in sd.items():
        if any(key.endswith(suffix) for suffix in EXCLUDE_SUFFIXES):
            continue
        all_tensors.append(tensor)
        if key.startswith("encoder."):
            encoder_tensors.append(tensor)
        elif key.startswith("head."):
            head_tensors.append(tensor)
        if "batch_norms" in key and key.endswith(".weight"):
            bn_gamma_tensors.append(tensor)

    return {
        "epoch": ckpt.get("epoch"),
        "all": tensor_stats(all_tensors),
        "encoder": tensor_stats(encoder_tensors),
        "head": tensor_stats(head_tensors),
        "bn_gamma": tensor_stats(bn_gamma_tensors) if bn_gamma_tensors else None,
    }


def print_trend(model_dir, epochs):
    print(f"\n=== {model_dir} ===")
    header = f"{'epoch':>6} | {'encoder RMS':>12} {'enc <1e-3':>10} | {'head RMS':>10} {'head <1e-3':>10} | {'BN gamma RMS':>13} {'BN gamma mean':>14}"
    print(header)
    print("-" * len(header))
    for ep in epochs:
        path = Path(f"models/{model_dir}/checkpoints/checkpoint_epoch_{ep}.pth")
        if not path.exists():
            continue
        stats = summarize_checkpoint(path)
        enc, head, bn = stats["encoder"], stats["head"], stats["bn_gamma"]
        bn_str = f"{bn['rms']:>13.5f} {bn['mean_abs']:>14.5f}" if bn else f"{'n/a':>13} {'n/a':>14}"
        print(
            f"{stats['epoch']:>6} | {enc['rms']:>12.5f} {enc['frac_near_zero']*100:>9.2f}% | "
            f"{head['rms']:>10.5f} {head['frac_near_zero']*100:>9.2f}% | {bn_str}"
        )


if __name__ == "__main__":
    print_trend("KHOP_B1024_H1024_200EP_GAT_9M_ZINC", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200])
    print_trend("TEACHER_TEMP_FIX_TEST_60EP", [10, 20, 30, 40, 50, 60])

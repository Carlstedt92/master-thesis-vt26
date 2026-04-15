"""Generate DINO-style atom heatmaps from a trained checkpoint.

The score explained per molecule is agreement between a global view and a masked local view:
  score = cosine(model(global), model(masked_local))

Per-atom importance is computed with gradient x input on node features and rendered as
atom highlights in an RDKit image.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
try:
    from rdkit.Chem.Draw import rdMolDraw2D
    _HAS_RDMOL_DRAW = True
except Exception:
    rdMolDraw2D = None
    _HAS_RDMOL_DRAW = False

from datahandling.graph_creation import smiles_to_pygdata
from model.config import ModelConfig
from model.gnn_model import GNNModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate atom heatmaps from a DINO checkpoint.")
    parser.add_argument("--model-name", required=True, help="Model directory under models/")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint filename under models/<model>/checkpoints")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: models/<model>/explainability/<checkpoint_stem>)")
    parser.add_argument("--smiles-file", default=None, help="Optional CSV file with a smiles column")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name when using --smiles-file")
    parser.add_argument("--num-samples", type=int, default=12, help="How many molecules to explain")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic masking/sampling")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda). Default uses config device")
    return parser.parse_args()


def _iter_smiles_from_csv(csv_path: Path, smiles_col: str) -> Iterable[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if smiles_col not in (reader.fieldnames or []):
            raise ValueError(f"Column '{smiles_col}' not found in {csv_path}")
        for row in reader:
            smiles = (row.get(smiles_col) or "").strip()
            if smiles:
                yield smiles


def _iter_smiles_from_smi_dir(data_dir: Path) -> Iterable[str]:
    smi_files = sorted(data_dir.glob("*.smi"))
    if not smi_files:
        raise ValueError(f"No .smi files found in {data_dir}")

    for smi_file in smi_files:
        with smi_file.open("r", encoding="utf-8") as handle:
            _ = handle.readline()  # header
            for line in handle:
                fields = line.strip().split()
                if not fields:
                    continue
                yield fields[0]


def _load_smiles(args: argparse.Namespace, config: ModelConfig) -> list[str]:
    if args.smiles_file:
        source = Path(args.smiles_file)
        smiles_iter = _iter_smiles_from_csv(source, args.smiles_col)
    else:
        data_path = Path(config.data_path)
        if data_path.is_file():
            smiles_iter = _iter_smiles_from_csv(data_path, "smiles")
        elif data_path.is_dir():
            smiles_iter = _iter_smiles_from_smi_dir(data_path)
        else:
            raise ValueError(
                "Could not infer SMILES source. Pass --smiles-file explicitly. "
                f"Config data_path not found: {data_path}"
            )

    smiles = []
    seen = set()
    for s in smiles_iter:
        if s in seen:
            continue
        seen.add(s)
        smiles.append(s)
        if len(smiles) >= args.num_samples:
            break

    if not smiles:
        raise ValueError("No valid SMILES found for explainability run")
    return smiles


def _sample_count(total_items: int, ratio: float) -> int:
    if total_items <= 0 or ratio <= 0:
        return 0
    return min(total_items, max(1, int(round(total_items * ratio))))


def _masked_local_x(x: torch.Tensor, ptr: list[tuple[int, int]], node_mask_ratio: float, feature_mask_ratio: float) -> torch.Tensor:
    out = x.clone()
    num_features = out.size(1) if out.dim() > 1 else 0
    for start, end in ptr:
        num_nodes = end - start

        node_k = _sample_count(num_nodes, node_mask_ratio)
        if node_k > 0:
            node_indices = random.sample(range(start, end), node_k)
            out[node_indices] = 0

        feat_k = _sample_count(num_features, feature_mask_ratio)
        if feat_k > 0:
            feat_indices = random.sample(range(num_features), feat_k)
            out[start:end, feat_indices] = 0

    return out


def _tensor_ptr_ranges(num_nodes: int) -> list[tuple[int, int]]:
    return [(0, num_nodes)]


def _normalize_scores(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    v_min = values.min()
    v_max = values.max()
    if torch.isclose(v_min, v_max):
        return torch.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


def _score_to_rgb(score: float) -> tuple[float, float, float]:
    # Blue (low) to red (high)
    s = max(0.0, min(1.0, float(score)))
    r = 0.2 + 0.8 * s
    g = 0.2 + 0.6 * (1.0 - abs(0.5 - s) * 2.0)
    b = 1.0 - 0.8 * s
    return (r, g, b)


def _draw_heatmap(smiles: str, atom_scores: list[float], out_path: Path) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Could not parse SMILES for rendering")

    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)

    if len(atom_scores) != mol.GetNumAtoms():
        raise ValueError(
            f"Atom score count ({len(atom_scores)}) does not match atom count ({mol.GetNumAtoms()})"
        )

    highlight_atoms = list(range(mol.GetNumAtoms()))
    highlight_atom_colors = {idx: _score_to_rgb(score) for idx, score in enumerate(atom_scores)}
    highlight_atom_radii = {idx: 0.18 + 0.30 * float(score) for idx, score in enumerate(atom_scores)}

    if not _HAS_RDMOL_DRAW:
        _draw_heatmap_fallback_matplotlib(
            mol=mol,
            atom_scores=atom_scores,
            out_path=out_path,
        )
        return

    drawer = rdMolDraw2D.MolDraw2DCairo(900, 700)
    draw_options = drawer.drawOptions()
    draw_options.addAtomIndices = True
    draw_options.bondLineWidth = 2

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightAtomRadii=highlight_atom_radii,
    )
    drawer.FinishDrawing()
    out_path.write_bytes(drawer.GetDrawingText())


def _draw_heatmap_fallback_matplotlib(mol: Chem.Mol, atom_scores: list[float], out_path: Path) -> None:
    # Import lazily so the script still runs if matplotlib is not installed and RDKit-Cairo works.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    xs = []
    ys = []
    colors = []
    sizes = []
    labels = []

    for atom_idx in range(n_atoms):
        p = conf.GetAtomPosition(atom_idx)
        xs.append(float(p.x))
        ys.append(float(p.y))
        score = float(atom_scores[atom_idx])
        colors.append(_score_to_rgb(score))
        sizes.append(250.0 + 850.0 * score)
        atom = mol.GetAtomWithIdx(atom_idx)
        labels.append(f"{atom.GetSymbol()}{atom_idx}")

    fig, ax = plt.subplots(figsize=(9, 7), dpi=140)

    # Draw bonds first so atom markers are layered on top.
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=(0.35, 0.35, 0.35), linewidth=2.0, zorder=1)

    ax.scatter(xs, ys, s=sizes, c=colors, edgecolors="black", linewidths=1.0, zorder=2)
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color="black", zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("Atom importance (fallback renderer)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _load_model(model_name: str, checkpoint_name: str, device: torch.device) -> tuple[GNNModel, ModelConfig, dict]:
    checkpoint_path = Path("models") / model_name / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint:
        raise KeyError("Checkpoint is missing config")

    config = ModelConfig.from_dict(checkpoint["config"])
    model = GNNModel.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, checkpoint


def _explain_smiles(
    model: GNNModel,
    smiles: str,
    device: torch.device,
    node_mask_ratio: float,
    feature_mask_ratio: float,
    explicit_hydrogens: bool,
    encode_hydrogen_count: bool,
) -> dict | None:
    data = smiles_to_pygdata(
        smiles,
        explicit_hydrogens=explicit_hydrogens,
        encode_hydrogen_count=encode_hydrogen_count,
    )
    if data is None or data.x is None or data.num_nodes == 0:
        return None

    x = data.x.to(device).detach().clone().requires_grad_(True)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else None
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    x_local = _masked_local_x(
        x,
        ptr=_tensor_ptr_ranges(x.size(0)),
        node_mask_ratio=node_mask_ratio,
        feature_mask_ratio=feature_mask_ratio,
    )

    model.zero_grad(set_to_none=True)
    z_global = model(x, edge_index, edge_attr, batch)
    z_local = model(x_local, edge_index, edge_attr, batch)

    score = F.cosine_similarity(z_global, z_local, dim=-1).mean()
    score.backward()

    grad = x.grad
    if grad is None:
        return None

    atom_importance = (grad * x).abs().sum(dim=1).detach().cpu()
    atom_importance = _normalize_scores(atom_importance)

    return {
        "smiles": smiles,
        "score": float(score.detach().cpu().item()),
        "atom_scores": [float(v) for v in atom_importance],
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    chosen_device = args.device if args.device else default_device
    device = torch.device(chosen_device)

    model, config, checkpoint = _load_model(args.model_name, args.checkpoint, device)

    checkpoint_stem = Path(args.checkpoint).stem
    output_dir = Path(args.output_dir) if args.output_dir else Path("models") / args.model_name / "explainability" / checkpoint_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles_list = _load_smiles(args, config)

    node_mask_ratio = float(getattr(config, "node_mask_ratio", 0.15))
    feature_mask_ratio = float(getattr(config, "feature_mask_ratio", 0.15))
    explicit_hydrogens = bool(getattr(config, "explicit_hydrogens", True))
    encode_hydrogen_count = bool(getattr(config, "encode_hydrogen_count", False))

    records = []
    for idx, smiles in enumerate(smiles_list):
        result = _explain_smiles(
            model=model,
            smiles=smiles,
            device=device,
            node_mask_ratio=node_mask_ratio,
            feature_mask_ratio=feature_mask_ratio,
            explicit_hydrogens=explicit_hydrogens,
            encode_hydrogen_count=encode_hydrogen_count,
        )
        if result is None:
            continue

        png_path = output_dir / f"heatmap_{idx:03d}.png"
        _draw_heatmap(smiles=result["smiles"], atom_scores=result["atom_scores"], out_path=png_path)

        result["image_path"] = str(png_path)
        records.append(result)
        print(f"[{idx + 1}/{len(smiles_list)}] Wrote {png_path}")

    summary = {
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "num_requested": int(args.num_samples),
        "num_rendered": int(len(records)),
        "explicit_hydrogens": explicit_hydrogens,
        "encode_hydrogen_count": encode_hydrogen_count,
        "node_mask_ratio": node_mask_ratio,
        "feature_mask_ratio": feature_mask_ratio,
        "records": records,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    csv_path = output_dir / "atom_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "smiles", "agreement_score", "atom_index", "atom_score", "image_path"])
        for idx, record in enumerate(records):
            for atom_idx, atom_score in enumerate(record["atom_scores"]):
                writer.writerow([
                    idx,
                    record["smiles"],
                    record["score"],
                    atom_idx,
                    atom_score,
                    record["image_path"],
                ])

    print("Done.")
    print(f"Summary: {summary_path}")
    print(f"Atom score CSV: {csv_path}")


if __name__ == "__main__":
    main()

# Master Thesis VT-26

DINO-style self-supervised pretraining of graph neural network molecular encoders
(GAT / GINE, via PyTorch Geometric) on the ZINC dataset, evaluated on downstream
molecular property prediction tasks from MoleculeNet (LIPO, BACE, Tox21) and the
TDC ADMET benchmark (BBB_Martins, hERG, AMES).

## Setup

```bash
uv sync
```

Dependencies and their pinned versions are managed with [uv](https://docs.astral.sh/uv/)
via `pyproject.toml` / `uv.lock`.

## Repository layout

- `model/` — GNN model definitions (`GATEncoder`, `GINEEncoder`, DINO projection head,
  downstream regression/classification heads).
- `training/` — the DINO self-supervised training loop and its schedule/momentum logic.
- `datahandling/` — SMILES-to-graph conversion, graph augmentations (k-hop subgraph,
  functional-group masking), dataset loaders (on-the-fly and precomputed-shard).
- `evaluation/` — downstream dataset loaders and splitters (MoleculeNet + TDC ADMET),
  kNN/MLP/RF probe evaluation utilities.
- `configs/` — JSON5 training configs. `configs/default.json5` is the baseline; other
  files are per-experiment variants (architecture, augmentation, or ablation configs).
- `models/` — expected to be a symlink to wherever trained checkpoints/results are
  actually stored (e.g. shared cluster storage) — not part of this repo's own content.
- `*.sbatch` — SLURM launcher scripts for the cluster this project was run on. Each one
  has `#SBATCH -A YOUR_SLURM_ACCOUNT` and `#SBATCH --mail-user=YOUR_EMAIL@example.com`
  placeholders — fill those in for your own account before submitting.

## SSL pretraining

```bash
uv run python train.py --config configs/default.json5
```

`--config` is optional; it defaults to `configs/default.json5` if omitted. Training can
be resumed from a checkpoint with `continue_training.py --model <name>` (see
`--checkpoint-path` / `--checkpoint-name` / `--checkpoint-epoch` for selecting which
checkpoint to resume from). `configs/*.json5` files hold every hyperparameter — dataset
path, encoder type/size, augmentation strategy, optimizer/schedule settings, etc.

ZINC SMILES data can be fetched with `download_zinc_data.py`.

### Precomputed graph workflow (optional)

To reduce CPU load from repeated SMILES parsing during training, PyG graphs can be
precomputed into shards ahead of time:

```bash
uv run python precompute_graphs.py --input data/zinc/zinc_data --output data/zinc/precomputed_graphs --pattern "*.smi" --shard-size 50000
```

Then in the training config:

```json5
"use_precomputed": true,
"precomputed_data_path": "data/zinc/precomputed_graphs",
```

Keep `"use_precomputed": false` to use the on-the-fly SMILES loader instead. Optional
RAM-caching toggles: `"cache_data_in_memory"` (source rows) and
`"precomputed_cache_in_memory"` (precomputed graphs). Augmentations are still generated
online in both modes, so DINO local/global view matching remains correct via
`graph_idx`. With `num_workers > 0`, each worker holds its own dataset instance, so RAM
usage scales with worker count.

## Downstream evaluation

Two separate evaluation pipelines exist, answering different questions:

### 1. Frozen-embedding probes (MLP / RF)

Freezes the pretrained encoder, extracts embeddings once, then fits a small MLP head and
a Random Forest on top — the standard way this project compares SSL pretraining recipes
against each other, since every model gets the exact same untuned probe. Split into two
phases so the GPU-bound embedding extraction and CPU-bound probe fitting run as separate
jobs:

```bash
uv run python eval_phase1_extract_embeddings.py --models <name1,name2,...> --device cuda --cache-dir embedding_cache
uv run python eval_phase2_fit_probes.py --models <name1,name2,...> --device cpu --cache-dir embedding_cache --output eval_many_models_mlp_rf_results.json
```

`--datasets` (comma-separated, default `lipo,bace,tox21`) selects which downstream
datasets to evaluate on; add `bbb_martins,herg,ames` for the TDC ADMET datasets. Each
dataset has its own `--<dataset>-data-dir` and `--<dataset>-splitter` args — TDC datasets
and BACE default to `admet_benchmark` (TDC's official protocol: a fixed test set + a
seeded reshuffle of the train/valid pool, 5 seeds), LIPO/Tox21 default to `random` splits
over 5 seeds.

### 2. End-to-end finetuning (fixed recipe)

Unfreezes the whole model and finetunes it end-to-end with one fixed recipe (lr, weight
decay, epoch count identical across every model), again split into GPU/CPU phases:

```bash
uv run python finetune_phase1_train.py --models <name1,name2,...> --device cuda
uv run python finetune_phase2_eval.py --models <name1,name2,...> --device auto --output finetune_eval_many_models_mlp_rf_results.json
```

Same `--datasets`/per-dataset data-dir/splitter args as the probe pipeline. The fixed
recipe is what makes cross-model comparisons from this pipeline valid — see
`finetune_hyperopt_search.py`'s docstring for why that matters.

### 3. Hyperparameter-tuned finetuning (single model, single dataset)

`finetune_hyperopt_search.py` is a deliberately separate script that runs a real
Hyperopt (TPE) search over finetuning hyperparameters — learning rate, weight decay,
epoch count, task-head depth/width, dropout, LR schedule — for one (model, dataset)
pair at a time:

```bash
uv run python finetune_hyperopt_search.py --model <name> --dataset <lipo|bace|tox21|bbb_martins|herg|ames> --device cuda
```

This is kept out of the standardized pipelines above on purpose: giving one model a
tuning budget the others didn't get would confound "does this model's representation
help" with "did this run get searched harder." The search itself never touches the test
set — only the validation split, averaged across a few seeds — and the resulting
hyperparameters are evaluated once across the dataset's full official seed set for the
reported result.

## License

MIT — see [LICENSE](LICENSE).

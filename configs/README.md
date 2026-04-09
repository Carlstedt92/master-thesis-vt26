# Config Notes

JSON does not support inline comments, so this file documents the main fields used by `train.py --config ...`.

## Core identity
- `name`: Run name used for output folders under `models/`.
- `head_type`: Keep as `dino` for SSL pretraining.
- `encoder_type`: `GINE` or `GAT`.

## Data
- `data_path`: Either a CSV path or a directory containing `.smi` files.
- `num_features`, `edge_features`: Input dimensions expected by the model.

## Augmentation
- `local_views`: Number of local views per graph.
- `local_augmentation_mode`: `k_hop` or `masking`.
- `k_hops`: Used when `local_augmentation_mode` is `k_hop`.
- `node_mask_ratio`, `feature_mask_ratio`: Used when mode is `masking`.

## Model size
- `hidden_dim`, `num_layers`, `dropout`, `epsilon`
- `projection_hidden_dim`, `projection_output_dim`, `projection_layers`

## Optimization
- `num_epochs`, `batch_size`
- `auto_scale_lr`, `lr_scale_base`, `lr_scale_reference_batch_size`
- `learning_rate`
- `weight_decay`, `weight_decay_start`, `weight_decay_end`
- `warmup_epochs`, `final_learning_rate`

## DINO-specific
- `teacher_temp`, `teacher_temp_final`, `teacher_temp_warmup_epochs`
- `student_temp`
- `teacher_momentum`, `center_momentum`

## Online evaluation (kNN-only)
- `online_eval_enabled`
- `online_eval_every_n_epochs`
- `online_eval_datasets`
- `online_eval_fixed_k`
- `online_eval_top_k_checkpoints`

## Runtime
- `seed`, `device`, `num_workers`

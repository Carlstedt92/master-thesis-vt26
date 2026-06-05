"""Training script using your existing dataloader setup with DINO SSL.

Supports modular training with configurable models saved to models/{model_name}/
"""

import json
import os
import torch
import torch.optim as optim
import math
from pathlib import Path
from model.gnn_model import GNNModel
from model.dino_ssl import DINOGraphSSL, cosine_scheduler
from datahandling.dataloader_creation import DataLoaderCreator
from model.config import ModelConfig
from training.online_evaluator import OnlineDownstreamEvaluator
from training.train_manager import TrainingManager
import time
from collections import defaultdict


SCHEDULE_STATE_FILENAME = "schedule_state.pt"


def _build_schedule_bundle(config: ModelConfig, train_loader_len: int) -> dict:
    """Build all per-iteration schedules used by DINO training."""
    if getattr(config, "auto_scale_lr", False):
        scaled_learning_rate = config.lr_scale_base * (
            config.batch_size / config.lr_scale_reference_batch_size
        )
    else:
        scaled_learning_rate = config.learning_rate

    lr_schedule = torch.from_numpy(
        cosine_scheduler(
            base_value=scaled_learning_rate,
            final_value=config.final_learning_rate,
            epochs=config.num_epochs,
            niter_per_ep=train_loader_len,
            warmup_epochs=config.warmup_epochs,
            start_warmup_value=0.0,
        ).astype("float32")
    )

    wd_schedule = torch.from_numpy(
        cosine_scheduler(
            base_value=config.weight_decay_start,
            final_value=config.weight_decay_end,
            epochs=config.num_epochs,
            niter_per_ep=train_loader_len,
            warmup_epochs=0,
            start_warmup_value=config.weight_decay_start,
        ).astype("float32")
    )

    momentum_schedule = torch.from_numpy(
        cosine_scheduler(
            base_value=config.teacher_momentum,
            final_value=1.0,
            epochs=config.num_epochs,
            niter_per_ep=train_loader_len,
        ).astype("float32")
    )

    warmup_epochs = max(0, min(int(config.teacher_temp_warmup_epochs), int(config.num_epochs)))
    warmup_iters = int(warmup_epochs * train_loader_len)
    total_iters = int(config.num_epochs * train_loader_len)
    if warmup_iters > 0:
        warmup_temp = torch.linspace(config.teacher_temp, config.teacher_temp_final, warmup_iters)
    else:
        warmup_temp = torch.tensor([], dtype=torch.float32)
    remain_iters = max(total_iters - warmup_iters, 0)
    hold_temp = torch.full((remain_iters,), float(config.teacher_temp_final), dtype=torch.float32)
    teacher_temp_schedule = torch.cat((warmup_temp.float(), hold_temp))

    return {
        "meta": {
            "num_epochs": int(config.num_epochs),
            "niter_per_ep": int(train_loader_len),
            "warmup_epochs": int(min(max(int(config.warmup_epochs), 0), int(config.num_epochs))),
            "teacher_temp_warmup_epochs": int(warmup_epochs),
        },
        "lr_schedule": lr_schedule,
        "wd_schedule": wd_schedule,
        "momentum_schedule": momentum_schedule,
        "teacher_temp_schedule": teacher_temp_schedule,
    }


def _load_schedule_bundle(schedule_path: Path) -> dict | None:
    if not schedule_path.exists():
        return None
    bundle = torch.load(schedule_path, map_location="cpu", weights_only=False)
    return bundle if isinstance(bundle, dict) else None


def _pad_or_trim_schedule(schedule: torch.Tensor, required_length: int) -> torch.Tensor:
    if schedule.numel() >= required_length:
        return schedule[:required_length].clone()
    if schedule.numel() == 0:
        return torch.zeros(required_length, dtype=torch.float32)
    pad_length = required_length - schedule.numel()
    pad_value = schedule[-1].detach().clone().reshape(1)
    pad = pad_value.repeat(pad_length)
    return torch.cat((schedule, pad), dim=0)


def _prepare_schedule_bundle(config: ModelConfig, train_loader_len: int, schedule_path: Path) -> dict:
    required_length = int(config.num_epochs * train_loader_len)
    bundle = _load_schedule_bundle(schedule_path)

    if bundle is None:
        bundle = _build_schedule_bundle(config, train_loader_len)
        torch.save(bundle, schedule_path)
        print(f"✓ Schedule state saved: {schedule_path}")
        return bundle

    loaded_niter = int(bundle.get("meta", {}).get("niter_per_ep", train_loader_len))
    if loaded_niter != train_loader_len:
        print(
            "Warning: loaded schedule state was built with a different batch count; "
            "rebuilding schedules for the current dataloader."
        )
        bundle = _build_schedule_bundle(config, train_loader_len)
        torch.save(bundle, schedule_path)
        print(f"✓ Schedule state saved: {schedule_path}")
        return bundle

    for key in ["lr_schedule", "wd_schedule", "momentum_schedule", "teacher_temp_schedule"]:
        bundle[key] = _pad_or_trim_schedule(bundle[key].float(), required_length)

    if int(bundle.get("meta", {}).get("num_epochs", 0)) < int(config.num_epochs):
        bundle.setdefault("meta", {})["num_epochs"] = int(config.num_epochs)
        bundle["meta"]["niter_per_ep"] = int(train_loader_len)
        torch.save(bundle, schedule_path)
        print(f"✓ Schedule state extended and saved: {schedule_path}")

    return bundle


def dino_train(config: ModelConfig, resume_checkpoint_path: str | None = None):
    """
    Train GNN with DINO using ModelConfig for modular training.
    
    Args:
        config: ModelConfig object with all model and training parameters
        
    Returns:
        Tuple: (dino_ssl, manager)
    """
    print("="*70)
    print("DINO SSL Training with Multi-Crop Augmentation")
    print("="*70)
    print(f"\nModel: {config.name}")
    print(f"Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Head type: {config.head_type}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size} graphs")
    print(f"  Learning rate (configured): {config.learning_rate}")
    print(f"  Layers: {config.num_layers}, Hidden dim: {config.hidden_dim}")
    local_views = getattr(config, 'local_views', 4)
    print(f"  Views per graph: 2 global + {local_views} local = {2 + local_views} total")
    effective_batch_size = config.batch_size * (2 + local_views)
    print(f"  Effective batch size: {effective_batch_size} views")
    print()
    if resume_checkpoint_path:
        print(f"  Resume checkpoint: {resume_checkpoint_path}")

    # Optional DINO-style linear LR scaling rule: lr = base * (batch_size / reference_batch_size)
    if getattr(config, "auto_scale_lr", False):
        scaled_learning_rate = config.lr_scale_base * (
            config.batch_size / config.lr_scale_reference_batch_size
        )
    else:
        scaled_learning_rate = config.learning_rate
    print(f"  Learning rate (used): {scaled_learning_rate}")
    profile_timing = bool(getattr(config, "profile_timing", False))
    profile_every = int(getattr(config, "profile_log_every_n_batches", 50))
    if profile_timing:
        print(f"  Profiling: enabled (log every {profile_every} batches)")

    online_eval_enabled = bool(getattr(config, "online_eval_enabled", False))
    online_eval_every_n_epochs = int(getattr(config, "online_eval_every_n_epochs", 1))
    online_eval_fixed_k = int(getattr(config, "online_eval_fixed_k", 5))
    online_eval_top_k = int(getattr(config, "online_eval_top_k_checkpoints", 5))
    online_eval_datasets = [
        item.strip()
        for item in str(getattr(config, "online_eval_datasets", "lipo")).split(",")
        if item.strip()
    ]

    if online_eval_enabled:
        print("  Online eval: enabled")
        print(f"    Datasets: {','.join(online_eval_datasets)}")
        print(f"    Every N epochs: {online_eval_every_n_epochs}")
        print(f"    Fixed k: {online_eval_fixed_k}")
        print("    Mode: kNN-only")
        print(f"    Top-k checkpoints kept: {online_eval_top_k}")
    else:
        print("  Online eval: disabled")

    # Initialize training manager
    manager = TrainingManager(config)

    # If resuming, restore existing loss/eval history so new epochs append cleanly.
    history_path = os.path.join(manager.model_dir, "loss_history.json")
    if resume_checkpoint_path and os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history_payload = json.load(f)
            dino_hist = history_payload.get("DINO_Loss", [])
            eval_hist = history_payload.get("Evaluation_Loss", {})
            if isinstance(dino_hist, list):
                manager.dino_loss_history = dino_hist
                if dino_hist:
                    manager.best_loss = min(float(item.get("train_loss", float("inf"))) for item in dino_hist)
                    val_losses = [float(item["val_loss"]) for item in dino_hist if item.get("val_loss") is not None]
                    if val_losses:
                        manager.best_val_loss = min(val_losses)
                        for item in dino_hist:
                            if item.get("val_loss") == manager.best_val_loss:
                                manager.best_val_epoch = int(item.get("epoch", 0))
                                break
            if isinstance(eval_hist, dict):
                manager.eval_loss_history = eval_hist
                online_hist = eval_hist.get("online_eval", [])
                if isinstance(online_hist, list):
                    manager.online_eval_history = online_hist
        except Exception as exc:
            print(f"Warning: failed to restore existing loss history ({exc}). Continuing with fresh history.")

    online_evaluator = None
    if online_eval_enabled:
        online_evaluator = OnlineDownstreamEvaluator(
            dataset_names=online_eval_datasets,
            fixed_k=online_eval_fixed_k,
            fingerprint_radius=2,
            fingerprint_nbits=2048,
        )
    
    # Create dataloaders - everything comes from config
    creator = DataLoaderCreator(config)
    validation_enabled = bool(getattr(config, "validation_enabled", False)) and float(getattr(config, "validation_split", 0.0)) > 0.0
    if validation_enabled:
        train_loader, val_loader = creator.create_train_val_dataloaders_auto()
    else:
        train_loader = creator.create_dataloader_auto()
        val_loader = None
    print(f"✓ DataLoader created with {len(train_loader)} batches\n")
    if val_loader is not None:
        print(f"✓ Validation DataLoader created with {len(val_loader)} batches\n")
    
    # Initialize GNN student model
    student_model = GNNModel.from_config(config)
    
    num_params = sum(p.numel() for p in student_model.parameters())
    print(f"✓ Model parameters: {num_params:,}\n")
    
    # Initialize DINO SSL framework
    dino_ssl = DINOGraphSSL.from_config(
        student_model=student_model,
        config=config,
        teacher_model=None,  # Will be created as copy
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        dino_ssl.student.parameters(),
        lr=scaled_learning_rate,
        weight_decay=config.weight_decay_start
    )
    
    schedule_path = Path(manager.model_dir) / SCHEDULE_STATE_FILENAME
    schedule_bundle = _prepare_schedule_bundle(config, len(train_loader), schedule_path)
    lr_schedule = schedule_bundle["lr_schedule"]
    wd_schedule = schedule_bundle["wd_schedule"]
    momentum_schedule = schedule_bundle["momentum_schedule"]
    teacher_temp_schedule = schedule_bundle["teacher_temp_schedule"]
    
    print("Starting training...\n")
    
    # Training loop
    start_epoch = 0
    iteration = 0

    if resume_checkpoint_path:
        checkpoint = torch.load(resume_checkpoint_path, map_location=torch.device(config.device), weights_only=False)
        dino_ssl.student.load_state_dict(checkpoint["model_state_dict"])
        dino_ssl.teacher.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        iteration = start_epoch * len(train_loader)
        print(f"✓ Resumed from checkpoint epoch {start_epoch} (next epoch to run)")
        if start_epoch >= config.num_epochs:
            raise ValueError(
                f"Checkpoint epoch {start_epoch} is >= target num_epochs ({config.num_epochs}). "
                "Increase num_epochs to continue training."
            )
    
    # Training time tracking
    start_time = time.time()

    collapse_loss_ref = math.log(float(config.projection_output_dim))

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config.num_epochs} started at {time.strftime('%H:%M:%S')}")
        print(f"{'='*70}")
        epoch_loss = 0
        num_batches = 0
        epoch_trained_graphs = 0
        epoch_diag_sums = {
            "teacher_entropy": 0.0,
            "student_entropy": 0.0,
            "embedding_std": 0.0,
            "encoder_embedding_std": 0.0,
        }
        epoch_timing_sums = defaultdict(float)
        epoch_timing_sums["batch_load"] = 0.0
        epoch_timing_sums["batch_total"] = 0.0
        epoch_timing_sums["collate_total"] = 0.0
        epoch_timing_sums["filter_invalid"] = 0.0
        epoch_timing_sums["augmentation"] = 0.0
        epoch_timing_sums["normalize_flatten"] = 0.0
        epoch_timing_sums["batch_from_data_list"] = 0.0
        
        batch_load_start = time.time()
        for batch_idx, batch in enumerate(train_loader):
            batch_load_time = time.time() - batch_load_start
            batch_start_time = time.time()

            if batch is None:
                # All samples in this worker batch were invalid SMILES.
                batch_load_start = time.time()
                continue

            if iteration >= len(lr_schedule):
                raise RuntimeError(
                    f"Schedule exhausted at iteration {iteration}; "
                    f"schedule length is {len(lr_schedule)}."
                )

            num_unique_graphs = len(torch.unique(batch['graph_idx']))
            epoch_trained_graphs += num_unique_graphs
            
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[iteration]
                param_group['weight_decay'] = wd_schedule[iteration]
            
            # Update teacher momentum
            dino_ssl.teacher_momentum = momentum_schedule[iteration]

            # Update teacher temperature (loss-side teacher softmax temperature)
            dino_ssl.loss_fn.teacher_temp = float(teacher_temp_schedule[iteration])
            
            # Training step - batch contains all augmented views
            # Teacher will automatically filter for global views (view==1)
            # Student sees all views
            # Loss computed between matching graph_idx
            train_step_start = time.time()
            step_info = dino_ssl.train_step(batch, optimizer)
            train_step_time = time.time() - train_step_start
            loss = step_info["loss"]
            step_metrics = step_info.get("metrics", {})
            step_timing = step_info.get("timing", {})
            collate_timing = getattr(batch, "profile_timing", {}) or {}

            epoch_loss += loss
            num_batches += 1
            iteration += 1
            for key in epoch_diag_sums:
                epoch_diag_sums[key] += float(step_metrics.get(key, 0.0))
            for key, value in step_timing.items():
                epoch_timing_sums[key] += float(value)
            for key, value in collate_timing.items():
                epoch_timing_sums[key] += float(value)
            epoch_timing_sums["batch_load"] += float(batch_load_time)
            
            total_batch_time = time.time() - batch_start_time
            epoch_timing_sums["batch_total"] += float(total_batch_time)
            if collate_timing:
                epoch_timing_sums["loader_wait_minus_collate"] += float(max(batch_load_time - collate_timing.get("collate_total", 0.0), 0.0))
            
            # Batch-level per-batch printing disabled; only epoch summaries printed below
            
            # Start timing for next batch load (at end of every iteration)
            batch_load_start = time.time()
        
        # Epoch summary
        if num_batches == 0:
            raise RuntimeError("No valid batches produced. Check dataset for invalid SMILES.")

        val_loss = None
        val_num_batches = 0
        val_diag_sums = {
            "teacher_entropy": 0.0,
            "student_entropy": 0.0,
            "embedding_std": 0.0,
            "encoder_embedding_std": 0.0,
        }
        if val_loader is not None:
            val_loss_total = 0.0
            for val_batch in val_loader:
                if val_batch is None:
                    continue
                val_step = dino_ssl.eval_step(val_batch)
                val_loss_total += float(val_step["loss"])
                val_num_batches += 1
                for key in val_diag_sums:
                    val_diag_sums[key] += float(val_step["metrics"].get(key, 0.0))
            if val_num_batches > 0:
                val_loss = val_loss_total / val_num_batches
            else:
                print("Warning: validation loader produced no valid batches.")

        total_graphs_in_epoch = len(train_loader.dataset)
        epoch_invalid_graphs = total_graphs_in_epoch - epoch_trained_graphs
        valid_pct = (epoch_trained_graphs / total_graphs_in_epoch) * 100 if total_graphs_in_epoch > 0 else 0.0

        avg_loss = epoch_loss / num_batches
        avg_teacher_entropy = epoch_diag_sums["teacher_entropy"] / num_batches
        avg_student_entropy = epoch_diag_sums["student_entropy"] / num_batches
        avg_embedding_std = epoch_diag_sums["embedding_std"] / num_batches
        avg_encoder_embedding_std = epoch_diag_sums["encoder_embedding_std"] / num_batches

        val_teacher_entropy = None
        val_student_entropy = None
        val_embedding_std = None
        val_encoder_embedding_std = None
        if val_num_batches > 0:
            val_teacher_entropy = val_diag_sums["teacher_entropy"] / val_num_batches
            val_student_entropy = val_diag_sums["student_entropy"] / val_num_batches
            val_embedding_std = val_diag_sums["embedding_std"] / val_num_batches
            val_encoder_embedding_std = val_diag_sums["encoder_embedding_std"] / val_num_batches

        collapse_warning = (
            abs(avg_loss - collapse_loss_ref) < 0.03
            and avg_teacher_entropy > 0.95 * collapse_loss_ref
            and avg_embedding_std < 0.02
        )

        epoch_diagnostics = {
            "teacher_entropy": avg_teacher_entropy,
            "student_entropy": avg_student_entropy,
            "embedding_std": avg_embedding_std,
            "encoder_embedding_std": avg_encoder_embedding_std,
            "collapse_warning": collapse_warning,
        }

        val_diagnostics = None
        if val_num_batches > 0:
            val_diagnostics = {
                "teacher_entropy": val_teacher_entropy,
                "student_entropy": val_student_entropy,
                "embedding_std": val_embedding_std,
                "encoder_embedding_std": val_encoder_embedding_std,
            }

        best_metric = val_loss if val_loss is not None else avg_loss
        is_best = best_metric < (manager.best_val_loss if val_loss is not None else manager.best_loss)
        manager.record_loss(
            epoch,
            avg_loss,
            diagnostics=epoch_diagnostics,
            val_loss=val_loss,
            val_diagnostics=val_diagnostics,
        )
        # Persist after each epoch so interrupted runs keep partial history.
        manager.save_loss_history(verbose=False)
        
        elapsed_total = (time.time() - start_time) / 60
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config.num_epochs} Complete at {time.strftime('%H:%M:%S')}")
        print(f"Average Loss: {avg_loss:.6f}")
        if val_loss is not None:
            print(f"Validation Loss: {val_loss:.6f}")
        print(f"Graphs trained: {epoch_trained_graphs}/{total_graphs_in_epoch} ({valid_pct:.2f}% valid)")
        print(f"Invalid SMILES skipped: {epoch_invalid_graphs}")
        print(
            f"Diagnostics: teacher_entropy={avg_teacher_entropy:.4f}, "
            f"student_entropy={avg_student_entropy:.4f}, "
            f"embedding_std={avg_embedding_std:.4f}, "
            f"encoder_embedding_std={avg_encoder_embedding_std:.4f}, "
            f"collapse={collapse_warning}"
        )
        if val_loss is not None:
            print(
                f"Validation diagnostics: teacher_entropy={val_teacher_entropy:.4f}, "
                f"student_entropy={val_student_entropy:.4f}, "
                f"embedding_std={val_embedding_std:.4f}, "
                f"encoder_embedding_std={val_encoder_embedding_std:.4f}"
            )
        if profile_timing and num_batches > 0:
            print("Timing summary (avg seconds/batch):")
            print(f"  batch_load: {epoch_timing_sums['batch_load'] / num_batches:.3f}")
            print(f"  collate_total: {epoch_timing_sums['collate_total'] / num_batches:.3f}")
            for key in [
                "filter_invalid",
                "augmentation",
                "normalize_flatten",
                "batch_from_data_list",
                "base_batch_from_data_list",
                "global_clone",
                "local_masking",
                "combine_views",
                "loader_wait_minus_collate",
            ]:
                if key in epoch_timing_sums:
                    print(f"  {key}: {epoch_timing_sums[key] / num_batches:.3f}")
            if epoch_timing_sums["loader_wait_minus_collate"] > 0:
                print(f"  loader_wait_minus_collate: {epoch_timing_sums['loader_wait_minus_collate'] / num_batches:.3f}")
            print(f"  batch_total: {epoch_timing_sums['batch_total'] / num_batches:.3f}")
            for key in ["to_device", "student_forward", "teacher_forward", "loss_compute", "backward_step", "ema_and_center", "train_step_total"]:
                print(f"  {key}: {epoch_timing_sums[key] / num_batches:.3f}")
        print(f"Total elapsed time: {elapsed_total:.1f} min")
        print(f"{'='*70}")

        saved_online_path = None
        if online_evaluator is not None and (epoch + 1) % max(1, online_eval_every_n_epochs) == 0:
            print(f"  • Running online downstream eval (fixed k={online_eval_fixed_k})...")
            online_eval_result = online_evaluator.evaluate_model(
                dino_ssl.student,
                torch.device(config.device),
                explicit_hydrogens=bool(getattr(config, "explicit_hydrogens", True)),
                encode_hydrogen_count=bool(getattr(config, "encode_hydrogen_count", False)),
            )
            aggregate_score = online_eval_result.get("aggregate_primary_score", float("-inf"))
            print(f"    ✓ Online eval done | aggregate validation score={aggregate_score:.6f}")

            saved_online_path = manager.update_top_eval_checkpoints(
                epoch=epoch,
                model=dino_ssl.student,
                optimizer=optimizer,
                ssl_loss=avg_loss,
                eval_result=online_eval_result,
                top_k=online_eval_top_k,
            )
            manager.record_online_eval(
                epoch=epoch,
                ssl_loss=avg_loss,
                eval_result=online_eval_result,
                saved_path=saved_online_path,
            )

            if saved_online_path is not None:
                print(f"    ✓ Top-{online_eval_top_k} checkpoint updated: {saved_online_path}")
            else:
                print(f"    • Checkpoint not in top-{online_eval_top_k}; not saved")

            manager.save_loss_history(verbose=False)
        
        # Save checkpoints
        manager.save_checkpoint(
            epoch,
            dino_ssl.student,
            optimizer,
            avg_loss,
            is_best=is_best,
            metric_value=val_loss,
        )
    
    # Save final results
    # Always save a final checkpoint for reproducibility
    if manager.dino_loss_history:
        last_epoch = len(manager.dino_loss_history) - 1
        last_loss = manager.dino_loss_history[-1].get("train_loss")
        try:
            manager.save_final_checkpoint(last_epoch, dino_ssl.student, optimizer, loss=last_loss)
        except Exception as exc:
            print(f"Warning: failed to save final checkpoint: {exc}")

    manager.save_loss_history()
    manager.save_model_metadata()
    manager.save_dino_metadata()
    manager.save_online_eval_metadata()
    
    # Training complete summary
    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*70}")
    print(f"✓ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total training time: {total_time:.1f} minutes ({total_time/60:.2f} hours)")
    print(f"Model: {config.name}")
    print(f"Epochs trained: {config.num_epochs}")
    print(f"Results saved to: models/{config.name}/")
    if manager.best_loss is not None:
        print(f"Best SSL loss: {manager.best_loss:.6f}")
    print(f"{'='*70}\\n")
    
    return dino_ssl, manager

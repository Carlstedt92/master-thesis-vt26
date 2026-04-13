"""Training script using your existing dataloader setup with DINO SSL.

Supports modular training with configurable models saved to models/{model_name}/
"""

import torch
import torch.optim as optim
import math
from model.gnn_model import GNNModel
from model.dino_ssl import DINOGraphSSL, cosine_scheduler
from datahandling.dataloader_creation import DataLoaderCreator
from model.config import ModelConfig
from training.online_evaluator import OnlineDownstreamEvaluator
from training.train_manager import TrainingManager
import time
from collections import defaultdict


def dino_train(config: ModelConfig):
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

    online_evaluator = None
    if online_eval_enabled:
        online_evaluator = OnlineDownstreamEvaluator(
            dataset_names=online_eval_datasets,
            fixed_k=online_eval_fixed_k,
            fingerprint_radius=2,
            fingerprint_nbits=2048,
        )
    
    # Create dataloader - everything comes from config
    creator = DataLoaderCreator(config)
    train_loader = creator.create_dataloader_auto()
    print(f"✓ DataLoader created with {len(train_loader)} batches\n")
    
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
    
    # Learning rate scheduler
    lr_schedule = cosine_scheduler(
        base_value=scaled_learning_rate,
        final_value=config.final_learning_rate,
        epochs=config.num_epochs,
        niter_per_ep=len(train_loader),
        warmup_epochs=config.warmup_epochs,
        start_warmup_value=0.0
    )

    # Weight decay cosine schedule (paper-style: 0.04 -> 0.4)
    wd_schedule = cosine_scheduler(
        base_value=config.weight_decay_start,
        final_value=config.weight_decay_end,
        epochs=config.num_epochs,
        niter_per_ep=len(train_loader),
        warmup_epochs=0,
        start_warmup_value=config.weight_decay_start,
    )
    
    # Teacher momentum scheduler
    momentum_schedule = cosine_scheduler(
        base_value=config.teacher_momentum,
        final_value=1.0,
        epochs=config.num_epochs,
        niter_per_ep=len(train_loader)
    )

    # Teacher temperature schedule: linear warmup then hold final value.
    warmup_iters = int(config.teacher_temp_warmup_epochs * len(train_loader))
    total_iters = int(config.num_epochs * len(train_loader))
    if warmup_iters > 0:
        warmup_temp = torch.linspace(config.teacher_temp, config.teacher_temp_final, warmup_iters)
    else:
        warmup_temp = torch.tensor([], dtype=torch.float32)
    remain_iters = max(total_iters - warmup_iters, 0)
    hold_temp = torch.full((remain_iters,), float(config.teacher_temp_final), dtype=torch.float32)
    teacher_temp_schedule = torch.cat((warmup_temp, hold_temp)).numpy()
    
    print("Starting training...\n")
    
    # Training loop
    iteration = 0
    
    # Training time tracking
    start_time = time.time()

    collapse_loss_ref = math.log(float(config.projection_output_dim))

    for epoch in range(config.num_epochs):
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
            
            # Print progress every 100 batches
            should_print_batch = batch_idx % 100 == 0 or (profile_timing and batch_idx < 10)
            if should_print_batch:
                current_lr = lr_schedule[iteration-1]
                current_momentum = momentum_schedule[iteration-1]
                current_wd = wd_schedule[iteration-1]
                current_teacher_temp = teacher_temp_schedule[iteration-1]
                
                # Count views in batch for reporting
                num_global = (batch['view'] == 1).sum().item()
                num_local = (batch['view'] == 0).sum().item()
                # GPU memory usage
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1e9  # GB
                
                elapsed_time_min = (time.time() - start_time) / 60
                print(f"  Batch [{batch_idx+1:4d}/{len(train_loader)}] Loss: {loss:.4f} | "
                      f"LR: {current_lr:.2e} | GPU Mem: {gpu_mem_allocated:.1f}/{gpu_mem_reserved:.1f}GB | "
                      f"Elapsed: {elapsed_time_min:.1f}min")
                if profile_timing and step_timing:
                    print(
                        "    Timing (s): "
                        f"load={batch_load_time:.3f}, "
                        f"collate={collate_timing.get('collate_total', 0.0):.3f}, "
                        f"to_device={step_timing.get('to_device', 0.0):.3f}, "
                        f"student_fwd={step_timing.get('student_forward', 0.0):.3f}, "
                        f"teacher_fwd={step_timing.get('teacher_forward', 0.0):.3f}, "
                        f"loss={step_timing.get('loss_compute', 0.0):.3f}, "
                        f"backward={step_timing.get('backward_step', 0.0):.3f}, "
                        f"ema={step_timing.get('ema_and_center', 0.0):.3f}, "
                        f"step_total={step_timing.get('train_step_total', 0.0):.3f}, "
                        f"batch_total={total_batch_time:.3f}"
                    )
            
            # Start timing for next batch load (at end of every iteration)
            batch_load_start = time.time()
        
        # Epoch summary
        if num_batches == 0:
            raise RuntimeError("No valid batches produced. Check dataset for invalid SMILES.")

        total_graphs_in_epoch = len(train_loader.dataset)
        epoch_invalid_graphs = total_graphs_in_epoch - epoch_trained_graphs
        valid_pct = (epoch_trained_graphs / total_graphs_in_epoch) * 100 if total_graphs_in_epoch > 0 else 0.0

        avg_loss = epoch_loss / num_batches
        avg_teacher_entropy = epoch_diag_sums["teacher_entropy"] / num_batches
        avg_student_entropy = epoch_diag_sums["student_entropy"] / num_batches
        avg_embedding_std = epoch_diag_sums["embedding_std"] / num_batches

        collapse_warning = (
            abs(avg_loss - collapse_loss_ref) < 0.03
            and avg_teacher_entropy > 0.95 * collapse_loss_ref
            and avg_embedding_std < 0.02
        )

        epoch_diagnostics = {
            "teacher_entropy": avg_teacher_entropy,
            "student_entropy": avg_student_entropy,
            "embedding_std": avg_embedding_std,
            "collapse_warning": collapse_warning,
        }

        is_best = avg_loss < manager.best_loss
        manager.record_loss(epoch, avg_loss, diagnostics=epoch_diagnostics)
        # Persist after each epoch so interrupted runs keep partial history.
        manager.save_loss_history(verbose=False)
        
        elapsed_total = (time.time() - start_time) / 60
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config.num_epochs} Complete at {time.strftime('%H:%M:%S')}")
        print(f"Average Loss: {avg_loss:.6f}")
        print(f"Graphs trained: {epoch_trained_graphs}/{total_graphs_in_epoch} ({valid_pct:.2f}% valid)")
        print(f"Invalid SMILES skipped: {epoch_invalid_graphs}")
        print(
            f"Diagnostics: teacher_entropy={avg_teacher_entropy:.4f}, "
            f"student_entropy={avg_student_entropy:.4f}, "
            f"embedding_std={avg_embedding_std:.4f}, "
            f"collapse={collapse_warning}"
        )
        if profile_timing and num_batches > 0:
            print("Timing summary (avg seconds/batch):")
            print(f"  batch_load: {epoch_timing_sums['batch_load'] / num_batches:.3f}")
            print(f"  collate_total: {epoch_timing_sums['collate_total'] / num_batches:.3f}")
            print(f"  filter_invalid: {epoch_timing_sums['filter_invalid'] / num_batches:.3f}")
            print(f"  augmentation: {epoch_timing_sums['augmentation'] / num_batches:.3f}")
            print(f"  normalize_flatten: {epoch_timing_sums['normalize_flatten'] / num_batches:.3f}")
            print(f"  batch_from_data_list: {epoch_timing_sums['batch_from_data_list'] / num_batches:.3f}")
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
            online_eval_result = online_evaluator.evaluate_model(dino_ssl.student, torch.device(config.device))
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
        manager.save_checkpoint(epoch, dino_ssl.student, optimizer, avg_loss, is_best=is_best)
    
    # Save final results
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

"""Training script using your existing dataloader setup with DINO SSL.

Supports modular training with configurable models saved to models/{model_name}/
"""

import torch
import torch.optim as optim
from model.gine_model import GINEModel
from model.dino_ssl import DINOGraphSSL, cosine_scheduler
from datahandling.dataloader_creation import DataLoaderCreator
from model.config import ModelConfig
from training.train_manager import TrainingManager
import time


def dino_train(config: ModelConfig):
    """
    Train GINE with DINO using ModelConfig for modular training.
    
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

    # Initialize training manager
    manager = TrainingManager(config)
    
    # Create dataloader - everything comes from config
    creator = DataLoaderCreator(config)
    train_loader = creator.create_dataloader_auto()
    print(f"✓ DataLoader created with {len(train_loader)} batches\n")
    
    # Initialize GINE student model
    student_model = GINEModel.from_config(config)
    
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

    for epoch in range(config.num_epochs):
        epoch_loss = 0
        num_batches = 0
        epoch_trained_graphs = 0
        
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
            loss = dino_ssl.train_step(batch, optimizer)
            train_step_time = time.time() - train_step_start
            
            epoch_loss += loss
            num_batches += 1
            iteration += 1
            
            total_batch_time = time.time() - batch_start_time
            
            # Print progress
            if batch_idx % 10 == 0:
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
                
                print(f"Epoch [{epoch+1}/{config.num_epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss:.4f} | "
                      f"Load: {batch_load_time:.3f}s | "
                      f"Train: {train_step_time:.3f}s | "
                      f"Total: {total_batch_time:.3f}s | "
                      f"GPU Mem: {gpu_mem_allocated:.2f}/{gpu_mem_reserved:.2f}GB | "
                      f"Elapsed Time: {(time.time() - start_time)/60:.2f} min | "
                      f"Graphs: {num_unique_graphs} "
                      f"(Global: {num_global}, Local: {num_local}) | "
                      f"LR: {current_lr:.6f}",
                      f"Teacher Momentum: {current_momentum:.4f}",
                      f"WD: {current_wd:.4f}",
                      f"Teacher Temp: {current_teacher_temp:.4f}")
            
            # Start timing for next batch load (at end of every iteration)
            batch_load_start = time.time()
        
        # Epoch summary
        if num_batches == 0:
            raise RuntimeError("No valid batches produced. Check dataset for invalid SMILES.")

        total_graphs_in_epoch = len(train_loader.dataset)
        epoch_invalid_graphs = total_graphs_in_epoch - epoch_trained_graphs
        valid_pct = (epoch_trained_graphs / total_graphs_in_epoch) * 100 if total_graphs_in_epoch > 0 else 0.0

        avg_loss = epoch_loss / num_batches
        is_best = avg_loss < manager.best_loss
        manager.record_loss(epoch, avg_loss)
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config.num_epochs} Summary: Avg Loss = {avg_loss:.6f}")
        print(f"Graphs trained this epoch: {epoch_trained_graphs}/{total_graphs_in_epoch} ({valid_pct:.2f}% valid)")
        print(f"Invalid SMILES skipped this epoch: {epoch_invalid_graphs}")
        print(f"{'='*70}\n")
        
        # Save checkpoints
        manager.save_checkpoint(epoch, dino_ssl.student, optimizer, avg_loss, is_best=is_best)
    
    # Save final results
    manager.save_loss_history()
    manager.save_model_metadata()
    manager.save_dino_metadata()
    
    return dino_ssl, manager

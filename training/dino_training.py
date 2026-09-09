"""Training script using your existing dataloader setup with DINO SSL.

Supports modular training with configurable models saved to models/{model_name}/
"""

import datetime
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

# NCCL's default collective timeout (10 min in this environment) is sized for
# tightly-synchronized collectives, not for "rank 0 goes off and does
# validation + online eval + checkpoint I/O alone while other ranks wait at
# the next barrier." At production scale (large validation sets, big
# checkpoints written to network storage) that solo work can legitimately
# take longer than 10 minutes, which previously killed the whole job with a
# spurious watchdog timeout rather than an actual hang. Generous but bounded.
DDP_COLLECTIVE_TIMEOUT = datetime.timedelta(minutes=60)


def _init_ddp(config) -> tuple[int, int, int, bool]:
    """Init torch.distributed if launched under torchrun (RANK/WORLD_SIZE/LOCAL_RANK
    env vars set), otherwise a no-op. Returns (rank, local_rank, world_size, is_ddp)."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return 0, 0, 1, False

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl", timeout=DDP_COLLECTIVE_TIMEOUT)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, True


def _ddp_sum(value: float, device) -> float:
    """Sum a scalar across ranks. Each rank's train/val loader only sees its
    own shard of the data, so per-epoch loss/diagnostic sums are local to
    that rank until reduced -- without this, rank 0's logged loss curve
    would silently reflect ~1/world_size of the data instead of the whole
    epoch, and 'best checkpoint' selection would be comparing noise."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item())


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

    # Node-reconstruction loss weight: same warmup-then-hold shape as
    # teacher_temp above, ramping 0 -> node_reconstruction_loss_weight
    # instead of decaying. Only built when the loss is actually in use;
    # None otherwise so non-node-recon models carry no trace of this.
    node_recon_weight_schedule = None
    node_recon_warmup_epochs = 0
    if bool(getattr(config, "use_node_reconstruction_loss", False)):
        node_recon_warmup_epochs = max(0, min(int(getattr(config, "node_reconstruction_loss_warmup_epochs", 0)), int(config.num_epochs)))
        node_recon_warmup_iters = int(node_recon_warmup_epochs * train_loader_len)
        if node_recon_warmup_iters > 0:
            node_recon_warmup = torch.linspace(0.0, float(config.node_reconstruction_loss_weight), node_recon_warmup_iters)
        else:
            node_recon_warmup = torch.tensor([], dtype=torch.float32)
        node_recon_remain_iters = max(total_iters - node_recon_warmup_iters, 0)
        node_recon_hold = torch.full((node_recon_remain_iters,), float(config.node_reconstruction_loss_weight), dtype=torch.float32)
        node_recon_weight_schedule = torch.cat((node_recon_warmup.float(), node_recon_hold))

    return {
        "meta": {
            "num_epochs": int(config.num_epochs),
            "niter_per_ep": int(train_loader_len),
            "warmup_epochs": int(min(max(int(config.warmup_epochs), 0), int(config.num_epochs))),
            "teacher_temp_warmup_epochs": int(warmup_epochs),
            "node_reconstruction_loss_warmup_epochs": int(node_recon_warmup_epochs),
        },
        "lr_schedule": lr_schedule,
        "wd_schedule": wd_schedule,
        "momentum_schedule": momentum_schedule,
        "teacher_temp_schedule": teacher_temp_schedule,
        "node_recon_weight_schedule": node_recon_weight_schedule,
    }


def _load_schedule_bundle(schedule_path: Path) -> dict | None:
    if not schedule_path.exists():
        return None
    bundle = torch.load(schedule_path, map_location="cpu", weights_only=False)
    return bundle if isinstance(bundle, dict) else None


def _split_decay_params(model):
    """Split model parameters into (decay, no_decay) groups for the optimizer.

    Weight decay pulls every parameter it's applied to toward zero every
    step. For ordinary weight matrices/conv kernels that's a mild, healthy
    regularizer. But BatchNorm's gamma directly rescales an
    already-unit-variance normalized signal (output = gamma * normalized +
    beta) -- decaying it isn't "prefer a simpler weight", it's "turn this
    layer's output down toward silence". Confirmed empirically on the KHOP
    production run: BatchNorm gamma's RMS dropped from ~0.97 (near its
    default init of 1.0) to ~0.05 over 200 epochs, which alone is enough to
    explain the encoder_embedding_std collapse seen in that run. Biases are
    excluded for the same reason they are in most standard recipes (BERT,
    ViT, DINO's own reference implementation): they're additive offsets, not
    weights whose magnitude should be pushed toward zero.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "batch_norms" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def _prepare_schedule_bundle(config: ModelConfig, train_loader_len: int, schedule_path: Path) -> dict:
    bundle = _load_schedule_bundle(schedule_path)

    if bundle is None:
        bundle = _build_schedule_bundle(config, train_loader_len)
        torch.save(bundle, schedule_path)
        print(f"✓ Schedule state saved: {schedule_path}")
        return bundle

    loaded_niter = int(bundle.get("meta", {}).get("niter_per_ep", train_loader_len))
    loaded_num_epochs = int(bundle.get("meta", {}).get("num_epochs", 0))

    if loaded_niter != train_loader_len or loaded_num_epochs != int(config.num_epochs):
        # A segmented/resumed run (continue_training.py --target-epochs) changes
        # config.num_epochs between segments. The schedules are cosine curves
        # shaped by num_epochs, so extending the target requires rebuilding
        # them fresh over the new total -- previously this branch instead
        # padded the old (shorter-horizon) schedule by repeating its last
        # value for every remaining epoch. That silently froze LR at its
        # old-target minimum, weight decay at its old-target maximum, and
        # teacher EMA momentum at ~1.0 (i.e. the teacher stopped updating
        # entirely) for the rest of training, for every run that used
        # segment chaining. Only epochs from here forward are ever read from
        # this array, so rebuilding the whole curve (which retroactively
        # changes values for already-completed epochs that are never
        # looked up again) is safe.
        reason = "batch count" if loaded_niter != train_loader_len else "num_epochs target"
        print(
            f"Warning: loaded schedule state was built with a different {reason}; "
            "rebuilding schedules for the current run."
        )
        bundle = _build_schedule_bundle(config, train_loader_len)
        torch.save(bundle, schedule_path)
        print(f"✓ Schedule state rebuilt and saved: {schedule_path}")

    return bundle


def _resolve_resume_checkpoint_pair(resume_checkpoint_path: str) -> tuple[Path, Path | None]:
    """Resolve the student checkpoint and its matching teacher checkpoint, if available."""
    student_path = Path(resume_checkpoint_path)
    checkpoint_name = student_path.name

    teacher_name: str | None = None
    if checkpoint_name.startswith("t_"):
        teacher_name = checkpoint_name
        student_name = checkpoint_name[2:]
        student_path = student_path.with_name(student_name)
    elif checkpoint_name.startswith("s_"):
        student_name = checkpoint_name[2:]
        student_path = student_path.with_name(student_name)
        teacher_name = f"t_{student_name}"
    else:
        teacher_name = f"t_{checkpoint_name}"

    teacher_path = student_path.with_name(teacher_name) if teacher_name is not None else None
    if teacher_path is not None and not teacher_path.exists():
        teacher_path = None

    return student_path, teacher_path


def dino_train(config: ModelConfig, resume_checkpoint_path: str | None = None):
    """
    Train GNN with DINO using ModelConfig for modular training.
    
    Args:
        config: ModelConfig object with all model and training parameters
        
    Returns:
        Tuple: (dino_ssl, manager)
    """
    rank, local_rank, world_size, is_ddp = _init_ddp(config)
    is_main_process = not is_ddp or rank == 0
    if is_ddp:
        config.device = f"cuda:{local_rank}"
        config.ddp_rank = rank
        config.ddp_world_size = world_size
        config.ddp_local_rank = local_rank
        config.use_ddp = True

    if is_main_process:
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
    if is_main_process:
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
    use_data_parallel = bool(getattr(config, "use_data_parallel", False))
    if is_main_process:
        print(f"  Learning rate (used): {scaled_learning_rate}")
        available_cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if is_ddp:
            print(f"  Parallel training: DistributedDataParallel across {world_size} processes/GPUs")
        elif use_data_parallel and available_cuda_devices > 1 and str(config.device).startswith("cuda"):
            print(f"  Parallel training: requested with DataParallel on {available_cuda_devices} CUDA devices")
        elif use_data_parallel:
            print("  Parallel training: requested, but falling back to a single device")
    profile_timing = bool(getattr(config, "profile_timing", False))
    loader_debug = bool(getattr(config, "loader_debug", False))
    profile_every = int(getattr(config, "profile_log_every_n_batches", 50))
    if profile_timing and is_main_process:
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

    if is_main_process:
        if online_eval_enabled:
            print("  Online eval: enabled")
            print(f"    Datasets: {','.join(online_eval_datasets)}")
            print(f"    Every N epochs: {online_eval_every_n_epochs}")
            print(f"    Fixed k: {online_eval_fixed_k}")
            print("    Mode: kNN-only")
            print(f"    Top-k checkpoints kept: {online_eval_top_k}")
        else:
            print("  Online eval: disabled")

    # Initialize training manager. Only rank 0 owns it -- TrainingManager's
    # constructor writes config.json/metadata.json unconditionally, which
    # would race if every DDP process constructed one.
    model_dir = os.path.join("./models", config.name)
    manager = TrainingManager(config) if is_main_process else None
    if is_ddp:
        torch.distributed.barrier()

    # If resuming, restore existing loss/eval history so new epochs append cleanly.
    # Only rank 0 tracks history/checkpoints; other ranks have manager=None.
    history_path = os.path.join(model_dir, "loss_history.json")
    if is_main_process and resume_checkpoint_path and os.path.exists(history_path):
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

        # dino_loss_history/online_eval_history are restored above, but the
        # top-k checkpoint index is a separate file and was never reloaded --
        # without this, a resumed run starts tracking "best" from an empty
        # list, can end up pointing best_online_eval_model.pth at a checkpoint
        # worse than one from before the resume, and pruning may delete files
        # this run didn't know were worth keeping.
        top_eval_index_path = os.path.join(manager.model_dir, "top_eval_checkpoints.json")
        if os.path.exists(top_eval_index_path):
            try:
                with open(top_eval_index_path, "r") as f:
                    top_eval_payload = json.load(f)
                restored_top_k = top_eval_payload.get("top_eval_checkpoints", [])
                if isinstance(restored_top_k, list):
                    manager.top_eval_checkpoints = restored_top_k
            except Exception as exc:
                print(f"Warning: failed to restore top_eval_checkpoints.json ({exc}). Continuing with fresh top-k tracking.")

    # Online eval hits disk (checkpoint writes via update_top_eval_checkpoints)
    # and duplicating the compute across ranks would desync their iteration
    # timing for no benefit, so it stays rank-0-only.
    online_evaluator = None
    if online_eval_enabled and is_main_process:
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
    if is_main_process:
        print(f"✓ DataLoader created with {len(train_loader)} batches\n")
        if val_loader is not None:
            print(f"✓ Validation DataLoader created with {len(val_loader)} batches\n")

    # Initialize GNN student model
    student_model = GNNModel.from_config(config)

    num_params = sum(p.numel() for p in student_model.parameters())
    if is_main_process:
        print(f"✓ Model parameters: {num_params:,}\n")

    # Initialize DINO SSL framework
    dino_ssl = DINOGraphSSL.from_config(
        student_model=student_model,
        config=config,
        teacher_model=None,  # Will be created as copy
    )
    if is_main_process:
        if dino_ssl.use_ddp:
            print(f"✓ Parallel training active: DistributedDataParallel on {world_size} GPUs (SyncBatchNorm enabled)")
        elif use_data_parallel:
            if dino_ssl.use_data_parallel:
                print("✓ Parallel training active: graph-aware DataParallel enabled")
            else:
                print("✓ Parallel training inactive: using a single device")
    student_checkpoint_model = dino_ssl.student_model

    # Optimizer. Split into decay/no-decay groups -- see _split_decay_params
    # docstring for why BatchNorm gamma/beta and biases must not get weight
    # decay. apply_wd_schedule marks which group the per-iteration wd_schedule
    # update below is allowed to touch; the no-decay group's weight_decay
    # stays pinned at 0 for the whole run.
    decay_params, no_decay_params = _split_decay_params(dino_ssl.student)
    if is_main_process:
        print(
            f"  Optimizer param groups: {sum(p.numel() for p in decay_params):,} params with weight decay, "
            f"{sum(p.numel() for p in no_decay_params):,} params (biases/BatchNorm gamma+beta) without"
        )
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay_start, "apply_wd_schedule": True},
            {"params": no_decay_params, "weight_decay": 0.0, "apply_wd_schedule": False},
        ],
        lr=scaled_learning_rate,
    )

    # Schedule length must match across ranks (ShardAwareBatchSampler.__len__
    # returns the same ceil(total_batches / world_size) estimate on every
    # rank regardless of which specific shards it lands), so only rank 0
    # needs to build+save it -- other ranks just wait then load the file.
    schedule_path = Path(model_dir) / SCHEDULE_STATE_FILENAME
    if is_main_process:
        schedule_bundle = _prepare_schedule_bundle(config, len(train_loader), schedule_path)
    if is_ddp:
        torch.distributed.barrier()
    if not is_main_process:
        schedule_bundle = _load_schedule_bundle(schedule_path)
    lr_schedule = schedule_bundle["lr_schedule"]
    wd_schedule = schedule_bundle["wd_schedule"]
    momentum_schedule = schedule_bundle["momentum_schedule"]
    teacher_temp_schedule = schedule_bundle["teacher_temp_schedule"]
    node_recon_weight_schedule = schedule_bundle.get("node_recon_weight_schedule")

    if is_main_process:
        print("Starting training...\n")
    
    # Training loop
    start_epoch = 0
    # niter_per_ep is the schedule's assumed batches/epoch (same rank-invariant
    # estimate used to build the schedule -- see ShardAwareBatchSampler.__len__).
    # Per-batch schedule index is derived as epoch*niter_per_ep + batch_idx
    # (clamped) rather than a free-running counter, so that under DDP a rank
    # that happens to yield a couple more/fewer batches than the estimate in
    # a given epoch can't drift its LR/momentum/temperature schedule out of
    # sync with the other ranks over the course of training.
    niter_per_ep = len(train_loader)

    if resume_checkpoint_path:
        student_resume_path, teacher_resume_path = _resolve_resume_checkpoint_pair(resume_checkpoint_path)
        checkpoint = torch.load(student_resume_path, map_location=torch.device(config.device), weights_only=False)
        student_checkpoint_model.load_state_dict(checkpoint["model_state_dict"])

        if teacher_resume_path is not None:
            teacher_checkpoint = torch.load(teacher_resume_path, map_location=torch.device(config.device), weights_only=False)
            dino_ssl.teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
            if is_main_process:
                print(f"✓ Resuming teacher from checkpoint: {teacher_resume_path}")
        else:
            dino_ssl.teacher.load_state_dict(checkpoint["model_state_dict"])
            if is_main_process:
                print("Warning: matching teacher checkpoint not found; teacher restored from student weights.")

        # Note: this will raise if resuming a checkpoint saved before the
        # decay/no-decay param-group split was added above (optimizer had 1
        # group then, has 2 now) -- such a checkpoint needs a fresh optimizer
        # rather than a resumed one, since the saved state doesn't map cleanly.
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if is_main_process:
            print(f"✓ Resumed from checkpoint epoch {start_epoch} (next epoch to run)")
        if start_epoch >= config.num_epochs:
            raise ValueError(
                f"Checkpoint epoch {start_epoch} is >= target num_epochs ({config.num_epochs}). "
                "Increase num_epochs to continue training."
            )

    # Training time tracking
    start_time = time.time()

    collapse_loss_ref = math.log(float(config.projection_output_dim))
    batch_progress_enabled = (profile_timing or loader_debug) and is_main_process
    batch_progress_every = max(1, profile_every)

    for epoch in range(start_epoch, config.num_epochs):
        # Shard-aware batch sampler needs the epoch number to vary its shuffle
        # (mirrors torch's DistributedSampler.set_epoch convention). No-op for
        # dataloaders using the default PyTorch batch sampler.
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        if val_loader is not None and hasattr(val_loader.batch_sampler, "set_epoch"):
            val_loader.batch_sampler.set_epoch(epoch)

        if is_main_process:
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
        use_node_reconstruction_loss = bool(getattr(config, "use_node_reconstruction_loss", False))
        node_recon_warmup_epochs = int(schedule_bundle["meta"]["node_reconstruction_loss_warmup_epochs"]) if use_node_reconstruction_loss else 0
        if use_node_reconstruction_loss:
            # Kept out of the always-present dict above so every model
            # trained without this flag gets a completely unchanged
            # loss_history.json schema.
            epoch_diag_sums["graph_dino_loss"] = 0.0
            epoch_diag_sums["node_recon_loss"] = 0.0
            epoch_node_recon_batches = 0  # batches that actually had >=1 masked node -- averaging node_recon_loss over ALL batches would understate it whenever a batch had none
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

            iteration = min(epoch * niter_per_ep + batch_idx, len(lr_schedule) - 1)

            num_unique_graphs = len(torch.unique(batch['graph_idx']))
            epoch_trained_graphs += num_unique_graphs
            
            # Update learning rate (all groups) and weight decay (decay group
            # only -- the no-decay group's weight_decay must stay pinned at 0,
            # see _split_decay_params).
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[iteration]
                if param_group.get('apply_wd_schedule', True):
                    param_group['weight_decay'] = wd_schedule[iteration]
            
            # Update teacher momentum
            dino_ssl.teacher_momentum = momentum_schedule[iteration]

            # Update teacher temperature (loss-side teacher softmax temperature)
            dino_ssl.loss_fn.teacher_temp = float(teacher_temp_schedule[iteration])
            if getattr(dino_ssl, "use_node_reconstruction_loss", False):
                # Shares the graph loss's own schedule rather than a second
                # one to tune (see DINOGraphSSL.__init__) -- without this the
                # node loss's teacher_temp would stay frozen at its initial
                # construction value for the whole run instead of following
                # the same warmup/anneal curve as the graph loss.
                dino_ssl.node_loss_fn.teacher_temp = float(teacher_temp_schedule[iteration])
                if node_recon_weight_schedule is not None:
                    # train_step/eval_step both read this attribute live every
                    # call (see model/dino_ssl.py), so updating it here is all
                    # that's needed for the warmup to actually take effect --
                    # no change to dino_ssl.py itself required.
                    dino_ssl.node_reconstruction_loss_weight = float(node_recon_weight_schedule[iteration])

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
            for key in epoch_diag_sums:
                value = step_metrics.get(key, 0.0)
                epoch_diag_sums[key] += float(value) if value is not None else 0.0
            if use_node_reconstruction_loss and step_metrics.get("node_recon_loss") is not None:
                epoch_node_recon_batches += 1
            for key, value in step_timing.items():
                epoch_timing_sums[key] += float(value)
            for key, value in collate_timing.items():
                epoch_timing_sums[key] += float(value)
            epoch_timing_sums["batch_load"] += float(batch_load_time)
            
            total_batch_time = time.time() - batch_start_time
            epoch_timing_sums["batch_total"] += float(total_batch_time)
            if collate_timing:
                epoch_timing_sums["loader_wait_minus_collate"] += float(max(batch_load_time - collate_timing.get("collate_total", 0.0), 0.0))
            
            if batch_progress_enabled and (
                batch_idx == 0
                or (batch_idx + 1) % batch_progress_every == 0
                or (batch_idx + 1) == len(train_loader)
            ):
                print(
                    f"[Epoch {epoch+1}/{config.num_epochs}] batch {batch_idx+1}/{len(train_loader)} "
                    f"graphs={num_unique_graphs} load={batch_load_time:.2f}s step={train_step_time:.2f}s "
                    f"loss={loss:.6f}",
                    flush=True,
                )
            
            # Start timing for next batch load (at end of every iteration)
            batch_load_start = time.time()
        
        # Epoch summary
        if num_batches == 0:
            raise RuntimeError("No valid batches produced. Check dataset for invalid SMILES.")

        # Each rank only trained on its own shard of the data (see
        # ShardAwareBatchSampler's DDP partitioning); reduce across ranks so
        # the logged loss/diagnostics reflect the full epoch, not one rank's
        # slice. Done BEFORE validation deliberately: this is a fast,
        # all-rank collective right after the training loop, while every
        # rank is already roughly synchronized (batch counts are kept in
        # lockstep by the sampler's padding). Rank 0's validation/online-eval/
        # checkpoint work below is comparatively slow and rank-0-only -- if
        # this reduction were placed after that block (as it originally was),
        # ranks 1..N-1 would sit idle inside all_reduce for however long
        # rank 0's solo work takes, which can exceed the process group's
        # collective timeout at production scale and kill the whole job.
        if is_ddp:
            epoch_loss = _ddp_sum(epoch_loss, config.device)
            num_batches = int(_ddp_sum(num_batches, config.device))
            epoch_trained_graphs = int(_ddp_sum(epoch_trained_graphs, config.device))
            for key in epoch_diag_sums:
                epoch_diag_sums[key] = _ddp_sum(epoch_diag_sums[key], config.device)
            if use_node_reconstruction_loss:
                epoch_node_recon_batches = int(_ddp_sum(epoch_node_recon_batches, config.device))

        val_loss = None
        val_num_batches = 0
        val_diag_sums = {
            "teacher_entropy": 0.0,
            "student_entropy": 0.0,
            "embedding_std": 0.0,
            "encoder_embedding_std": 0.0,
        }
        if use_node_reconstruction_loss:
            val_diag_sums["graph_dino_loss"] = 0.0
            val_diag_sums["node_recon_loss"] = 0.0
            val_node_recon_batches = 0
        # val_loader is partitioned across DDP ranks the same way the train
        # loader is (see DataLoaderCreator's partition_across_ranks docstring):
        # eval_step bypasses the DDP wrapper entirely, so there's no per-batch
        # collective for a rank-count mismatch to desync -- every rank runs
        # its own slice in parallel, then a single _ddp_sum reduction combines
        # them below. This is what lets validation actually benefit from
        # multi-GPU parallelism instead of running single-rank while every
        # other GPU sits idle.
        if val_loader is not None:
            val_loss_total = 0.0
            for val_batch in val_loader:
                if val_batch is None:
                    continue
                val_step = dino_ssl.eval_step(val_batch)
                val_loss_total += float(val_step["loss"])
                val_num_batches += 1
                for key in val_diag_sums:
                    value = val_step["metrics"].get(key, 0.0)
                    val_diag_sums[key] += float(value) if value is not None else 0.0
                if use_node_reconstruction_loss and val_step["metrics"].get("node_recon_loss") is not None:
                    val_node_recon_batches += 1
            if is_ddp:
                val_loss_total = _ddp_sum(val_loss_total, config.device)
                val_num_batches = int(_ddp_sum(val_num_batches, config.device))
                for key in val_diag_sums:
                    val_diag_sums[key] = _ddp_sum(val_diag_sums[key], config.device)
                if use_node_reconstruction_loss:
                    val_node_recon_batches = int(_ddp_sum(val_node_recon_batches, config.device))
            if val_num_batches > 0:
                val_loss = val_loss_total / val_num_batches
            elif is_main_process:
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
        if use_node_reconstruction_loss:
            epoch_diagnostics["graph_dino_loss"] = epoch_diag_sums["graph_dino_loss"] / num_batches
            epoch_diagnostics["node_recon_loss"] = (
                epoch_diag_sums["node_recon_loss"] / epoch_node_recon_batches if epoch_node_recon_batches > 0 else None
            )

        val_diagnostics = None
        if val_num_batches > 0:
            val_diagnostics = {
                "teacher_entropy": val_teacher_entropy,
                "student_entropy": val_student_entropy,
                "embedding_std": val_embedding_std,
                "encoder_embedding_std": val_encoder_embedding_std,
            }
            if use_node_reconstruction_loss:
                val_diagnostics["graph_dino_loss"] = val_diag_sums["graph_dino_loss"] / val_num_batches
                val_diagnostics["node_recon_loss"] = (
                    val_diag_sums["node_recon_loss"] / val_node_recon_batches if val_node_recon_batches > 0 else None
                )

        if is_main_process:
            best_metric = val_loss if val_loss is not None else avg_loss

            if use_node_reconstruction_loss and node_recon_warmup_epochs > 0 and (epoch + 1) <= node_recon_warmup_epochs:
                # The node-reconstruction loss weight is still ramping up (see
                # node_recon_weight_schedule / schedule_bundle["meta"] above) -- the total
                # loss is structurally lower during warmup than it will be once the
                # auxiliary term reaches full weight, so comparing it against post-warmup
                # epochs (or letting it become the bar every later epoch is compared
                # against) is an apples-to-oranges comparison. best_model.pth tracking is
                # disabled entirely until warmup completes, rather than risk permanently
                # locking in a checkpoint from before the auxiliary loss ever took effect
                # (this exact failure mode was observed in NODE_RECON_WARMUP_TEST_60EP and
                # NODE_RECON_EXTFEAT_WARMUP_TEST_60EP's best_model.pth, both selected at
                # epoch 8-10 of a 15-epoch warmup).
                is_best = False
            elif use_node_reconstruction_loss and node_recon_warmup_epochs > 0:
                # Past warmup: compare only against the best value seen AFTER warmup
                # completed too. manager.best_val_loss/best_loss are a raw min() over the
                # WHOLE history (TrainingManager.record_loss recomputes them that way
                # unconditionally), which would otherwise let an artificially-low
                # pre-warmup epoch permanently block every later, fairly-comparable epoch
                # from ever being selected as best.
                metric_key = "val_loss" if val_loss is not None else "train_loss"
                post_warmup_values = [
                    r[metric_key] for r in manager.dino_loss_history
                    if r.get(metric_key) is not None and r.get("epoch", 0) > node_recon_warmup_epochs
                ]
                comparison_best = min(post_warmup_values) if post_warmup_values else float("inf")
                is_best = best_metric < comparison_best
            else:
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
            if use_node_reconstruction_loss:
                node_recon_str = f"{epoch_diagnostics['node_recon_loss']:.4f}" if epoch_diagnostics["node_recon_loss"] is not None else "n/a (no masked nodes this epoch)"
                print(f"Node reconstruction: graph_dino_loss={epoch_diagnostics['graph_dino_loss']:.4f}, node_recon_loss={node_recon_str}")
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

        if is_main_process:
            saved_online_path = None
            if online_evaluator is not None and (epoch + 1) % max(1, online_eval_every_n_epochs) == 0:
                print(f"  • Running online downstream eval (fixed k={online_eval_fixed_k})...")
                online_eval_result = online_evaluator.evaluate_model(
                    student_checkpoint_model,
                    torch.device(config.device),
                    explicit_hydrogens=bool(getattr(config, "explicit_hydrogens", True)),
                    encode_hydrogen_count=bool(getattr(config, "encode_hydrogen_count", False)),
                    use_extended_features=bool(getattr(config, "use_extended_features", False)),
                    scale_eccentricity=bool(getattr(config, "scale_eccentricity", False)),
                )
                teacher_online_eval_result = online_evaluator.evaluate_model(
                    dino_ssl.teacher,
                    torch.device(config.device),
                    explicit_hydrogens=bool(getattr(config, "explicit_hydrogens", True)),
                    encode_hydrogen_count=bool(getattr(config, "encode_hydrogen_count", False)),
                    use_extended_features=bool(getattr(config, "use_extended_features", False)),
                    scale_eccentricity=bool(getattr(config, "scale_eccentricity", False)),
                )
                aggregate_score = online_eval_result.get("aggregate_primary_score", float("-inf"))
                teacher_aggregate_score = teacher_online_eval_result.get("aggregate_primary_score", float("-inf"))
                print(f"    ✓ Student online eval done | aggregate validation score={aggregate_score:.6f}")
                print(f"    ✓ Teacher online eval done | aggregate validation score={teacher_aggregate_score:.6f}")

                saved_online_path = manager.update_top_eval_checkpoints(
                    epoch=epoch,
                    model=student_checkpoint_model,
                    optimizer=optimizer,
                    ssl_loss=avg_loss,
                    eval_result=online_eval_result,
                    top_k=online_eval_top_k,
                    teacher_model=dino_ssl.teacher,
                )
                manager.record_online_eval(
                    epoch=epoch,
                    ssl_loss=avg_loss,
                    eval_result=online_eval_result,
                    saved_path=saved_online_path,
                    teacher_eval_result=teacher_online_eval_result,
                    teacher_saved_path=(
                        None if saved_online_path is None else os.path.join(
                            os.path.dirname(saved_online_path),
                            f"t_{os.path.basename(saved_online_path)}",
                        )
                    ),
                )

                if saved_online_path is not None:
                    print(f"    ✓ Top-{online_eval_top_k} checkpoint updated: {saved_online_path}")
                else:
                    print(f"    • Checkpoint not in top-{online_eval_top_k}; not saved")

                manager.save_loss_history(verbose=False)

            # Save checkpoints
            manager.save_checkpoint(
                epoch,
                student_checkpoint_model,
                optimizer,
                avg_loss,
                is_best=is_best,
                metric_value=val_loss,
                teacher_model=dino_ssl.teacher,
            )

        if is_ddp:
            # Rank 0 is doing extra disk I/O (checkpoints/online eval) other
            # ranks don't do; keep every rank's epoch loop in step so the
            # next epoch's set_epoch()/shard shuffle stays synchronized.
            torch.distributed.barrier()

    # Save final results (rank 0 only -- see TrainingManager guard above)
    if is_main_process:
        # Always save a final checkpoint for reproducibility
        if manager.dino_loss_history:
            last_epoch = len(manager.dino_loss_history) - 1
            last_loss = manager.dino_loss_history[-1].get("train_loss")
            try:
                manager.save_final_checkpoint(
                    last_epoch,
                    student_checkpoint_model,
                    optimizer,
                    loss=last_loss,
                    teacher_model=dino_ssl.teacher,
                )
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

    if is_ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    return dino_ssl, manager

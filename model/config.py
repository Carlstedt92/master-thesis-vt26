"""Configuration for model training."""

from dataclasses import dataclass, asdict, fields
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """Configuration for GINE model training."""
    
    # Model identifier
    name: str                           # e.g., "dino_gine_5layer", "dino_gine_3layer"
    head_type: str = "dino"             # Head type: "dino", "classification", etc.
    data_path: str = "data/delaney-processed.csv"  # Path to dataset CSV file
    use_precomputed: bool = False       # If true, load precomputed graph shards instead of parsing SMILES online
    precomputed_data_path: str = ""     # Directory containing shard_*.pt and optional metadata.json
    cache_data_in_memory: bool = False   # Keep source rows/graphs in RAM after first read
    precomputed_cache_in_memory: bool = False  # Keep all precomputed graphs in RAM (per dataloader worker)
    explicit_hydrogens: bool = True      # Add hydrogen atoms as explicit graph nodes during SMILES parsing
    encode_hydrogen_count: bool = False  # Append total hydrogen count as an extra atom feature
    use_extended_features: bool = False  # Append Gasteiger partial charge + topological eccentricity (2 extra node feature dims). Must match what the precomputed shards / num_features were built with -- see datahandling/graph_creation.py.
    scale_eccentricity: bool = False    # Apply the fitted StandardScaler to eccentricity (see fit_eccentricity_scaler.py). Independent of use_extended_features: checkpoints trained before this scaler existed expect raw eccentricity, so must stay False for them -- only new checkpoints trained WITH scaling should set this True.
    precomputed_max_cached_shards: int = 4  # LRU cache size for shard files when precomputed_cache_in_memory=False
    loader_debug: bool = False           # Verbose logging for dataset/dataloader startup and shard cache behavior
    profile_timing: bool = False         # Emit detailed per-batch timing for loader/train-step components
    profile_log_every_n_batches: int = 50  # How often to print cumulative timing summaries
    use_batched_mask_collate: bool = False  # Fast-path collate for masking: batch first, then mask batched tensors
    seed: int = 42                      # Random seed for reproducibility
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # Device to train on (cuda or cpu)
    num_workers: int = 0                # Number of worker processes for data loading (0 = main process)
    local_views: int = 4                # Number of local augmented views per graph (default: 4)
    k_hops: int = 2                     # Number of hops for local subgraph extraction

    # Data dimensions
    num_features: int = 24              # Node feature dimension
    edge_features: int = 12             # Edge attribute dimension
    
    # Encoder architecture
    encoder_type: str = "GINE"          # Encoder type: "GINE", "GAT"
    global_pooling: str | None = None   # Global pooling: "add", "mean", or None for encoder default
    hidden_dim: int = 128               # Hidden dimension for GINE layers
    num_layers: int = 5                 # Number of GINE convolutional layers
    gat_heads: int = 1                  # GAT only: number of attention heads per layer, concatenated back to hidden_dim (so hidden_dim must be divisible by gat_heads). No effect on GINE.
    dropout: float = 0.0                # Dropout rate (0 for GIN paper - no dropout for molecule graphs)
    epsilon: float = 0.0                # GINE epsilon parameter

    # Augmentation mode
    local_augmentation_mode: str = "k_hop"   # "k_hop", "masking", "functional_group_masking", or "functional_group_k_hop"
    node_mask_ratio: float = 0.15             # Fraction of nodes to mask in masking mode
    feature_mask_ratio: float = 0.15          # Fraction of node features to mask in masking mode
    local_view_modes: Optional[List[str]] = None  # Optional per-local-view mode override (must have exactly local_views entries), e.g. ["functional_group_k_hop", "functional_group_masking"] to mix augmentation styles across views. None (default) uses local_augmentation_mode uniformly for every view, unchanged from before.

    # Masked-node reconstruction auxiliary loss (iBOT/DINOv2-style): alongside the
    # existing graph-level DINO loss, have the student predict the teacher's own
    # per-node embedding at each node masked by a masking-based local view. Only
    # has any effect on nodes that came from a "masking" or "functional_group_masking"
    # view (k_hop-only views have nothing masked to reconstruct) -- requires GAT
    # + head_type="dino" + encoder_type="GAT", and requires
    # use_batched_mask_collate=False whenever local_augmentation_mode=="masking"
    # (the fast batched-collate path bypasses GraphAugmentation entirely and never
    # produces the mask needed here -- functional_group_masking always goes through
    # GraphAugmentation regardless of that flag, so it's unaffected). False by
    # default -- fully backward compatible, every existing config/checkpoint is
    # unaffected.
    use_node_reconstruction_loss: bool = False
    node_reconstruction_loss_weight: float = 1.0  # Target weight, reached at the END of the warmup below (or from step 1 if warmup_epochs=0).
    node_reconstruction_loss_warmup_epochs: int = 0  # Linearly ramp the node-loss weight 0 -> node_reconstruction_loss_weight over this many epochs, then hold flat -- 0 (default) means full weight from epoch 1, unchanged from the first run of this feature (NODE_RECON_TEST_60EP). A randomly-initialized mask_token/node_head has nothing useful to teach the encoder on step 1; that run's downstream eval came back worse than the no-node-loss baseline on the two apples-to-apples datasets (LIPO, Tox21) while the node loss itself never settled cleanly -- a warmup lets the graph-level objective get a head start before the noisier auxiliary one phases in, mirroring teacher_temp_warmup_epochs's own warmup-then-hold shape.

    # Projection head
    projection_hidden_dim: int = 256    # MLP hidden dimension (reduced for molecular graphs)
    projection_output_dim: int = 128    # Output embedding dimension
    projection_bottleneck_dim: int = 256  # Bottleneck dimension before final projection layer
    projection_layers: int = 2          # Number of projection head layers
    
    # Training parameters
    num_epochs: int = 100               # Number of training epochs
    batch_size: int = 32                # Batch size (number of graphs)
    auto_scale_lr: bool = False         # Apply linear LR scaling from effective batch size
    use_data_parallel: bool = False     # Wrap the student model in torch.nn.DataParallel when multiple CUDA devices are available
    # DDP fields below are set programmatically by dino_train()/_init_ddp() when launched
    # under torchrun (RANK/WORLD_SIZE/LOCAL_RANK env vars) -- not meant to be hand-set in
    # config JSON files. Kept on the dataclass so DataLoaderCreator/DINOGraphSSL can read
    # them off the same config object already threaded through the whole training call stack.
    use_ddp: bool = False                     # True once _init_ddp() has initialized the process group
    ddp_rank: int = 0                         # Global rank of this process
    ddp_world_size: int = 1                   # Total number of DDP processes
    ddp_local_rank: int = 0                   # GPU index on this node for this process
    lr_scale_base: float = 5e-4         # Base LR in the linear scaling rule
    lr_scale_reference_batch_size: int = 256  # Reference batch size for LR scaling rule
    learning_rate: float = 0.0005       # Initial learning rate
    weight_decay: float = 0.04          # Kept for backward compatibility
    weight_decay_start: float = 0.04    # Start value for cosine weight decay schedule
    weight_decay_end: float = 0.4       # End value for cosine weight decay schedule
    
    # DINO SSL parameters
    teacher_temp: float = 0.04          # Temperature for teacher softmax
    teacher_temp_final: float = 0.07    # Final teacher temperature after warmup
    teacher_temp_warmup_epochs: int = 30  # Epochs for linear teacher temp warmup
    student_temp: float = 0.1           # Temperature for student softmax
    teacher_momentum: float = 0.996     # EMA momentum for teacher network
    center_momentum: float = 0.9        # EMA momentum for loss center
    sync_teacher_batchnorm_buffers: bool = True  # Copy student BatchNorm running_mean/var/num_batches_tracked to teacher each step (these are buffers, not parameters, so the EMA loop over .parameters() never touches them). Set False to restore the old behavior where teacher buffers stay frozen at initialization.

    # Learning rate schedule
    warmup_epochs: int = 10             # Warmup epochs before cosine annealing
    final_learning_rate: float = 1e-6   # Final learning rate after schedule

    # SSL validation split
    validation_enabled: bool = True     # Hold out a validation split for SSL monitoring
    validation_split: float = 0.1       # Fraction of data reserved for validation

    # Regular checkpoint cadence (training/train_manager.py::save_checkpoint saves
    # checkpoint_epoch_{N}.pth whenever (epoch+1) % this == 0). 10 (default) matches
    # every prior run in the project -- unaffected unless a config sets this
    # explicitly. Runs that must be split across many wall-clock-limited SLURM jobs
    # (e.g. a 500-epoch run needing ~4-5 sequential 48h submissions) can lose up to
    # (this - 1) epochs of otherwise-completed training to a wall-clock kill landing
    # between saves -- lower this for a long/expensive run to bound that loss more
    # tightly, at the cost of more frequent checkpoint I/O and more disk used by
    # regular checkpoints.
    checkpoint_save_every_n_epochs: int = 10

    # Online downstream evaluation during SSL training
    online_eval_enabled: bool = False          # Run downstream eval during SSL training
    online_eval_every_n_epochs: int = 1        # Evaluate every N epochs (1 = every epoch)
    online_eval_datasets: str = "lipo"         # Comma-separated downstream datasets
    online_eval_fixed_k: int = 5               # Fixed k for kNN speed during training
    online_eval_top_k_checkpoints: int = 5     # Keep top-K checkpoints by eval score
    
    def to_dict(self):
        """Convert config to dictionary for saving."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary, ignoring unknown keys for compatibility."""
        valid_field_names = {field.name for field in fields(cls)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_field_names}
        return cls(**filtered_config)

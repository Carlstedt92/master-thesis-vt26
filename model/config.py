"""Configuration for model training."""

from dataclasses import dataclass, asdict, fields
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
    precomputed_max_cached_shards: int = 4  # LRU cache size for shard files when precomputed_cache_in_memory=False
    loader_debug: bool = False           # Verbose logging for dataset/dataloader startup and shard cache behavior
    profile_timing: bool = False         # Emit detailed per-batch timing for loader/train-step components
    profile_log_every_n_batches: int = 50  # How often to print cumulative timing summaries
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
    hidden_dim: int = 128               # Hidden dimension for GINE layers
    num_layers: int = 5                 # Number of GINE convolutional layers
    dropout: float = 0.0                # Dropout rate (0 for GIN paper - no dropout for molecule graphs)
    epsilon: float = 0.0                # GINE epsilon parameter

    # Augmentation mode
    local_augmentation_mode: str = "k_hop"   # "k_hop" or "masking"
    node_mask_ratio: float = 0.15             # Fraction of nodes to mask in masking mode
    feature_mask_ratio: float = 0.15          # Fraction of node features to mask in masking mode
    
    # Projection head
    projection_hidden_dim: int = 256    # MLP hidden dimension (reduced for molecular graphs)
    projection_output_dim: int = 128    # Output embedding dimension
    projection_layers: int = 2          # Number of projection head layers
    
    # Training parameters
    num_epochs: int = 100               # Number of training epochs
    batch_size: int = 32                # Batch size (number of graphs)
    auto_scale_lr: bool = False         # Apply linear LR scaling from effective batch size
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
    
    # Learning rate schedule
    warmup_epochs: int = 10             # Warmup epochs before cosine annealing
    final_learning_rate: float = 1e-6   # Final learning rate after schedule

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

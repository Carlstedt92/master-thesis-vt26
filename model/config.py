"""Configuration for model training."""

from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for GINE model training."""
    
    # Model identifier
    name: str                           # e.g., "dino_gine_5layer", "dino_gine_3layer"
    head_type: str = "dino"             # Head type: "dino", "classification", etc.
    
    # Data dimensions
    num_features: int = 21              # Node feature dimension
    edge_features: int = 6              # Edge attribute dimension
    
    # Encoder architecture
    hidden_dim: int = 128               # Hidden dimension for GINE layers
    num_layers: int = 5                 # Number of GINE convolutional layers
    dropout: float = 0.5                # Dropout rate
    epsilon: float = 0.0                # GINE epsilon parameter
    
    # Projection head
    projection_hidden_dim: int = 2048   # MLP hidden dimension
    projection_output_dim: int = 256    # Output embedding dimension
    projection_layers: int = 3          # Number of projection head layers
    
    # Training parameters
    num_epochs: int = 100               # Number of training epochs
    batch_size: int = 32                # Batch size (number of graphs)
    learning_rate: float = 0.0005       # Initial learning rate
    weight_decay: float = 0.04          # AdamW weight decay
    
    # DINO SSL parameters
    teacher_temp: float = 0.04          # Temperature for teacher softmax
    student_temp: float = 0.1           # Temperature for student softmax
    teacher_momentum: float = 0.996     # EMA momentum for teacher network
    center_momentum: float = 0.9        # EMA momentum for loss center
    
    # Learning rate schedule
    warmup_epochs: int = 10             # Warmup epochs before cosine annealing
    final_learning_rate: float = 1e-6   # Final learning rate after schedule
    
    def to_dict(self):
        """Convert config to dictionary for saving."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)

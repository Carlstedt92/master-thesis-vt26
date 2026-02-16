import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool


class GINEEncoder(nn.Module):
    """GINE Encoder for graph representation learning."""
    
    def __init__(self, num_features: int, edge_features: int, hidden_dim: int = 64, 
                 num_layers: int = 3, dropout: float = 0.5, epsilon: float = 0):
        super(GINEEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.node_encoder = nn.Linear(num_features, hidden_dim)
        
        # GINE convolutional layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            conv = GINEConv(mlp, eps=epsilon, edge_dim=edge_features, train_eps=True)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Convert to float for processing (node features are typically Long)
        x = x.float()
        if edge_attr is not None:
            edge_attr = edge_attr.float()
        
        # Encode node features
        x = self.node_encoder(x)
        
        # Apply GINE layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        graph_embedding = global_mean_pool(x, batch)
        
        return graph_embedding


class ProjectionHead(nn.Module):
    """Projection head for DINO SSL framework."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 256, 
                 num_layers: int = 3, bottleneck_dim: int = 256):
        super(ProjectionHead, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            current_dim = hidden_dim
        
        # Bottleneck layer
        layers.append(nn.Linear(current_dim, bottleneck_dim))
        layers.append(nn.GELU())
        
        # Output layer (L2 normalized)
        layers.append(nn.Linear(bottleneck_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.mlp(x)
        # L2 normalize for DINO stability
        return F.normalize(x, dim=-1, p=2)


class SSL_GINEModel(nn.Module):
    """Graph Isomorphism Network (GINE) for self-supervised learning.
    This model is designed for self-supervised learning using the DINO framework."""

    def __init__(self, num_features: int, edge_features: int, hidden_dim: int = 64, 
                 num_layers: int = 3, dropout: float = 0.5, epsilon: float = 0,
                 projection_hidden_dim: int = 2048, projection_output_dim: int = 256,
                 projection_layers: int = 3, head_type: str = "dino"):
        super(SSL_GINEModel, self).__init__()
        
        # GINE encoder backbone
        self.encoder = GINEEncoder(
            num_features=num_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            epsilon=epsilon
        )
        if head_type == "dino":
            # Projection head for DINO
            self.projection_head = ProjectionHead(
                input_dim=hidden_dim,
                hidden_dim=projection_hidden_dim,
                output_dim=projection_output_dim,
                num_layers=projection_layers
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")
    
    def forward(self, x, edge_index, edge_attr, batch):
        """Forward pass through encoder and projection head."""
        # Get graph embeddings
        embeddings = self.encoder(x, edge_index, edge_attr, batch)
        
        # Project to DINO space
        projections = self.projection_head(embeddings)
        
        return projections
    
    def get_embeddings(self, x, edge_index, edge_attr, batch):
        """Get graph embeddings without projection (for downstream tasks)."""
        return self.encoder(x, edge_index, edge_attr, batch)

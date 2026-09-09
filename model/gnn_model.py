import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, GATConv, global_mean_pool


class GINEEncoder(nn.Module):
    """GINE Encoder for graph representation learning."""
    
    def __init__(self, num_features: int, edge_features: int, hidden_dim: int = 64, 
                 num_layers: int = 3, dropout: float = 0.5, epsilon: float = 0,
                 global_pooling: str | None = None):
        super(GINEEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.global_pooling = global_pooling
        
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
            conv = GINEConv(mlp, eps=epsilon, edge_dim=edge_features, train_eps=False)
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
        if self.global_pooling == "add":
            graph_embedding = global_add_pool(x, batch)
        elif self.global_pooling in (None, "mean"):
            graph_embedding = global_mean_pool(x, batch)
        else:
            raise ValueError(f"Unsupported global pooling type for GINE: {self.global_pooling}")
        
        return graph_embedding

class GATEncoder(nn.Module):
    """GAT Encoder for graph representation learning."""

    def __init__(self, num_features: int, edge_features: int, hidden_dim: int = 64,
                 num_layers: int = 3, dropout: float = 0.5,
                 global_pooling: str | None = None, heads: int = 1):
        super(GATEncoder, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.global_pooling = global_pooling

        # Input projection
        self.node_encoder = nn.Linear(num_features, hidden_dim)

        # GAT convolutional layers. Multi-head attention concatenates each
        # head's output back to hidden_dim (standard GAT-paper / Transformer
        # style: same total width, split into `heads` independent attention
        # patterns instead of one) -- so hidden_dim must divide evenly by
        # heads, and each head computes hidden_dim // heads channels.
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}) "
                "for concatenated multi-head GAT attention."
            )
        head_dim = hidden_dim // heads
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = GATConv(hidden_dim, head_dim, heads=heads, concat=True, edge_dim=edge_features)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, edge_attr, batch, mask_token=None, node_mask=None, return_node_embeddings=False):
        """mask_token/node_mask/return_node_embeddings are all optional and default
        to producing the exact old behavior (a single graph_embedding tensor) --
        only used by the masked-node reconstruction auxiliary loss, gated behind
        config.use_node_reconstruction_loss. See GNNModel.forward."""
        # Convert to float for processing (node features are typically Long)
        x = x.float()
        if edge_attr is not None:
            edge_attr = edge_attr.float()

        # Encode node features
        x = self.node_encoder(x)

        # Substitute the learned mask token at masked node positions -- done
        # AFTER node_encoder (in hidden_dim space, not raw feature space) so
        # every GAT layer sees the same-shaped placeholder regardless of
        # num_features, and the message-passing layers below have to fill in
        # that position's representation from its unmasked neighbors alone.
        if node_mask is not None and mask_token is not None and node_mask.any():
            x = torch.where(node_mask.unsqueeze(-1), mask_token.to(x.dtype), x)

        # Apply GAT layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.global_pooling == "mean":
            graph_embedding = global_mean_pool(x, batch)
        elif self.global_pooling in (None, "add"):
            graph_embedding = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unsupported global pooling type for GAT: {self.global_pooling}")

        if return_node_embeddings:
            return graph_embedding, x
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
    
class RegressionHead(nn.Module):
    """Regression head for downstream tasks."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super(RegressionHead, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


class ClassificationHead(nn.Module):
    """Binary classification head for downstream tasks."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super(ClassificationHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # Return logits; loss should apply sigmoid internally (BCEWithLogitsLoss).
        return self.mlp(x)


class GNNModel(nn.Module):
    """GNN model with flexible encoder and head for both SSL training and downstream tasks.
    The model is designed to be flexible and can be used for both SSL training (with projection head) and downstream tasks (without projection head).
    """

    def __init__(self, num_features: int, edge_features: int, hidden_dim: int = 64,
                 num_layers: int = 3, dropout: float = 0.5, epsilon: float = 0,
                 projection_hidden_dim: int = 2048, projection_output_dim: int = 256,
                 projection_bottleneck_dim: int = 256, projection_layers: int = 3,
                 head_type: str = "dino", encoder: str = "GINE",
                 global_pooling: str | None = None, gat_heads: int = 1,
                 use_node_reconstruction_loss: bool = False):
        super(GNNModel, self).__init__()

        self.use_node_reconstruction_loss = bool(use_node_reconstruction_loss)
        if self.use_node_reconstruction_loss:
            if encoder != "GAT":
                raise ValueError("use_node_reconstruction_loss requires encoder='GAT' (GATEncoder.forward is the only one that accepts mask_token/node_mask).")
            if head_type != "dino":
                raise ValueError("use_node_reconstruction_loss only makes sense with head_type='dino' -- it's a second SSL pretext loss, not a downstream task head.")

        # GINE encoder backbone
        if encoder == "GINE":
            self.encoder = GINEEncoder(
                num_features=num_features,
                edge_features=edge_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                epsilon=epsilon,
                global_pooling=global_pooling,
            )
        elif encoder == "GAT":
            self.encoder = GATEncoder(
                num_features=num_features,
                edge_features=edge_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                global_pooling=global_pooling,
                heads=gat_heads,
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder}")
        # Head Type
        if head_type == "dino":
            # Projection head for DINO
            self.head = ProjectionHead(
                input_dim=hidden_dim,
                hidden_dim=projection_hidden_dim,
                output_dim=projection_output_dim,
                num_layers=projection_layers,
                bottleneck_dim=projection_bottleneck_dim,
            )
        elif head_type == "regression":
            # Regression head for downstream tasks
            self.head = RegressionHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=1
            )
        elif head_type == "classification":
            # Binary classification head for downstream tasks
            self.head = ClassificationHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=1,
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

        # Masked-node reconstruction auxiliary loss (iBOT/DINOv2-style, see
        # ModelConfig.use_node_reconstruction_loss). mask_token replaces a
        # masked node's projected features before message passing (standard
        # MAE/BEiT-style learned placeholder); node_head projects per-node
        # embeddings into their own DINO-style space, separate from the
        # graph-level head above -- mirrors how the graph head has its own
        # dedicated projection rather than reusing raw encoder output.
        if self.use_node_reconstruction_loss:
            self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
            nn.init.normal_(self.mask_token, std=0.02)
            self.node_head = ProjectionHead(
                input_dim=hidden_dim,
                hidden_dim=projection_hidden_dim,
                output_dim=projection_output_dim,
                num_layers=projection_layers,
                bottleneck_dim=projection_bottleneck_dim,
            )

    @classmethod
    def from_config(cls, config, head_type: str | None = None):
        """Build model from config, with optional head_type override."""
        resolved_head_type = config.head_type if head_type is None else head_type
        return cls(
            num_features=config.num_features,
            edge_features=config.edge_features,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            epsilon=config.epsilon,
            projection_hidden_dim=config.projection_hidden_dim,
            projection_output_dim=config.projection_output_dim,
            projection_bottleneck_dim=getattr(config, "projection_bottleneck_dim", 256),
            projection_layers=config.projection_layers,
            head_type=resolved_head_type,
            encoder=config.encoder_type if hasattr(config, "encoder_type") else "GINE",
            global_pooling=getattr(config, "global_pooling", None),
            gat_heads=getattr(config, "gat_heads", 1),
            use_node_reconstruction_loss=getattr(config, "use_node_reconstruction_loss", False),
        )
    
    def forward(self, x, edge_index, edge_attr, batch, node_reconstruction_mask=None):
        """Forward pass through encoder and projection head.

        node_reconstruction_mask is optional and only meaningful when this
        model was built with use_node_reconstruction_loss=True -- every
        existing call site (with 4 positional args, no 5th kwarg) gets
        exactly the old behavior and return shape (a single graph projection
        tensor) unchanged. When it IS passed on a model built with the flag
        on, this also runs the masked positions through node_head and
        returns (graph_projection, masked_node_projection) instead -- kept
        as a single forward() call (rather than two separate calls) so a
        DDP-wrapped student only needs one hooked forward pass per step, and
        so DDP's gradient sync -- which is tied to the actual forward()
        invocation -- correctly covers mask_token/node_head too.
        """
        if node_reconstruction_mask is not None:
            if not self.use_node_reconstruction_loss:
                raise ValueError("node_reconstruction_mask was passed but this model wasn't built with use_node_reconstruction_loss=True.")
            graph_embedding, node_embeddings = self.encoder(
                x, edge_index, edge_attr, batch,
                mask_token=self.mask_token,
                node_mask=node_reconstruction_mask,
                return_node_embeddings=True,
            )
            graph_projection = self.head(graph_embedding)
            masked_node_projection = self.node_head(node_embeddings[node_reconstruction_mask])
            return graph_projection, masked_node_projection

        # Get graph embeddings
        embeddings = self.encoder(x, edge_index, edge_attr, batch)

        return self.head(embeddings)

    def get_embeddings(self, x, edge_index, edge_attr, batch):
        """Get graph embeddings without projection (for downstream tasks)."""
        return self.encoder(x, edge_index, edge_attr, batch)

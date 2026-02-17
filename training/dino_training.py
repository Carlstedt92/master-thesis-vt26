"""Training script using your existing dataloader setup with DINO SSL.

Supports modular training with configurable models saved to models/{model_name}/
"""

import torch
import torch.optim as optim
from model.gine_model import GINEModel
from model.dino_ssl import DINOGraphSSL, cosine_scheduler
from datahandling.dataloader_creation import create_dataloader
from model.config import ModelConfig
from training.train_manager import TrainingManager
from typing import Union
from utils.seed import set_seed


def dino_train(config: ModelConfig, csv_path: str,
               device: Union[str, torch.device] = 'cuda',
               seed: int | None = 42):
    """
    Train GINE with DINO using ModelConfig for modular training.
    
    Args:
        config: ModelConfig object with all model and training parameters
        csv_path: Path to CSV file with SMILES
        device: Device to train on (cuda or cpu)
        
    Returns:
        Tuple: (dino_ssl, manager)
    """
    print("="*70)
    print("DINO SSL Training with Multi-Crop Augmentation")
    print("="*70)
    print(f"\nModel: {config.name}")
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Head type: {config.head_type}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size} graphs")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Layers: {config.num_layers}, Hidden dim: {config.hidden_dim}")
    print(f"  Views per graph: 2 global + 4 local = 6 total")
    print(f"  Effective batch size: {config.batch_size * 6} views")
    print()
    
    if seed is not None:
        set_seed(seed)

    # Initialize training manager
    manager = TrainingManager(config)
    
    # Create dataloader
    train_loader = create_dataloader(
        csv_path=csv_path,
        batch_size=config.batch_size,
        shuffle=True,
        seed=seed
    )
    print(f"✓ DataLoader created with {len(train_loader)} batches\n")
    
    # Initialize GINE student model
    student_model = GINEModel(
        num_features=config.num_features,
        edge_features=config.edge_features,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        epsilon=config.epsilon,
        projection_hidden_dim=config.projection_hidden_dim,
        projection_output_dim=config.projection_output_dim,
        projection_layers=config.projection_layers,
        head_type=config.head_type  # ← Pass head_type from config
    )
    
    num_params = sum(p.numel() for p in student_model.parameters())
    print(f"✓ Model parameters: {num_params:,}\n")
    
    # Initialize DINO SSL framework
    dino_ssl = DINOGraphSSL(
        student_model=student_model,
        teacher_model=None,  # Will be created as copy
        device=device,
        teacher_temp=config.teacher_temp,
        student_temp=config.student_temp,
        center_momentum=config.center_momentum,
        teacher_momentum=config.teacher_momentum
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        dino_ssl.student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    lr_schedule = cosine_scheduler(
        base_value=config.learning_rate,
        final_value=config.final_learning_rate,
        epochs=config.num_epochs,
        niter_per_ep=len(train_loader),
        warmup_epochs=config.warmup_epochs,
        start_warmup_value=0.0
    )
    
    # Teacher momentum scheduler
    momentum_schedule = cosine_scheduler(
        base_value=config.teacher_momentum,
        final_value=1.0,
        epochs=config.num_epochs,
        niter_per_ep=len(train_loader)
    )
    
    print("Starting training...\n")
    
    # Training loop
    iteration = 0
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[iteration]
            
            # Update teacher momentum
            dino_ssl.teacher_momentum = momentum_schedule[iteration]
            
            # Training step - batch contains all augmented views
            # Teacher will automatically filter for global views (view==1)
            # Student sees all views
            # Loss computed between matching graph_idx
            loss = dino_ssl.train_step(batch, optimizer)
            
            epoch_loss += loss
            num_batches += 1
            iteration += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                current_lr = lr_schedule[iteration-1]
                current_momentum = momentum_schedule[iteration-1]
                
                # Count views in batch for reporting
                num_global = (batch['view'] == 1).sum().item()
                num_local = (batch['view'] == 0).sum().item()
                num_unique_graphs = len(torch.unique(batch['graph_idx']))
                
                print(f"Epoch [{epoch+1}/{config.num_epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss:.4f} | "
                      f"Graphs: {num_unique_graphs} "
                      f"(Global: {num_global}, Local: {num_local}) | "
                      f"LR: {current_lr:.6f}",
                      f"Teacher Momentum: {current_momentum:.4f}")
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        is_best = avg_loss < manager.best_loss
        manager.record_loss(epoch, avg_loss)
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config.num_epochs} Summary: Avg Loss = {avg_loss:.6f}")
        print(f"{'='*70}\n")
        
        # Save checkpoints
        manager.save_checkpoint(epoch, dino_ssl.student, optimizer, avg_loss, is_best=is_best)
    
    # Save final results
    manager.save_loss_history()
    manager.save_metadata()
    
    return dino_ssl, manager


if __name__ == "__main__":
    import sys
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Default CSV path
    csv_path = "data/delaney-processed.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    # Create config (can be customized)
    config = ModelConfig(
        name="dino_gine_5layer",  # Model identifier
        num_epochs=100,
        batch_size=32,
        # ... other parameters use defaults from ModelConfig
    )
    
    # Override with command line args if provided
    if len(sys.argv) > 2:
        config.num_epochs = int(sys.argv[2])
    if len(sys.argv) > 3:
        config.batch_size = int(sys.argv[3])
    if len(sys.argv) > 4:
        config.name = sys.argv[4]  # Allow custom model name
    
    # Train model
    dino_ssl, manager = dino_train(
        config=config,
        csv_path=csv_path,
        device=device
    )
    
    print("\n" + "="*70)
    print("To use the trained embeddings:")
    print("="*70)
    print(f"""
# Load the trained model from models/{config.name}/checkpoints/best_model.pth
checkpoint = torch.load('models/{config.name}/checkpoints/best_model.pth', weights_only=False)
student_model = SSL_GINEModel(
    num_features={config.num_features},
    edge_features={config.edge_features},
    hidden_dim={config.hidden_dim},
    num_layers={config.num_layers},
)
student_model.load_state_dict(checkpoint['model_state_dict'])

# Extract embeddings
from torch_geometric.loader import DataLoader
from model.dataset_creation import SmilesCsvDataset

dataset = SmilesCsvDataset('{csv_path}')
loader = DataLoader(dataset, batch_size=64, shuffle=False)

student_model.eval()
embeddings = []
for batch in loader:
    with torch.no_grad():
        emb = student_model.get_embeddings(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        embeddings.append(emb.cpu())
        
embeddings = torch.cat(embeddings, dim=0)
print(f"Extracted embeddings: {{embeddings.shape}}")
    """)

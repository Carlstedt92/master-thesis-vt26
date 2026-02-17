"""Training manager for handling model directory structure and checkpointing."""

import os
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


class TrainingManager:
    """Manages model directory structure, checkpoints, and loss history."""
    
    def __init__(self, config):
        """
        Initialize training manager.
        
        Args:
            config: ModelConfig object with model and training settings
        """
        self.config = config
        self.model_dir = self._setup_directories()
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        self.loss_history: List[Dict[str, float | None]] = []
        self.best_loss = float('inf')
        self.start_time = datetime.now()
        
        # Save config
        self._save_config()
    
    def _setup_directories(self) -> str:
        """Create model directory structure.
        
        Returns:
            Path to model directory
        """
        model_dir = os.path.join("./models", self.config.name)
        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"✓ Model directory: {model_dir}")
        
        return model_dir
    
    def _save_config(self):
        """Save configuration to JSON."""
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        print(f"✓ Config saved: {config_path}")
    
    def record_loss(self, epoch: int, train_loss: float,
                    val_loss: float | None = None,
                    test_loss: float | None = None):
        """Record losses for an epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            train_loss: Average training loss for the epoch
            val_loss: Optional validation loss for the epoch
            test_loss: Optional test loss for the epoch
        """
        self.loss_history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "test_loss": None if test_loss is None else float(test_loss),
        })
        
        if train_loss < self.best_loss:
            self.best_loss = train_loss
    
    def save_checkpoint(self, epoch: int, model, optimizer, loss: float, 
                       is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number (0-indexed)
            model: Model to save
            optimizer: Optimizer state
            loss: Loss value for this checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config.to_dict(),
        }
        
        # Regular checkpoint (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Checkpoint saved: epoch {epoch + 1}")
        
        # Best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved (loss: {loss:.6f})")
    
    def save_loss_history(self):
        """Save loss history to JSON."""
        history_path = os.path.join(self.model_dir, "loss_history.json")
        history_records = self.loss_history
        with open(history_path, 'w') as f:
            json.dump(history_records, f, indent=2)
        print(f"✓ Loss history saved: {history_path}")
    
    def save_metadata(self):
        """Save training metadata and summary."""
        elapsed_time = datetime.now() - self.start_time
        
        train_losses = [entry["train_loss"] for entry in self.loss_history]
        val_losses = [entry["val_loss"] for entry in self.loss_history if entry["val_loss"] is not None]
        test_losses = [entry["test_loss"] for entry in self.loss_history if entry["test_loss"] is not None]

        metadata = {
            'model_name': self.config.name,
            'best_loss': float(self.best_loss),
            'best_epoch': train_losses.index(self.best_loss) + 1 if train_losses else 0,
            'final_loss': float(train_losses[-1]) if train_losses else None,
            'total_epochs': len(self.loss_history),
            'training_time_seconds': elapsed_time.total_seconds(),
            'training_time_hours': elapsed_time.total_seconds() / 3600,
            'timestamp': self.start_time.isoformat(),
        }

        if val_losses:
            best_val_loss = min(val_losses)
            metadata.update({
                'best_val_loss': float(best_val_loss),
                'best_val_epoch': next(
                    entry["epoch"] for entry in self.loss_history
                    if entry["val_loss"] is not None and entry["val_loss"] == best_val_loss
                ),
                'final_val_loss': float(val_losses[-1]),
            })

        if test_losses:
            best_test_loss = min(test_losses)
            metadata.update({
                'best_test_loss': float(best_test_loss),
                'best_test_epoch': next(
                    entry["epoch"] for entry in self.loss_history
                    if entry["test_loss"] is not None and entry["test_loss"] == best_test_loss
                ),
                'final_test_loss': float(test_losses[-1]),
            })
        
        metadata_path = os.path.join(self.model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Training metadata saved: {metadata_path}")
        print(f"\nTraining Summary:")
        print(f"  Best loss: {metadata['best_loss']:.6f} (epoch {metadata['best_epoch']})")
        print(f"  Final loss: {metadata['final_loss']:.6f}")
        if 'best_val_loss' in metadata:
            print(f"  Best val loss: {metadata['best_val_loss']:.6f} (epoch {metadata['best_val_epoch']})")
            print(f"  Final val loss: {metadata['final_val_loss']:.6f}")
        if 'best_test_loss' in metadata:
            print(f"  Best test loss: {metadata['best_test_loss']:.6f} (epoch {metadata['best_test_epoch']})")
            print(f"  Final test loss: {metadata['final_test_loss']:.6f}")
        print(f"  Total time: {metadata['training_time_hours']:.2f} hours")
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer=None):
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            
        Returns:
            Dictionary with checkpoint info
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.6f}")
        
        return checkpoint
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics.
        
        Returns:
            Dictionary with training metrics
        """
        if not self.loss_history:
            return {}
        
        train_losses = [entry["train_loss"] for entry in self.loss_history]
        return {
            'current_epoch': len(self.loss_history),
            'current_loss': train_losses[-1],
            'best_loss': min(train_losses),
            'best_epoch': train_losses.index(min(train_losses)) + 1,
        }

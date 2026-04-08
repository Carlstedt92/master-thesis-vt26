"""Training manager for handling model directory structure and checkpointing."""

import os
import json
import shutil
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
        # Separate loss histories for DINO SSL and downstream evaluation
        self.dino_loss_history: List[Dict[str, float | None]] = []
        self.eval_loss_history: Dict[str, List[Dict[str, Any]]] = {}
        self.best_loss = float('inf')  # For DINO training
        self.best_eval_metric = None  # For validation metrics (MSE for regression, ROC-AUC for classification)
        self.best_eval_epoch = None
        self.online_eval_history: List[Dict[str, Any]] = []
        self.top_eval_checkpoints: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
        # Save config
        self._save_config()
        self._initialize_metadata_file()
    
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

    def _initialize_metadata_file(self):
        """Ensure metadata.json exists and contains a valid object."""
        metadata_path = os.path.join(self.model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            with open(metadata_path, 'w') as f:
                json.dump({}, f, indent=2)
            print(f"✓ Metadata initialized: {metadata_path}")

    def _read_metadata(self) -> Dict[str, Any]:
        """Read metadata.json safely."""
        metadata_path = os.path.join(self.model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return {}

        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _write_metadata(self, metadata: Dict[str, Any]):
        """Write metadata.json."""
        metadata_path = os.path.join(self.model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _update_metadata_section(self, section_name: str, section_data: Dict[str, Any]):
        """Merge a metadata section into metadata.json."""
        metadata = self._read_metadata()
        metadata[section_name] = section_data
        self._write_metadata(metadata)
    
    def record_loss(self, epoch: int, train_loss: float, diagnostics: Dict[str, Any] | None = None):
        """Record DINO SSL training loss for an epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            train_loss: Average training loss for the epoch
        """
        record: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
        }
        if diagnostics is not None:
            record.update(diagnostics)

        self.dino_loss_history.append(record)
        
        if train_loss < self.best_loss:
            self.best_loss = train_loss
    
    def record_eval_metrics(self, method_name: str, epoch: int, **metrics):
        """Record evaluation metrics for downstream tasks and track best metric.
        
        Args:
            method_name: Name of evaluation method (e.g., 'regression', 'classification')
            epoch: Current epoch number (0-indexed)
            **metrics: Variable metrics (e.g., mse, rmse, r2 for regression)
        """
        if method_name not in self.eval_loss_history:
            self.eval_loss_history[method_name] = []
        
        record = {"epoch": epoch + 1}
        record.update({k: float(v) if isinstance(v, (int, float)) else v 
                      for k, v in metrics.items()})
        self.eval_loss_history[method_name].append(record)
        
        # Track best validation metric
        if method_name == 'regression' and 'val_mse' in metrics:
            current_metric = float(metrics['val_mse'])
            # Lower is better for MSE
            if self.best_eval_metric is None or current_metric < self.best_eval_metric:
                self.best_eval_metric = current_metric
                self.best_eval_epoch = epoch + 1
        elif method_name == 'classification' and 'val_roc_auc' in metrics:
            current_metric = float(metrics['val_roc_auc'])
            # Higher is better for ROC-AUC
            if self.best_eval_metric is None or current_metric > self.best_eval_metric:
                self.best_eval_metric = current_metric
                self.best_eval_epoch = epoch + 1
    
    def save_checkpoint(self, epoch: int, model, optimizer, loss: float = None, 
                       is_best: bool = False, metric_value: float = None):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number (0-indexed)
            model: Model to save
            optimizer: Optimizer state
            loss: Loss value for this checkpoint (optional, for DINO training)
            is_best: Whether this is the best model so far
            metric_value: Evaluation metric value for this checkpoint (optional)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss if loss is not None else 0.0,
            'eval_metric': metric_value,
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
            if metric_value is not None:
                print(f"  ✓ Best model saved (eval metric: {metric_value:.6f})")
            elif loss is not None:
                print(f"  ✓ Best model saved (loss: {loss:.6f})")
            else:
                print(f"  ✓ Best model saved")

    def _to_json_safe(self, value: Any):
        """Recursively convert values to JSON-serializable Python scalars/containers."""
        if isinstance(value, dict):
            return {str(k): self._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [self._to_json_safe(v) for v in value]
        if isinstance(value, (str, bool)) or value is None:
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    def _save_top_eval_index(self):
        """Save a compact top-k checkpoint index for plotting/reporting."""
        index_path = os.path.join(self.model_dir, "top_eval_checkpoints.json")
        payload = {
            "top_eval_checkpoints": self.top_eval_checkpoints,
            "generated_at": datetime.now().isoformat(),
        }
        with open(index_path, "w") as f:
            json.dump(payload, f, indent=2)

    def record_online_eval(self, epoch: int, ssl_loss: float, eval_result: Dict[str, Any], saved_path: str | None):
        """Record online downstream evaluation results for one epoch."""
        record = {
            "epoch": int(epoch + 1),
            "ssl_train_loss": float(ssl_loss),
            "saved_checkpoint_path": saved_path,
            "evaluation": self._to_json_safe(eval_result),
        }
        self.online_eval_history.append(record)
        self.eval_loss_history["online_eval"] = self.online_eval_history

    def update_top_eval_checkpoints(
        self,
        epoch: int,
        model,
        optimizer,
        ssl_loss: float,
        eval_result: Dict[str, Any],
        top_k: int = 5,
    ) -> str | None:
        """Keep only top-k checkpoints by online aggregate eval score (higher is better)."""
        if top_k <= 0:
            return None

        score = float(eval_result.get("aggregate_primary_score", float("-inf")))
        if not self.top_eval_checkpoints:
            should_save = True
        elif len(self.top_eval_checkpoints) < top_k:
            should_save = True
        else:
            worst_score = min(item["score"] for item in self.top_eval_checkpoints)
            should_save = score > worst_score

        if not should_save:
            return None

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": float(ssl_loss),
            "eval_metric": score,
            "online_eval": self._to_json_safe(eval_result),
            "config": self.config.to_dict(),
        }

        checkpoint_name = f"checkpoint_online_eval_epoch_{epoch + 1}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)

        entry = {
            "epoch": int(epoch + 1),
            "score": score,
            "ssl_train_loss": float(ssl_loss),
            "checkpoint_path": checkpoint_path,
            "metric_definition": eval_result.get(
                "aggregate_primary_score_definition",
                "mean(validation primary metric across datasets)",
            ),
            "datasets": eval_result.get("dataset_names", []),
        }
        self.top_eval_checkpoints.append(entry)
        self.top_eval_checkpoints.sort(key=lambda item: (item["score"], -item["epoch"]), reverse=True)

        while len(self.top_eval_checkpoints) > top_k:
            dropped = self.top_eval_checkpoints.pop()
            dropped_path = dropped.get("checkpoint_path")
            if dropped_path and os.path.exists(dropped_path):
                os.remove(dropped_path)

        best_entry = self.top_eval_checkpoints[0]
        best_source = best_entry.get("checkpoint_path")
        best_target = os.path.join(self.checkpoint_dir, "best_online_eval_model.pth")
        if best_source and os.path.exists(best_source):
            shutil.copy2(best_source, best_target)

        self._save_top_eval_index()
        return checkpoint_path

    def save_loss_history(self, verbose: bool = True):
        """Save structured loss history to JSON."""
        history_path = os.path.join(self.model_dir, "loss_history.json")
        
        history_records = {
            "DINO_Loss": self.dino_loss_history,
            "Evaluation_Loss": self.eval_loss_history
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_records, f, indent=2)
        if verbose:
            print(f"✓ Loss history saved: {history_path}")
    
    def save_model_metadata(self):
        """Save model-level metadata under 'Model data'."""
        elapsed_time = datetime.now() - self.start_time

        model_data = {
            'name': self.config.name,
            'seed': self.config.seed,
            'head_type': self.config.head_type,
            'training_time_seconds': elapsed_time.total_seconds(),
            'training_time_hours': elapsed_time.total_seconds() / 3600,
            'timestamp': self.start_time.isoformat(),
        }

        self._update_metadata_section('Model data', model_data)
        print(f"✓ Model metadata saved: {os.path.join(self.model_dir, 'metadata.json')}")

    def save_dino_metadata(self):
        """Save DINO training metadata under 'DINO_data'."""
        train_losses = [entry["train_loss"] for entry in self.dino_loss_history]
        if not train_losses:
            print("✓ DINO metadata skipped: no DINO training history")
            return

        warning_epochs = [
            int(entry["epoch"]) for entry in self.dino_loss_history
            if bool(entry.get("collapse_warning", False))
        ]

        dino_data = {
            'data_path': self.config.data_path,
            'best_loss': float(self.best_loss),
            'best_epoch': train_losses.index(self.best_loss) + 1,
            'final_loss': float(train_losses[-1]),
            'total_epochs': len(self.dino_loss_history),
            'collapse_warning_epochs': warning_epochs,
            'collapse_warning_count': len(warning_epochs),
        }

        if warning_epochs:
            dino_data['first_collapse_warning_epoch'] = int(warning_epochs[0])

        self._update_metadata_section('DINO_data', dino_data)
        print(f"✓ DINO metadata saved: {os.path.join(self.model_dir, 'metadata.json')}")
        print(f"  Best loss: {dino_data['best_loss']:.6f} (epoch {dino_data['best_epoch']})")
        print(f"  Final loss: {dino_data['final_loss']:.6f}")

    def save_online_eval_metadata(self):
        """Save online eval metadata under 'Online_eval_data'."""
        if not self.online_eval_history:
            return

        best_entry = None
        for entry in self.online_eval_history:
            evaluation = entry.get("evaluation", {})
            score = evaluation.get("aggregate_primary_score")
            if score is None:
                continue
            if best_entry is None or float(score) > float(best_entry["evaluation"]["aggregate_primary_score"]):
                best_entry = entry

        online_eval_data = {
            "total_evaluated_epochs": len(self.online_eval_history),
            "top_k_retained": int(getattr(self.config, "online_eval_top_k_checkpoints", 5)),
            "fixed_k": int(getattr(self.config, "online_eval_fixed_k", 5)),
            "every_n_epochs": int(getattr(self.config, "online_eval_every_n_epochs", 1)),
            "top_eval_checkpoints": self.top_eval_checkpoints,
        }

        if best_entry is not None:
            online_eval_data["best_epoch"] = int(best_entry["epoch"])
            online_eval_data["best_aggregate_primary_score"] = float(
                best_entry["evaluation"]["aggregate_primary_score"]
            )

        self._update_metadata_section("Online_eval_data", online_eval_data)
        print(f"✓ Online eval metadata saved: {os.path.join(self.model_dir, 'metadata.json')}")

    def save_regression_metadata(self):
        """Save regression evaluation metadata under 'Regression_data'."""
        regression_history = self.eval_loss_history.get('regression', [])
        if not regression_history:
            print("✓ Regression metadata skipped: no regression history")
            return

        final_regression = regression_history[-1]
        regression_data = {
            'total_epochs': len(regression_history),
            'final_metrics': final_regression,
        }

        val_mse_entries = [
            entry for entry in regression_history
            if entry.get('val_mse') is not None
        ]
        if val_mse_entries:
            best_val_entry = min(val_mse_entries, key=lambda entry: entry['val_mse'])
            regression_data.update({
                'best_val_mse': float(best_val_entry['val_mse']),
                'best_val_epoch': int(best_val_entry['epoch']),
            })

        self._update_metadata_section('Regression_data', regression_data)
        print(f"✓ Regression metadata saved: {os.path.join(self.model_dir, 'metadata.json')}")

    def save_classification_metadata(self):
        """Save classification evaluation metadata under 'Classification_data'."""
        classification_history = self.eval_loss_history.get('classification', [])
        if not classification_history:
            print("✓ Classification metadata skipped: no classification history")
            return

        final_classification = classification_history[-1]
        classification_data = {
            'total_epochs': len(classification_history),
            'final_metrics': final_classification,
        }

        val_entries = [
            entry for entry in classification_history
            if entry.get('val_roc_auc') is not None
        ]
        if val_entries:
            best_val_entry = max(val_entries, key=lambda entry: entry['val_roc_auc'])
            classification_data.update({
                'best_val_roc_auc': float(best_val_entry['val_roc_auc']),
                'best_val_epoch': int(best_val_entry['epoch']),
            })

        self._update_metadata_section('Classification_data', classification_data)
        print(f"✓ Classification metadata saved: {os.path.join(self.model_dir, 'metadata.json')}")

    def save_metadata(self):
        """Backward-compatible metadata save wrapper."""
        self.save_model_metadata()
        self.save_dino_metadata()
        self.save_online_eval_metadata()
        self.save_regression_metadata()
        self.save_classification_metadata()
    
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
        if not self.dino_loss_history:
            return {}
        
        train_losses = [entry["train_loss"] for entry in self.dino_loss_history]
        return {
            'current_epoch': len(self.dino_loss_history),
            'current_loss': train_losses[-1],
            'best_loss': min(train_losses),
            'best_epoch': train_losses.index(min(train_losses)) + 1,
        }
    
    def is_best_eval_metric(self, method_name: str, metric_value: float) -> bool:
        """Check if the current metric is the best so far.
        
        Args:
            method_name: Task name ('regression' or 'classification')
            metric_value: Current metric value (val_mse or val_roc_auc)
            
        Returns:
            True if this is the best metric so far
        """
        if self.best_eval_metric is None:
            return True
        
        if method_name == 'regression':
            # Lower is better for MSE
            return metric_value < self.best_eval_metric
        elif method_name == 'classification':
            # Higher is better for ROC-AUC
            return metric_value > self.best_eval_metric
        
        return False

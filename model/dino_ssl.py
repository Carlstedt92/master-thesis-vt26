"""DINO SSL Framework for Graph Neural Networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class DINOLoss(nn.Module):
    """DINO loss with centering and sharpening."""
    
    def __init__(self, out_dim: int, teacher_temp: float = 0.04, student_temp: float = 0.1,
                 center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
    
    def forward(self, student_output, teacher_output):
        """
        Args:
            student_output: (batch_size, out_dim) tensor from student
            teacher_output: (batch_size, out_dim) tensor from teacher
        """
        student_out = student_output / self.student_temp

        # Proper DINO teacher branch: center then sharpen.
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()  # Stop gradient
        
        # Cross-entropy loss: KL divergence
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        
        return loss.mean()
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output with exponential moving average."""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOGraphSSL:
    """DINO Self-Supervised Learning framework for graphs."""
    
    def __init__(self, student_model, teacher_model=None, device='cuda', 
                 teacher_temp=0.04, student_temp=0.1, center_momentum=0.9,
                 teacher_momentum=0.996):
        """
        Args:
            student_model: Student network (SSL_GINEModel)
            teacher_model: Teacher network (if None, will be created as copy of student)
            device: Device to run on
            teacher_temp: Temperature for teacher softmax
            student_temp: Temperature for student softmax
            center_momentum: Momentum for center update
            teacher_momentum: Momentum for teacher EMA update
        """
        self.device = device
        self.student = student_model.to(device)
        
        # Initialize teacher as copy of student
        if teacher_model is None:
            self.teacher = deepcopy(student_model).to(device)
        else:
            self.teacher = teacher_model.to(device)
        
        # Teacher doesn't require gradients
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.teacher_momentum = teacher_momentum
        
        # DINO loss
        output_dim = student_model.head.mlp[-1].out_features
        self.loss_fn = DINOLoss(
            out_dim=output_dim,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            center_momentum=center_momentum
        ).to(device)

    @classmethod
    def from_config(cls, student_model, config, teacher_model=None):
        """Build DINO SSL wrapper from config."""
        return cls(
            student_model=student_model,
            teacher_model=teacher_model,
            device=config.device,
            teacher_temp=config.teacher_temp,
            student_temp=config.student_temp,
            center_momentum=config.center_momentum,
            teacher_momentum=config.teacher_momentum,
        )
    
    @torch.no_grad()
    def update_teacher(self):
        """Update teacher network with EMA of student parameters."""
        for student_param, teacher_param in zip(self.student.parameters(), 
                                                  self.teacher.parameters()):
            teacher_param.data.mul_(self.teacher_momentum)
            teacher_param.data.add_((1 - self.teacher_momentum) * student_param.data)
    
    def train_step(self, batch, optimizer):
        """
        Single training step with multi-crop augmented batch.
        
        Args:
            batch: PyG Batch containing multiple views (global + local) of multiple graphs
                   Each graph in batch has attributes:
                   - view: tensor [1] for global, [0] for local
                   - graph_idx: original graph identifier tensor
            optimizer: Optimizer for student network
            
        Returns:
            Dictionary with scalar loss and batch diagnostics
        """
        self.student.train()
        self.teacher.eval()
        
        # Move data to device
        batch = batch.to(self.device)
        
        # Get number of graphs in batch (each graph is one view)
        num_graphs = batch.num_graphs
        
        # Identify global views (view == 1)
        # batch['view'] is concatenated tensor: [1, 0, 1, 0, ...] for each graph
        global_mask = (batch['view'] == 1).squeeze()
        if global_mask.dim() == 0:
            global_mask = global_mask.unsqueeze(0)
        
        # Get indices of global view graphs in the batch
        global_indices = torch.where(global_mask)[0]
        
        # Forward pass: Student sees ALL views
        student_out_all = self.student(
            batch.x, batch.edge_index, 
            batch.edge_attr, batch.batch
        )
        
        # Forward pass: Teacher sees only GLOBAL views
        with torch.no_grad():
            # Create a mask for nodes belonging to global view graphs
            global_graph_mask = torch.isin(batch.batch, global_indices)
            
            # Extract global view node indices
            global_node_indices = torch.where(global_graph_mask)[0]
            
            # Filter edges: both source and target must be in global views
            edge_mask = global_graph_mask[batch.edge_index[0]] & global_graph_mask[batch.edge_index[1]]
            global_edge_index = batch.edge_index[:, edge_mask]
            
            # Remap node indices to be contiguous (0 to len(global_node_indices)-1)
            node_mapping = torch.full((batch.x.size(0),), -1, dtype=torch.long, device=self.device)
            node_mapping[global_node_indices] = torch.arange(len(global_node_indices), device=self.device)
            global_edge_index = node_mapping[global_edge_index]
            
            # Get edge attributes for global views
            global_edge_attr = batch.edge_attr[edge_mask] if batch.edge_attr is not None else None
            
            # Remap batch assignment for global views
            global_batch_raw = batch.batch[global_node_indices]
            # Map old graph indices to new contiguous indices
            unique_global_graphs = global_indices
            graph_mapping = torch.full((num_graphs,), -1, dtype=torch.long, device=self.device)
            graph_mapping[unique_global_graphs] = torch.arange(len(unique_global_graphs), device=self.device)
            global_batch = graph_mapping[global_batch_raw]
            
            # Forward through teacher
            teacher_out = self.teacher(
                batch.x[global_node_indices],
                global_edge_index,
                global_edge_attr,
                global_batch
            )
        
        # Compute DINO loss by matching student/teacher graph_idx pairs.
        # Vectorized implementation avoids Python-side nested loops.
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get graph_idx for each view (concatenated tensor)
        # batch['graph_idx'] is concatenated across all graphs in batch
        student_graph_idx = batch['graph_idx'].squeeze()  # Shape: [num_graphs]
        if student_graph_idx.dim() == 0:
            student_graph_idx = student_graph_idx.unsqueeze(0)
        
        teacher_graph_idx = student_graph_idx[global_indices]  # Only global views
        
        # Pairwise cross-entropy matrix:
        # pair_loss[t, s] = -sum( teacher_prob[t] * log_softmax(student_logits[s]) )
        student_log_probs = F.log_softmax(student_out_all / self.loss_fn.student_temp, dim=-1)
        teacher_probs = F.softmax(
            (teacher_out - self.loss_fn.center) / self.loss_fn.teacher_temp,
            dim=-1,
        ).detach()
        pair_loss = -torch.matmul(teacher_probs, student_log_probs.T)

        # Keep only student/teacher pairs that came from the same source graph.
        # Shape: [num_teacher_views, num_student_views]
        match_mask = teacher_graph_idx.unsqueeze(1) == student_graph_idx.unsqueeze(0)
        num_matches = match_mask.sum()

        if num_matches > 0:
            loss = pair_loss[match_mask].mean()
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        with torch.no_grad():
            student_probs = F.softmax(student_out_all / self.loss_fn.student_temp, dim=-1)
            teacher_entropy = (-teacher_probs * torch.log(teacher_probs.clamp_min(1e-12))).sum(dim=-1).mean()
            student_entropy = (-student_probs * torch.log(student_probs.clamp_min(1e-12))).sum(dim=-1).mean()
            embedding_std = student_out_all.detach().std(dim=0, unbiased=False).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update teacher with EMA
        self.update_teacher()
        
        # Update center with teacher outputs
        self.loss_fn.update_center(teacher_out)
        
        return {
            "loss": float(loss.item()),
            "metrics": {
                "teacher_entropy": float(teacher_entropy.item()),
                "student_entropy": float(student_entropy.item()),
                "embedding_std": float(embedding_std.item()),
            },
        }
    
    def get_embeddings(self, data):
        """Extract embeddings from student encoder (for downstream tasks)."""
        self.student.eval()
        with torch.no_grad():
            data = data.to(self.device)
            embeddings = self.student.get_embeddings(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
        return embeddings


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, 
                     warmup_epochs=0, start_warmup_value=0.0):
    """Cosine learning rate schedule with warmup."""
    warmup_schedule = torch.linspace(start_warmup_value, base_value, 
                                      warmup_epochs * niter_per_ep)
    
    iters = torch.arange(epochs * niter_per_ep - warmup_epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + torch.cos(torch.pi * iters / len(iters))
    )
    
    schedule = torch.cat((warmup_schedule, schedule))
    return schedule.numpy()

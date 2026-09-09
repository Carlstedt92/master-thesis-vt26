"""DINO SSL Framework for Graph Neural Networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from copy import deepcopy
import time

try:
    from torch_geometric.nn import DataParallel as PyGDataParallel
except ImportError:
    PyGDataParallel = None


class GraphBatchParallelWrapper(nn.Module):
    """Wrap a graph model so DataParallel can split by graph list instead of node tensors."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch_or_data_list):
        if isinstance(batch_or_data_list, Batch):
            batch = batch_or_data_list
        else:
            if not batch_or_data_list:
                output_dim = int(self.model.head.mlp[-1].out_features)
                device = next(self.model.parameters()).device
                return torch.empty((0, output_dim), device=device)

            batch = Batch.from_data_list(batch_or_data_list)

        return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


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
        """Update center used for teacher output with exponential moving average.

        Under DDP each rank only sees its own shard of the batch, so the raw
        per-rank mean is not the true batch center -- all-reduce it first so
        every rank centers against the same global statistic. Without this,
        ranks would silently drift onto different centers, which is worse
        than not running DDP at all.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        count = torch.tensor(float(teacher_output.size(0)), device=teacher_output.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_center, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        batch_center = batch_center / count.clamp_min(1.0)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOGraphSSL:
    """DINO Self-Supervised Learning framework for graphs."""
    
    def __init__(self, student_model, teacher_model=None, device='cuda',
                 teacher_temp=0.04, student_temp=0.1, center_momentum=0.9,
                 teacher_momentum=0.996, profile_timing: bool = False,
                 use_data_parallel: bool = False,
                 use_ddp: bool = False,
                 ddp_local_rank: int = 0,
                 sync_teacher_batchnorm_buffers: bool = True,
                 use_node_reconstruction_loss: bool = False,
                 node_reconstruction_loss_weight: float = 1.0):
        """
        Args:
            student_model: Student network (SSL_GINEModel)
            teacher_model: Teacher network (if None, will be created as copy of student)
            device: Device to run on
            teacher_temp: Temperature for teacher softmax
            student_temp: Temperature for student softmax
            center_momentum: Momentum for center update
            teacher_momentum: Momentum for teacher EMA update
            use_ddp: Wrap the student in torch.nn.parallel.DistributedDataParallel.
                Unlike use_data_parallel (single-process, graph-list splitting),
                DDP is multi-process: each rank already gets its own data shard
                via the sampler, so forward() keeps the plain (x, edge_index,
                edge_attr, batch) signature -- DDP only adds gradient
                all-reduce on backward(). Requires torch.distributed to already
                be initialized (see training/dino_training.py) and expects
                `device` to be this process's own cuda:<local_rank>.
            ddp_local_rank: Local GPU index for this process, passed to DDP's
                device_ids so it knows which device the module lives on.
            sync_teacher_batchnorm_buffers: Copy student BatchNorm running stats
                (buffers, not parameters) to the teacher each step. Buffers are
                excluded from the parameter EMA loop, so without this the
                teacher's BatchNorm running_mean/running_var stay frozen at
                their values from the initial deepcopy for the whole run.
        """
        self.device = device
        self.student_model = student_model.to(device)
        device_type = torch.device(device).type if not isinstance(device, torch.device) else device.type
        self.use_data_parallel = bool(
            use_data_parallel
            and device_type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
            and PyGDataParallel is not None
        )
        self.use_ddp = bool(
            use_ddp
            and device_type == "cuda"
            and torch.cuda.is_available()
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if self.use_ddp and self.use_data_parallel:
            raise ValueError("use_ddp and use_data_parallel are mutually exclusive")

        self.use_node_reconstruction_loss = bool(use_node_reconstruction_loss)
        self.node_reconstruction_loss_weight = float(node_reconstruction_loss_weight)
        if self.use_node_reconstruction_loss and self.use_data_parallel:
            raise ValueError(
                "use_node_reconstruction_loss is not supported together with use_data_parallel "
                "(the graph-list-splitting wrapper doesn't thread node_reconstruction_mask through "
                "to each shard). Use DDP (use_ddp=True) or single-device instead -- both are already "
                "supported."
            )

        if self.use_data_parallel:
            self.student = PyGDataParallel(GraphBatchParallelWrapper(self.student_model))
        elif self.use_ddp:
            # SyncBatchNorm computes running stats across all ranks' local
            # batches via all-reduce during forward(). Without it each rank's
            # BatchNorm buffers reflect only its own shard and diverge from
            # every other rank's -- the same class of bug as the
            # student-to-teacher buffer copy this file already fixes, just
            # across processes instead of across networks.
            self.student_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.student_model)
            self.student = torch.nn.parallel.DistributedDataParallel(
                self.student_model, device_ids=[ddp_local_rank]
            )
        else:
            self.student = self.student_model

        # Initialize teacher as copy of student
        if teacher_model is None:
            self.teacher = deepcopy(self.student_model).to(device)
        else:
            self.teacher = teacher_model.to(device)
        
        # Teacher doesn't require gradients
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.teacher_momentum = teacher_momentum
        self.profile_timing = profile_timing
        self.sync_teacher_batchnorm_buffers = bool(sync_teacher_batchnorm_buffers)
        
        # DINO loss
        output_dim = student_model.head.mlp[-1].out_features
        self.loss_fn = DINOLoss(
            out_dim=output_dim,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            center_momentum=center_momentum
        ).to(device)

        # Second DINO-style loss for the masked-node reconstruction auxiliary
        # objective (see ModelConfig.use_node_reconstruction_loss) -- its own
        # center buffer, since centering a node-level embedding space against
        # a graph-level one's running mean would be meaningless. Shares the
        # same teacher/student temperature schedule as the graph loss rather
        # than introducing a second one to tune, matching common practice
        # (e.g. DINOv2) of sharing schedules across both loss terms.
        if self.use_node_reconstruction_loss:
            node_output_dim = student_model.node_head.mlp[-1].out_features
            self.node_loss_fn = DINOLoss(
                out_dim=node_output_dim,
                teacher_temp=teacher_temp,
                student_temp=student_temp,
                center_momentum=center_momentum,
            ).to(device)

    def _forward_student(self, batch, node_reconstruction_mask=None):
        if self.use_data_parallel:
            return self.student(batch.to_data_list())
        # DDP wraps forward() transparently (same call signature as the plain
        # student_model), so it shares this branch with the single-device path.
        if node_reconstruction_mask is not None:
            return self.student(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                                 node_reconstruction_mask=node_reconstruction_mask)
        return self.student(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    def _forward_student_eval(self, batch):
        """Eval-time forward (used by eval_step, no backward pass follows) that
        bypasses the DDP wrapper entirely. DistributedDataParallel broadcasts
        buffers on every forward() call regardless of train/eval mode or
        no_grad -- if eval_step ran through self.student and different ranks
        called it a different number of times (e.g. validation only run on
        rank 0), that mismatch alone desyncs the process group and hangs.
        There's no gradient sync needed for eval, so route around DDP
        entirely and call the plain module directly, matching how
        _student_get_embeddings already bypasses it for the same reason."""
        if self.use_ddp:
            if self.use_node_reconstruction_loss:
                return self.student_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                                           node_reconstruction_mask=batch.node_reconstruction_mask)
            return self.student_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        if self.use_node_reconstruction_loss:
            return self._forward_student(batch, node_reconstruction_mask=batch.node_reconstruction_mask)
        return self._forward_student(batch)

    def _student_get_embeddings(self, data):
        if self.use_data_parallel or self.use_ddp:
            # get_embeddings() is a custom method, not forward() -- DDP/DataParallel
            # wrappers only proxy __call__, so call it on the unwrapped module directly.
            return self.student_model.get_embeddings(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
        return self.student.get_embeddings(data.x, data.edge_index, data.edge_attr, data.batch)

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
            profile_timing=bool(getattr(config, "profile_timing", False)),
            use_data_parallel=bool(getattr(config, "use_data_parallel", False)),
            use_ddp=bool(getattr(config, "use_ddp", False)),
            ddp_local_rank=int(getattr(config, "ddp_local_rank", 0)),
            sync_teacher_batchnorm_buffers=bool(getattr(config, "sync_teacher_batchnorm_buffers", True)),
            use_node_reconstruction_loss=bool(getattr(config, "use_node_reconstruction_loss", False)),
            node_reconstruction_loss_weight=float(getattr(config, "node_reconstruction_loss_weight", 1.0)),
        )
    
    @torch.no_grad()
    def update_teacher(self):
        """Update teacher network with EMA of student parameters."""
        for student_param, teacher_param in zip(self.student.parameters(),
                                                  self.teacher.parameters()):
            teacher_param.data.mul_(self.teacher_momentum)
            teacher_param.data.add_((1 - self.teacher_momentum) * student_param.data)

        if self.sync_teacher_batchnorm_buffers:
            # BatchNorm's running_mean/running_var/num_batches_tracked are
            # buffers, not parameters, so the EMA loop above never touches
            # them -- copy directly rather than re-EMA-ing values that are
            # already an exponential average internally.
            for student_buffer, teacher_buffer in zip(self.student.buffers(),
                                                        self.teacher.buffers()):
                teacher_buffer.data.copy_(student_buffer.data)
    
    def _match_teacher_node_embeddings(self, batch, num_graphs, global_node_indices, student_graph_idx, teacher_node_embeddings_global):
        """Match every masked student node to its corresponding teacher
        global-view node and project the match through teacher.node_head.
        Shared by train_step and eval_step so this correspondence logic only
        has one place to get right (or fix later).

        Returns teacher_node_out aligned 1:1, in order, with
        torch.where(batch.node_reconstruction_mask)[0] -- or None if nothing
        was masked in this batch.

        A masking-mode local view preserves the exact same per-molecule node
        order/count as its global view (masking only zeros features in
        place, never reindexes -- see graph_augmentation.py), so a node's
        position within its own Data object, combined with graph_idx, is a
        correspondence key valid across both view types without any
        separate node-id bookkeeping.
        """
        student_masked_idx = torch.where(batch.node_reconstruction_mask)[0]
        if student_masked_idx.numel() == 0:
            return None

        node_counts = torch.bincount(batch.batch, minlength=num_graphs)
        graph_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=self.device)
        graph_ptr[1:] = torch.cumsum(node_counts, dim=0)
        node_pos_in_own_view = torch.arange(batch.x.size(0), device=self.device) - graph_ptr[batch.batch]
        node_graph_idx_all = student_graph_idx[batch.batch]

        KEY_MULT = 100_000  # safely larger than any molecule's atom count
        student_keys = node_graph_idx_all[student_masked_idx] * KEY_MULT + node_pos_in_own_view[student_masked_idx]
        teacher_keys = node_graph_idx_all[global_node_indices] * KEY_MULT + node_pos_in_own_view[global_node_indices]

        sorted_teacher_keys, sort_order = torch.sort(teacher_keys)
        insert_pos = torch.searchsorted(sorted_teacher_keys, student_keys).clamp(max=sorted_teacher_keys.numel() - 1)
        found = sorted_teacher_keys[insert_pos] == student_keys
        if not bool(found.all()):
            raise RuntimeError(
                "Masked-node reconstruction: a masked student node had no matching "
                "teacher global-view node. Every graph_idx group always has >=1 global "
                "view, so this should be impossible -- check the node-order assumptions "
                "in graph_augmentation.py before debugging further."
            )
        teacher_row_idx = sort_order[insert_pos]
        return self.teacher.node_head(teacher_node_embeddings_global[teacher_row_idx])

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
        profile_enabled = self.profile_timing
        timing = {}
        step_start = time.perf_counter()
        
        # Move data to device
        to_device_start = time.perf_counter()
        batch = batch.to(self.device, non_blocking=True)
        if profile_enabled:
            timing["to_device"] = time.perf_counter() - to_device_start
        
        # Get number of graphs in batch (each graph is one view)
        num_graphs = batch.num_graphs
        
        # Identify global views (view == 1)
        # batch['view'] is concatenated tensor: [1, 0, 1, 0, ...] for each graph
        global_mask = (batch['view'] == 1).squeeze()
        if global_mask.dim() == 0:
            global_mask = global_mask.unsqueeze(0)
        
        # Get indices of global view graphs in the batch
        global_indices = torch.where(global_mask)[0]

        # Get graph_idx for each view (concatenated tensor) -- computed here
        # (rather than down in the loss section, where this used to live)
        # since the node-reconstruction correspondence matching below also
        # needs it. batch['graph_idx'] is one value per graph/view.
        student_graph_idx = batch['graph_idx'].squeeze()  # Shape: [num_graphs]
        if student_graph_idx.dim() == 0:
            student_graph_idx = student_graph_idx.unsqueeze(0)

        # Forward pass: Student sees ALL views
        student_fwd_start = time.perf_counter()
        student_node_out = None
        if self.use_node_reconstruction_loss:
            student_out_all, student_node_out = self._forward_student(
                batch, node_reconstruction_mask=batch.node_reconstruction_mask
            )
        else:
            student_out_all = self._forward_student(batch)
        if profile_enabled:
            timing["student_forward"] = time.perf_counter() - student_fwd_start
        
        # Forward pass: Teacher sees only GLOBAL views
        with torch.no_grad():
            teacher_fwd_start = time.perf_counter()
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
            
            # Forward through teacher. When the node-reconstruction loss is on,
            # route through encoder+head manually instead of self.teacher(...)
            # so the per-node embeddings (needed for the node loss) come from
            # this SAME forward pass rather than a second, wasted one -- the
            # graph-level teacher_out this produces is identical either way.
            teacher_node_out = None
            if self.use_node_reconstruction_loss:
                teacher_graph_embedding, teacher_node_embeddings_global = self.teacher.encoder(
                    batch.x[global_node_indices], global_edge_index, global_edge_attr, global_batch,
                    return_node_embeddings=True,
                )
                teacher_out = self.teacher.head(teacher_graph_embedding)
            else:
                teacher_out = self.teacher(
                    batch.x[global_node_indices],
                    global_edge_index,
                    global_edge_attr,
                    global_batch,
                )
            if profile_enabled:
                timing["teacher_forward"] = time.perf_counter() - teacher_fwd_start

            if self.use_node_reconstruction_loss:
                node_recon_start = time.perf_counter()
                teacher_node_out = self._match_teacher_node_embeddings(
                    batch, num_graphs, global_node_indices, student_graph_idx, teacher_node_embeddings_global
                )
                if profile_enabled:
                    timing["node_recon_match"] = time.perf_counter() - node_recon_start
        
        # Compute DINO loss by matching student/teacher graph_idx pairs.
        # Vectorized implementation avoids Python-side nested loops.
        loss_start = time.perf_counter()
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

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

        graph_dino_loss_value = float(loss.item())
        node_recon_loss_value = None
        if self.use_node_reconstruction_loss and teacher_node_out is not None:
            # student_node_out/teacher_node_out are already row-aligned 1:1
            # pairs (matched above), so the existing all-pairs machinery
            # above isn't needed here -- DINOLoss.forward already does
            # exactly this (center, sharpen, cross-entropy, mean) for
            # row-aligned student/teacher pairs.
            node_loss = self.node_loss_fn(student_node_out, teacher_node_out)
            loss = loss + self.node_reconstruction_loss_weight * node_loss
            node_recon_loss_value = float(node_loss.item())

        if profile_enabled:
            timing["loss_compute"] = time.perf_counter() - loss_start

        with torch.no_grad():
            student_probs = F.softmax(student_out_all / self.loss_fn.student_temp, dim=-1)
            teacher_entropy = (-teacher_probs * torch.log(teacher_probs.clamp_min(1e-12))).sum(dim=-1).mean()
            student_entropy = (-student_probs * torch.log(student_probs.clamp_min(1e-12))).sum(dim=-1).mean()
            
            # Embedding std on projection head output
            embedding_std = student_out_all.detach().std(dim=0, unbiased=False).mean()
            
            # Encoder embedding std (before projection head)
            encoder_embeddings = self._student_get_embeddings(batch)
            encoder_embedding_std = encoder_embeddings.detach().std(dim=0, unbiased=False).mean()
        
        # Backward pass
        backward_start = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if profile_enabled:
            timing["backward_step"] = time.perf_counter() - backward_start
        
        # Update teacher with EMA
        ema_start = time.perf_counter()
        self.update_teacher()
        
        # Update center with teacher outputs
        self.loss_fn.update_center(teacher_out)
        if self.use_node_reconstruction_loss and teacher_node_out is not None:
            self.node_loss_fn.update_center(teacher_node_out)
        if profile_enabled:
            timing["ema_and_center"] = time.perf_counter() - ema_start
            timing["train_step_total"] = time.perf_counter() - step_start

        metrics = {
            "teacher_entropy": float(teacher_entropy.item()),
            "student_entropy": float(student_entropy.item()),
            "embedding_std": float(embedding_std.item()),
            "encoder_embedding_std": float(encoder_embedding_std.item()),
        }
        if self.use_node_reconstruction_loss:
            # graph_dino_loss lets the two components be told apart in
            # loss_history.json even though "loss" (used for backward/
            # checkpoint-selection/plots) is their combined sum -- without
            # this, one term silently dominating the other is invisible
            # until it's already too late to catch mid-run.
            metrics["graph_dino_loss"] = graph_dino_loss_value
            metrics["node_recon_loss"] = node_recon_loss_value  # None on a batch with zero masked nodes this step

        return {
            "loss": float(loss.item()),
            "metrics": metrics,
            "timing": timing,
        }

    @torch.no_grad()
    def eval_step(self, batch):
        """Compute the SSL loss and diagnostics without updating weights or center."""
        self.student.eval()
        self.teacher.eval()

        batch = batch.to(self.device, non_blocking=True)

        num_graphs = batch.num_graphs
        global_mask = (batch['view'] == 1).squeeze()
        if global_mask.dim() == 0:
            global_mask = global_mask.unsqueeze(0)

        global_indices = torch.where(global_mask)[0]

        student_node_out = None
        if self.use_node_reconstruction_loss:
            student_out_all, student_node_out = self._forward_student_eval(batch)
        else:
            student_out_all = self._forward_student_eval(batch)

        global_graph_mask = torch.isin(batch.batch, global_indices)
        global_node_indices = torch.where(global_graph_mask)[0]

        edge_mask = global_graph_mask[batch.edge_index[0]] & global_graph_mask[batch.edge_index[1]]
        global_edge_index = batch.edge_index[:, edge_mask]

        node_mapping = torch.full((batch.x.size(0),), -1, dtype=torch.long, device=self.device)
        node_mapping[global_node_indices] = torch.arange(len(global_node_indices), device=self.device)
        global_edge_index = node_mapping[global_edge_index]

        global_edge_attr = batch.edge_attr[edge_mask] if batch.edge_attr is not None else None

        global_batch_raw = batch.batch[global_node_indices]
        unique_global_graphs = global_indices
        graph_mapping = torch.full((num_graphs,), -1, dtype=torch.long, device=self.device)
        graph_mapping[unique_global_graphs] = torch.arange(len(unique_global_graphs), device=self.device)
        global_batch = graph_mapping[global_batch_raw]

        teacher_node_out = None
        if self.use_node_reconstruction_loss:
            teacher_graph_embedding, teacher_node_embeddings_global = self.teacher.encoder(
                batch.x[global_node_indices], global_edge_index, global_edge_attr, global_batch,
                return_node_embeddings=True,
            )
            teacher_out = self.teacher.head(teacher_graph_embedding)
        else:
            teacher_out = self.teacher(
                batch.x[global_node_indices],
                global_edge_index,
                global_edge_attr,
                global_batch,
            )

        student_graph_idx = batch['graph_idx'].squeeze()
        if student_graph_idx.dim() == 0:
            student_graph_idx = student_graph_idx.unsqueeze(0)

        if self.use_node_reconstruction_loss:
            teacher_node_out = self._match_teacher_node_embeddings(
                batch, num_graphs, global_node_indices, student_graph_idx, teacher_node_embeddings_global
            )

        teacher_graph_idx = student_graph_idx[global_indices]

        student_log_probs = F.log_softmax(student_out_all / self.loss_fn.student_temp, dim=-1)
        teacher_probs = F.softmax(
            (teacher_out - self.loss_fn.center) / self.loss_fn.teacher_temp,
            dim=-1,
        ).detach()
        pair_loss = -torch.matmul(teacher_probs, student_log_probs.T)

        match_mask = teacher_graph_idx.unsqueeze(1) == student_graph_idx.unsqueeze(0)
        num_matches = match_mask.sum()

        if num_matches > 0:
            loss = pair_loss[match_mask].mean()
        else:
            loss = torch.tensor(0.0, device=self.device)

        student_probs = F.softmax(student_out_all / self.loss_fn.student_temp, dim=-1)
        teacher_entropy = (-teacher_probs * torch.log(teacher_probs.clamp_min(1e-12))).sum(dim=-1).mean()
        student_entropy = (-student_probs * torch.log(student_probs.clamp_min(1e-12))).sum(dim=-1).mean()
        
        # Embedding std on projection head output
        embedding_std = student_out_all.detach().std(dim=0, unbiased=False).mean()
        
        # Encoder embedding std (before projection head)
        encoder_embeddings = self._student_get_embeddings(batch)
        encoder_embedding_std = encoder_embeddings.detach().std(dim=0, unbiased=False).mean()

        eval_metrics = {
            "teacher_entropy": float(teacher_entropy.item()),
            "student_entropy": float(student_entropy.item()),
            "embedding_std": float(embedding_std.item()),
            "encoder_embedding_std": float(encoder_embedding_std.item()),
        }
        if self.use_node_reconstruction_loss:
            eval_metrics["graph_dino_loss"] = float(loss.item())
            if teacher_node_out is not None:
                node_loss = self.node_loss_fn(student_node_out, teacher_node_out)
                loss = loss + self.node_reconstruction_loss_weight * node_loss
                eval_metrics["node_recon_loss"] = float(node_loss.item())
            else:
                eval_metrics["node_recon_loss"] = None

        return {
            "loss": float(loss.item()),
            "metrics": eval_metrics,
        }

    def get_embeddings(self, data):
        """Extract embeddings from student encoder (for downstream tasks)."""
        self.student.eval()
        with torch.no_grad():
            data = data.to(self.device, non_blocking=True)
            embeddings = self.student_model.get_embeddings(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
        return embeddings


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, 
                     warmup_epochs=0, start_warmup_value=0.0):
    """Cosine learning rate schedule with warmup."""
    warmup_epochs = max(0, min(int(warmup_epochs), int(epochs)))
    warmup_iters = warmup_epochs * niter_per_ep
    total_iters = epochs * niter_per_ep

    warmup_schedule = torch.linspace(start_warmup_value, base_value, warmup_iters)
    
    iters = torch.arange(max(total_iters - warmup_iters, 0))
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + torch.cos(torch.pi * iters / len(iters)) if len(iters) > 0 else torch.tensor([], dtype=torch.float32)
    )
    
    schedule = torch.cat((warmup_schedule, schedule))
    return schedule.numpy()

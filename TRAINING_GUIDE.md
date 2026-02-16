# DINO SSL Training Guide for Your Setup

## Overview

Your DINO SSL setup is now complete and follows the proper multi-crop strategy where:

1. **DataLoader** creates batches with multiple augmented views per graph
2. **Teacher** processes only global views  
3. **Student** processes all views (global + local)
4. **Loss** is computed between student and teacher outputs with matching `graph_idx`

## Architecture Flow

```
Original Graph (from CSV)
    ↓
GraphAugmentation (in collate_fn)
    ↓
2 Global Views + 4 Local Views (per graph)
    ↓
Batched together (Batch.from_data_list)
    ↓
Single batch with metadata:
    - view: [1, 1, 0, 0, 0, 0, ...] (1=global, 0=local)
    - graph_idx: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, ...]
    ↓
┌──────────────────────────────────────┐
│  DINO Training Step                  │
│                                      │
│  Teacher:                            │
│    - Filters batch['view'] == 1      │
│    - Processes only global views     │
│                                      │
│  Student:                            │
│    - Processes all views             │
│                                      │
│  Loss:                               │
│    - Match by graph_idx              │
│    - Compare student vs teacher      │
└──────────────────────────────────────┘
```

## Key Components

### 1. Graph Augmentation ([graph_augmentation.py](model/graph_augmentation.py))

```python
class GraphAugmentation:
    def __init__(self, local_views=4):
        self.local_views = local_views
    
    def __call__(self, data):
        # Creates 2 global + 4 local views
        return [
            global_view1,   # view=1, graph_idx=X
            global_view2,   # view=1, graph_idx=X
            local_view1,    # view=0, graph_idx=X
            local_view2,    # view=0, graph_idx=X
            local_view3,    # view=0, graph_idx=X
            local_view4,    # view=0, graph_idx=X
        ]
```

### 2. DataLoader ([dataloader_creation.py](model/dataloader_creation.py))

```python
def collate_fn(batch):
    """Apply augmentation and flatten into single batch."""
    augmenter = GraphAugmentation(local_views=4)
    augmented = [augmenter(data) for data in batch]
    flat = [view for views in augmented for view in views]
    return Batch.from_data_list(flat)
```

**Example**: If batch_size=2, the collate function produces:
- 2 graphs × 6 views each = 12 graphs in the batch
- `batch.view`: `[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]`
- `batch.graph_idx`: `[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]`

### 3. DINO Training Step ([dino_ssl.py](model/dino_ssl.py))

```python
def train_step(self, batch, optimizer):
    # 1. Identify global views\n    global_mask = (batch.view == 1).squeeze()
    global_indices = torch.where(global_mask)[0]
    
    # 2. Student processes ALL views
    student_out_all = self.student(batch.x, batch.edge_index, 
                                    batch.edge_attr, batch.batch)
    
    # 3. Teacher processes only GLOBAL views
    with torch.no_grad():
        # Extract nodes/edges for global views only
        global_graph_mask = torch.isin(batch.batch, global_indices)
        # ... [filtering logic] ...
        teacher_out = self.teacher(global_x, global_edge_index, 
                                   global_edge_attr, global_batch)
    
    # 4. Compute loss by matching graph_idx
    for s_idx in range(num_graphs):
        s_gid = batch.graph_idx[s_idx]
        # Find teacher outputs with same graph_idx
        matching_teacher = (teacher_graph_idx == s_gid)
        # Compute loss for each match
        ...
```

## Usage

### Quick Start

```bash
python train_dino.py
```

### With Custom Parameters

```bash
python train_dino.py data/your_data.csv 100 32
# Arguments: csv_path num_epochs batch_size
```

### In Code

```python
from model.dataloader_creation import create_dataloader
from model.gin_model import SSL_GINEModel
from model.dino_ssl import DINOGraphSSL

# Create dataloader (applies augmentation automatically)
train_loader = create_dataloader("data/delaney-processed.csv", batch_size=32)

# Initialize model
student_model = SSL_GINEModel(num_features=9, edge_features=3, hidden_dim=128)

# Initialize DINO
dino_ssl = DINOGraphSSL(student_model, device='cuda')

# Optimizer
optimizer = torch.optim.AdamW(dino_ssl.student.parameters(), lr=0.0005)

# Training loop
for batch in train_loader:
    # batch contains all augmented views with metadata
    loss = dino_ssl.train_step(batch, optimizer)
```

## Understanding the Batch Structure

When you print a batch, you'll see:

```python
DataBatch(
    x=[N, num_features],          # Node features (all views concatenated)
    edge_index=[2, E],            # Edges (all views concatenated)
    edge_attr=[E, edge_features], # Edge features (all views concatenated)
    batch=[N],                    # Node-to-graph mapping (0, 0, ..., 1, 1, ..., 11)
    view=[G, 1],                  # View type per graph (1=global, 0=local)
    graph_idx=[G, 1],             # Original graph ID (0, 0, ..., 0, 1, 1, ..., 1)
    num_graphs=G                  # Total number of views in batch (batch_size × 6)
)
```

Where:
- `N` = total number of nodes across all views
- `E` = total number of edges across all views
- `G` = total number of graphs (views) in batch = `batch_size × 6`

## Example Walkthrough

### Input
- Batch size: 2 graphs
- Each graph creates: 2 global + 4 local views
- Total views in batch: 12

### Processing

1. **Student Forward Pass**
   - Sees all 12 views
   - Produces 12 output embeddings

2. **Teacher Forward Pass**
   - Filters to get only global views (where `view==1`)
   - Extracts 4 global views (2 per original graph)
   - Produces 4 output embeddings

3. **Loss Computation**
   - For each of 12 student outputs:
     - Find teacher outputs with matching `graph_idx`
     - Compute cross-entropy loss
   - Example for graph_idx=0:
     - 6 student views (2 global + 4 local)
     - 2 teacher views (2 global)
     - 6 × 2 = 12 comparisons

4. **Total Comparisons**
   - 2 graphs × 6 student views × 2 teacher views = 24 loss terms
   - Average over all comparisons

## Hyperparameters

### Model Architecture
```python
hidden_dim = 128           # GINE hidden dimension
num_layers = 5             # Number of GINE layers
dropout = 0.5              # Dropout rate
projection_hidden_dim = 2048
projection_output_dim = 256
```

### DINO Settings
```python
teacher_temp = 0.04        # Teacher temperature (sharpening)
student_temp = 0.1         # Student temperature
center_momentum = 0.9      # Center update momentum
teacher_momentum = 0.996   # EMA momentum (increases to 1.0)
```

### Training
```python
learning_rate = 0.0005     # Base learning rate
weight_decay = 0.04        # AdamW weight decay
warmup_epochs = 10         # LR warmup period
batch_size = 32            # Number of original graphs per batch
```

## Monitoring Training

The training script prints:

```
Epoch [1/100] Batch [0/50] Loss: 2.3456 | Graphs: 32 (Global: 64, Local: 128) | LR: 0.000050
```

- **Graphs**: Number of unique original graphs in batch
- **Global**: Number of global views (should be 2 × Graphs)
- **Local**: Number of local views (should be 4 × Graphs)
- **Loss**: Average DINO loss across all comparisons
- **LR**: Current learning rate (with warmup and cosine decay)

## Extracting Embeddings

After training:

```python
# Load trained model
checkpoint = torch.load('best_dino_model.pth')
model = SSL_GINEModel(**checkpoint['config'])
model.load_state_dict(checkpoint['student_state_dict'])
model.eval()

# Extract embeddings (without augmentation)
from torch_geometric.loader import DataLoader
from model.dataset_creation import SmilesCsvDataset

dataset = SmilesCsvDataset('data/delaney-processed.csv')
loader = DataLoader(dataset, batch_size=64, shuffle=False)

all_embeddings = []
with torch.no_grad():
    for batch in loader:
        embeddings = model.get_embeddings(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        all_embeddings.append(embeddings.cpu())

embeddings = torch.cat(all_embeddings, dim=0)
# Shape: [num_molecules, hidden_dim]
```

## Troubleshooting

### Issue: Loss not decreasing

**Check**:
- Is teacher updating? (print teacher params)
- Are augmentations correct? (visualize views)
- Is batch.view correct? (should be mix of 0s and 1s)
- Is graph_idx preserved? (should match original graph IDs)

**Fix**:
```python
# Debug batch structure
print(f"Batch views: {batch.view}")
print(f"Graph IDs: {batch.graph_idx}")
print(f"Unique graphs: {len(torch.unique(batch.graph_idx))}")
```

### Issue: Dimension mismatch

**Check**:
- Node features match num_features
- Edge features match edge_features
- All graphs have edge_attr or none have it

**Fix**:
```python
# Ensure consistent edge attributes
if batch.edge_attr is None:
    edge_features = 1  # Use dummy dimension
```

### Issue: Out of memory

**Reduce**:
- Batch size (fewer original graphs)
- Hidden dimension
- Number of layers
- Local views (4 → 2)

## Files Reference

- [`model/gin_model.py`](model/gin_model.py) - GINE encoder and SSL model
- [`model/dino_ssl.py`](model/dino_ssl.py) - DINO framework
- [`model/graph_augmentation.py`](model/graph_augmentation.py) - Multi-crop augmentation
- [`model/dataloader_creation.py`](model/dataloader_creation.py) - DataLoader with augmentation
- [`model/dataset_creation.py`](model/dataset_creation.py) - SMILES dataset
- [`model/graph_creation.py`](model/graph_creation.py) - SMILES to graph conversion
- [`train_dino.py`](train_dino.py) - Complete training script

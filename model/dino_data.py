# DINO-style: generate multiple views per molecule, flatten into one batch, keep mol_id/view_type mapping.
from typing import List
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

def simple_augment(data: Data, feature_mask_prob: float = 0.1) -> Data:
    """Return a lightly augmented copy of a graph by masking node features."""
    aug = data.clone()
    if aug.x is not None and aug.x.numel() > 0:
        mask = torch.rand_like(aug.x) < feature_mask_prob
        aug.x = aug.x.masked_fill(mask, 0.0)
    return aug

class DINOViewDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list: List[str], num_global_views: int = 2, num_local_views: int = 4):
        self.smiles_list = smiles_list
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> List[Data]:
        base = smiles_to_pyg(self.smiles_list[idx])
        views: List[Data] = []
        for _ in range(self.num_global_views):
            views.append(simple_augment(base, feature_mask_prob=0.05))
        for _ in range(self.num_local_views):
            views.append(simple_augment(base, feature_mask_prob=0.2))
        return views

def dino_collate(view_lists: List[List[Data]]) -> Batch:
    """Flatten views and attach mapping fields for regrouping."""
    flat_views: List[Data] = []
    for mol_id, views in enumerate(view_lists):
        for view_id, view in enumerate(views):
            view_type = 0 if view_id < 2 else 1  # 0 = global, 1 = local
            view.mol_id = torch.tensor([mol_id], dtype=torch.long)
            view.view_id = torch.tensor([view_id], dtype=torch.long)
            view.view_type = torch.tensor([view_type], dtype=torch.long)
            flat_views.append(view)
    return Batch.from_data_list(flat_views)

# Example: batch_size=2 molecules, 2 global + 4 local views each => 12 graphs total
smiles_list = [molecules["Methanol"], molecules["Ethanol"], molecules["Benzene"]]
dataset = DINOViewDataset(smiles_list, num_global_views=2, num_local_views=4)
loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dino_collate)

batch = next(iter(loader))
print(type(batch))
print(batch)
print("Total graphs in batch:", batch.num_graphs)
print("x shape (total_nodes, num_features):", batch.x.shape)
print("nodes per graph:", torch.bincount(batch.batch).tolist())
print("mol_id per graph:", batch.mol_id.view(-1).tolist())
print("view_type per graph:", batch.view_type.view(-1).tolist())

# Regroup views by molecule inside the training step# DINO-style: generate multiple views per molecule, flatten into one batch, keep mol_id/view_type mapping.
from typing import List
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

def simple_augment(data: Data, feature_mask_prob: float = 0.1) -> Data:
    """Return a lightly augmented copy of a graph by masking node features."""
    aug = data.clone()
    if aug.x is not None and aug.x.numel() > 0:
        mask = torch.rand_like(aug.x) < feature_mask_prob
        aug.x = aug.x.masked_fill(mask, 0.0)
    return aug

class DINOViewDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list: List[str], num_global_views: int = 2, num_local_views: int = 4):
        self.smiles_list = smiles_list
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> List[Data]:
        base = smiles_to_pyg(self.smiles_list[idx])
        views: List[Data] = []
        for _ in range(self.num_global_views):
            views.append(simple_augment(base, feature_mask_prob=0.05))
        for _ in range(self.num_local_views):
            views.append(simple_augment(base, feature_mask_prob=0.2))
        return views

def dino_collate(view_lists: List[List[Data]]) -> Batch:
    """Flatten views and attach mapping fields for regrouping."""
    flat_views: List[Data] = []
    for mol_id, views in enumerate(view_lists):
        for view_id, view in enumerate(views):
            view_type = 0 if view_id < 2 else 1  # 0 = global, 1 = local
            view.mol_id = torch.tensor([mol_id], dtype=torch.long)
            view.view_id = torch.tensor([view_id], dtype=torch.long)
            view.view_type = torch.tensor([view_type], dtype=torch.long)
            flat_views.append(view)
    return Batch.from_data_list(flat_views)

# Example: batch_size=2 molecules, 2 global + 4 local views each => 12 graphs total
smiles_list = [molecules["Methanol"], molecules["Ethanol"], molecules["Benzene"]]
dataset = DINOViewDataset(smiles_list, num_global_views=2, num_local_views=4)
loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dino_collate)

batch = next(iter(loader))
print(type(batch))
print(batch)
print("Total graphs in batch:", batch.num_graphs)
print("x shape (total_nodes, num_features):", batch.x.shape)
print("nodes per graph:", torch.bincount(batch.batch).tolist())
print("mol_id per graph:", batch.mol_id.view(-1).tolist())
print("view_type per graph:", batch.view_type.view(-1).tolist())

# Regroup views by molecule inside the training step
unique_mol_ids = torch.unique(batch.mol_id.view(-1))
views_by_mol = {
    int(mol_id): (batch.mol_id.view(-1) == mol_id).nonzero(as_tuple=True)[0]
    for mol_id in unique_mol_ids
}
print("Indices per molecule:", {k: v.tolist() for k, v in views_by_mol.items()})
unique_mol_ids = torch.unique(batch.mol_id.view(-1))
views_by_mol = {
    int(mol_id): (batch.mol_id.view(-1) == mol_id).nonzero(as_tuple=True)[0]
    for mol_id in unique_mol_ids
}
print("Indices per molecule:", {k: v.tolist() for k, v in views_by_mol.items()})

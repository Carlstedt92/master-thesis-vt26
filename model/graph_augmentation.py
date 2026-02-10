
import random
import torch
from torch_geometric.data import Data

class GraphAugmentation:
    def __init__(self, local_views=4):
        self.local_views = local_views

    def global_augmentation(self, data):
        """Create a global augmented view by copying the graph structure."""
        return Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
            y=data.y.clone() if data.y is not None else None,
            graph_idx=data.graph_idx.clone() if hasattr(data, 'graph_idx') else None,
        )

    def local_augmentation(self, data):
        """Create local augmented view by selecting random atom and its 2-hop neighbors."""
        num_atoms = data.x.size(0)
        if num_atoms == 0:
            return data
        
        # Select random center atom
        center_atom_idx = random.randint(0, num_atoms - 1)
        
        # Find 1-hop neighbors
        neighbors = set([center_atom_idx])
        edge_index = data.edge_index
        
        for i in range(edge_index.size(1)):
            if edge_index[0, i].item() == center_atom_idx:
                neighbors.add(edge_index[1, i].item())
            elif edge_index[1, i].item() == center_atom_idx:
                neighbors.add(edge_index[0, i].item())
        
        # Find 2-hop neighbors
        neighbors_2hop = set(neighbors)
        for neighbor in list(neighbors):
            for i in range(edge_index.size(1)):
                if edge_index[0, i].item() == neighbor:
                    neighbors_2hop.add(edge_index[1, i].item())
                elif edge_index[1, i].item() == neighbor:
                    neighbors_2hop.add(edge_index[0, i].item())
        
        neighbors_list = sorted(list(neighbors_2hop))
        
        # Create node index mapping
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(neighbors_list)}
        
        # Extract node features
        mask = torch.zeros(num_atoms, dtype=torch.bool)
        for idx in neighbors_list:
            mask[idx] = True
        new_x = data.x[mask]
        
        # Extract edges within the subgraph
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        new_edge_index = edge_index[:, edge_mask].clone()
        
        # Remap edge indices to match subgraph node ordering
        for i in range(new_edge_index.size(1)):
            new_edge_index[0, i] = node_map[new_edge_index[0, i].item()]
            new_edge_index[1, i] = node_map[new_edge_index[1, i].item()]
        
        # Extract edge attributes if they exist
        new_edge_attr = None
        if data.edge_attr is not None:
            new_edge_attr = data.edge_attr[edge_mask]
        
        return Data(
            x=new_x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            y=data.y.clone() if data.y is not None else None,
            graph_idx=data.graph_idx.clone() if hasattr(data, 'graph_idx') else None,
        )

    def __call__(self, data):
        """Generate multiple augmented views of the input graph."""
        aug_data_list = []
        
        # Create two global views
        aug_data_list.append(self.global_augmentation(data))
        aug_data_list.append(self.global_augmentation(data))
        
        # Create local views
        for _ in range(self.local_views):
            aug_data_list.append(self.local_augmentation(data))
        
        return aug_data_list
    
class DINOMoleculeDataset(torch.utils.data.Dataset):
    """Dataset wrapper for DINO-style contrastive learning with graph augmentations."""
    
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        aug_data_list = self.transform(data)
        return aug_data_list
    

# Example usage:
if __name__ == "__main__":
    from rdkit import Chem
    from graph_creation import smiles_to_pygdata

    smiles = "CCO"
    data = smiles_to_pygdata(smiles)
    augmenter = GraphAugmentation(local_views=2)
    augmented_views = augmenter(data)
    print(augmented_views)
    for i, aug_data in enumerate(augmented_views):
        print(f"Augmented View {i}:")
        print(aug_data)
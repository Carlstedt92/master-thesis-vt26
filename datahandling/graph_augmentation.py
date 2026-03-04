
import random
import torch
from torch_geometric.data import Data

class GraphAugmentation:
    def __init__(self, local_views=4, k_hops=2):
        self.local_views = local_views
        self.k_hops = k_hops

    def global_augmentation(self, data):
        """Create a global augmented view by copying the graph structure."""
        return Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
            y=data.y.clone() if data.y is not None else None,
            graph_idx=data.graph_idx.clone() if hasattr(data, 'graph_idx') else None,
            view=torch.tensor([1], dtype=torch.long)  # 1 = global
        )

    def local_augmentation(self, data, num_hops=2):
        """Create local augmented view by selecting random atom and its k-hop neighbors."""
        num_atoms = data.x.size(0)
        if num_atoms == 0:
            return data
        
        # Select random center atom
        center_atom_idx = random.randint(0, num_atoms - 1)
        
        # Find k-hop neighbors
        edge_index = data.edge_index
        neighbors_list = self.k_hop_subgraph(center_atom_idx, num_hops, edge_index)
        
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
            view=torch.tensor([0], dtype=torch.long)  # 0 = local
        )

    # Function for creating K-hop neighbor subgraph
    def k_hop_subgraph(self, node_idx, num_hops, edge_index):
        neighbors = set([node_idx])
        for _ in range(num_hops):
            new_neighbors = set()
            for neighbor in neighbors:
                for i in range(edge_index.size(1)):
                    if edge_index[0, i].item() == neighbor:
                        new_neighbors.add(edge_index[1, i].item())
                    elif edge_index[1, i].item() == neighbor:
                        new_neighbors.add(edge_index[0, i].item())
            neighbors.update(new_neighbors)
        return sorted(list(neighbors))

    def __call__(self, data):
        """Generate multiple augmented views of the input graph."""
        aug_data_list = []
        
        # Create two global views
        aug_data_list.append(self.global_augmentation(data))
        aug_data_list.append(self.global_augmentation(data))
        
        # Create local views
        for _ in range(self.local_views):
            aug_data_list.append(self.local_augmentation(data, num_hops=self.k_hops))
        
        return aug_data_list



# Example usage:
if __name__ == "__main__":
    from .graph_creation import smiles_to_pygdata

    smiles = "CCO"
    data = smiles_to_pygdata(smiles)
    augmenter = GraphAugmentation(local_views=2, k_hops=1)
    augmented_views = augmenter(data)
    print(augmented_views)
    for i, aug_data in enumerate(augmented_views):
        print(f"Augmented View {i}:")
        print(aug_data)
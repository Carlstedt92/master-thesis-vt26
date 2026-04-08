
import random
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

class GraphAugmentation:
    def __init__(
        self,
        local_views=4,
        k_hops=2,
        local_augmentation_mode="k_hop",
        node_mask_ratio=0.15,
        feature_mask_ratio=0.15,
    ):
        self.local_views = local_views
        self.k_hops = k_hops
        self.local_augmentation_mode = local_augmentation_mode
        self.node_mask_ratio = node_mask_ratio
        self.feature_mask_ratio = feature_mask_ratio

    def _sample_count(self, total_items, ratio):
        if total_items <= 0 or ratio <= 0:
            return 0
        return min(total_items, max(1, int(round(total_items * ratio))))

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
        
        # Use PyG's efficient k_hop_subgraph function
        # Returns: subset (node indices), edge_index (remapped), mapping, edge_mask
        subset, new_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=center_atom_idx,
            num_hops=num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=num_atoms
        )
        
        # Extract node features for the subset
        new_x = data.x[subset]
        
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

    def masked_local_augmentation(self, data, node_mask_ratio=0.15, feature_mask_ratio=0.15):
        """Create a local view by masking nodes first and then masking features on the remaining nodes."""
        if data.x is None:
            return self.global_augmentation(data)

        num_nodes, num_features = data.x.size()
        if num_nodes == 0 or num_features == 0:
            return self.global_augmentation(data)

        num_node_mask = self._sample_count(num_nodes, node_mask_ratio)
        node_mask_indices = set(random.sample(range(num_nodes), num_node_mask)) if num_node_mask > 0 else set()
        remaining_indices = [idx for idx in range(num_nodes) if idx not in node_mask_indices]

        masked_x = data.x.clone()
        if node_mask_indices:
            masked_x[list(node_mask_indices)] = 0

        num_feature_mask = self._sample_count(num_features, feature_mask_ratio)
        feature_mask_indices = random.sample(range(num_features), num_feature_mask) if num_feature_mask > 0 else []
        if remaining_indices and feature_mask_indices:
            for node_idx in remaining_indices:
                masked_x[node_idx, feature_mask_indices] = 0

        return Data(
            x=masked_x,
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
            y=data.y.clone() if data.y is not None else None,
            graph_idx=data.graph_idx.clone() if hasattr(data, 'graph_idx') else None,
            view=torch.tensor([0], dtype=torch.long),  # 0 = local, keep compatibility with existing code
        )

    def __call__(self, data):
        """Generate multiple augmented views of the input graph."""
        aug_data_list = []

        # Create two global views
        aug_data_list.append(self.global_augmentation(data))
        aug_data_list.append(self.global_augmentation(data))

        # Create local views according to the configured strategy
        for _ in range(self.local_views):
            if self.local_augmentation_mode == "masking":
                aug_data_list.append(
                    self.masked_local_augmentation(
                        data,
                        node_mask_ratio=self.node_mask_ratio,
                        feature_mask_ratio=self.feature_mask_ratio,
                    )
                )
            elif self.local_augmentation_mode == "k_hop":
                aug_data_list.append(self.local_augmentation(data, num_hops=self.k_hops))
            else:
                raise ValueError(f"Unsupported local_augmentation_mode: {self.local_augmentation_mode}")
        
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
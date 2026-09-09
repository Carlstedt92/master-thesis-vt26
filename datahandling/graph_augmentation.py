
import random
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from rdkit import Chem

class GraphAugmentation:
    # Order matters: matches are claimed greedily and exclusively (see
    # _functional_group_atom_sets), so a pattern earlier in this list can
    # "swallow" atoms a later, narrower pattern would also have matched.
    # Phenol is deliberately placed (and widened to the whole ring) before
    # the generic Alcohol pattern -- [OX2H][#6] would otherwise match a
    # phenolic O-H's 2-atom anchor first and permanently claim it, leaving
    # the dedicated Phenol pattern unreachable and the ring itself unusable
    # as a local-view seed (verified empirically: previously, a phenol's
    # ring could never be selected via Aromatic_Ring either, since the
    # 2-atom Alcohol claim always overlapped it).
    FUNCTIONAL_GROUP_PATTERNS = [
        ("Carboxylic_Acid", "[CX3](=O)[OX2H1]"),
        ("Ester", "[CX3](=O)[OX2][#6]"),
        ("Phenol", "[OX2H]c1ccccc1"),
        ("Alcohol", "[OX2H][#6]"),
        ("Aldehyde", "[CX3H1](=O)[#6]"),
        ("Ketone", "[#6][CX3](=O)[#6]"),
        ("Amine_Primary", "[NX3;H2][#6]"),
        ("Amine_Secondary", "[NX3;H1]([#6])[#6]"),
        ("Amine_Tertiary", "[NX3;H0]([#6])([#6])[#6]"),
        # Generic aromatic-ring queries (RDKit "a" = any aromatic atom, not
        # just carbon) -- covers heteroaryl rings (pyridine, pyrimidine,
        # furan, thiophene, pyrrole, imidazole, ...) and 5-membered aromatic
        # rings, not just literal benzene. Replaces the old benzo-only
        # "c1ccccc1" pattern, which missed the majority of aromatic rings
        # responsible for the functional_group_k_hop fallback rate (measured:
        # ~81% of fallback molecules had a heteroaromatic and/or 5-membered
        # aromatic ring that the old pattern couldn't see at all).
        ("Aromatic_Ring_6", "[a]1[a][a][a][a][a]1"),
        ("Aromatic_Ring_5", "[a]1[a][a][a][a]1"),
        ("Ether", "[OX2]([#6])[#6]"),
        ("Amide", "[NX3][CX3](=[OX1])[#6]"),
    ]

    _compiled_functional_group_patterns = None

    def __init__(
        self,
        local_views=4,
        k_hops=2,
        local_augmentation_mode="k_hop",
        node_mask_ratio=0.15,
        feature_mask_ratio=0.15,
        local_view_modes=None,
    ):
        self.local_views = local_views
        self.k_hops = k_hops
        self.local_augmentation_mode = local_augmentation_mode
        self.node_mask_ratio = node_mask_ratio
        self.feature_mask_ratio = feature_mask_ratio
        # Optional per-view mode override -- e.g. ["functional_group_k_hop",
        # "functional_group_masking"] to mix augmentation styles across local
        # views instead of using local_augmentation_mode uniformly for all of
        # them. Must have exactly local_views entries if set. None (default)
        # reproduces the original uniform-mode behavior exactly.
        if local_view_modes is not None and len(local_view_modes) != local_views:
            raise ValueError(
                f"local_view_modes has {len(local_view_modes)} entries but "
                f"local_views={local_views}; they must match."
            )
        self.local_view_modes = local_view_modes

    @classmethod
    def _get_compiled_functional_group_patterns(cls):
        """Compile SMARTS patterns once and reuse across all instances/calls."""
        if cls._compiled_functional_group_patterns is None:
            compiled = []
            for name, smarts in cls.FUNCTIONAL_GROUP_PATTERNS:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is not None:
                    compiled.append((name, pattern))
            cls._compiled_functional_group_patterns = compiled
        return cls._compiled_functional_group_patterns

    def _copy_optional_metadata(self, data):
        kwargs = {}
        if hasattr(data, "y") and data.y is not None:
            kwargs["y"] = data.y.clone()
        if hasattr(data, "graph_idx") and data.graph_idx is not None:
            kwargs["graph_idx"] = data.graph_idx.clone()
        if hasattr(data, "smiles") and data.smiles is not None:
            kwargs["smiles"] = data.smiles
        return kwargs

    def _sample_count(self, total_items, ratio):
        if total_items <= 0 or ratio <= 0:
            return 0
        return min(total_items, max(1, int(round(total_items * ratio))))

    def _smiles_to_mol(self, data):
        smiles = getattr(data, "smiles", None)
        if not smiles:
            return None
        return Chem.MolFromSmiles(smiles)

    def _functional_group_atom_sets(self, data):
        """Compute functional-group atom sets once per graph (not once per view)."""
        mol = self._smiles_to_mol(data)
        if mol is None:
            return []

        assigned_atoms = set()
        groups = []

        for _, pattern in self._get_compiled_functional_group_patterns():
            for match in mol.GetSubstructMatches(pattern):
                atom_set = set(match)
                if atom_set.isdisjoint(assigned_atoms):
                    groups.append(atom_set)
                    assigned_atoms.update(atom_set)

        return groups

    def _select_functional_group_atoms(self, data, functional_group_options=None):
        groups = functional_group_options
        if groups is None:
            groups = self._functional_group_atom_sets(data)
        if not groups:
            return None
        return random.choice(groups)

    def global_augmentation(self, data):
        """Create a global augmented view by copying the graph structure."""
        return Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
            **self._copy_optional_metadata(data),
            view=torch.tensor([1], dtype=torch.long),  # 1 = global
            # Nothing is masked in a global view -- present on every view type
            # (masked or not) purely so batching sees a uniform attribute
            # across the whole multi-crop batch. Only consumed when
            # use_node_reconstruction_loss is set.
            node_reconstruction_mask=torch.zeros(data.x.size(0), dtype=torch.bool),
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
            **self._copy_optional_metadata(data),
            view=torch.tensor([0], dtype=torch.long),  # 0 = local
            # k-hop views are a strict subgraph, not a masked-in-place copy --
            # nothing here is "masked", it's simply excluded, so there's no
            # node to reconstruct. See global_augmentation for why this is
            # attached unconditionally.
            node_reconstruction_mask=torch.zeros(new_x.size(0), dtype=torch.bool),
        )

    def functional_group_local_augmentation(self, data, num_hops=2, functional_group_options=None):
        """Create a local view centered on a detected functional group."""
        num_atoms = data.x.size(0)
        if num_atoms == 0:
            return data

        selected_atoms = self._select_functional_group_atoms(data, functional_group_options)
        if not selected_atoms:
            return self.local_augmentation(data, num_hops=num_hops)

        subset, new_edge_index, _, edge_mask = k_hop_subgraph(
            node_idx=torch.tensor(sorted(selected_atoms), dtype=torch.long),
            num_hops=num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=num_atoms,
        )

        new_x = data.x[subset]
        new_edge_attr = None
        if data.edge_attr is not None:
            new_edge_attr = data.edge_attr[edge_mask]

        return Data(
            x=new_x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            **self._copy_optional_metadata(data),
            view=torch.tensor([0], dtype=torch.long),
            # Same reasoning as local_augmentation -- a k-hop subgraph, nothing masked.
            node_reconstruction_mask=torch.zeros(new_x.size(0), dtype=torch.bool),
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
        node_reconstruction_mask = torch.zeros(num_nodes, dtype=torch.bool)
        if node_mask_indices:
            masked_node_idx = list(node_mask_indices)
            masked_x[masked_node_idx] = 0
            node_reconstruction_mask[masked_node_idx] = True

        num_feature_mask = self._sample_count(num_features, feature_mask_ratio)
        feature_mask_indices = random.sample(range(num_features), num_feature_mask) if num_feature_mask > 0 else []
        if remaining_indices and feature_mask_indices:
            for node_idx in remaining_indices:
                masked_x[node_idx, feature_mask_indices] = 0

        return Data(
            x=masked_x,
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
            **self._copy_optional_metadata(data),
            view=torch.tensor([0], dtype=torch.long),  # 0 = local, keep compatibility with existing code
            # Node order/count is preserved exactly (masked_x is a clone with
            # rows zeroed in place, never reindexed) -- so position i here is
            # position i in this same molecule's global view too. That 1:1
            # alignment is what makes the masked-node reconstruction loss's
            # student/teacher correspondence possible without any separate
            # node-id bookkeeping.
            node_reconstruction_mask=node_reconstruction_mask,
        )

    def functional_group_masked_local_augmentation(
        self, data, node_mask_ratio=0.15, feature_mask_ratio=0.15, functional_group_options=None
    ):
        """Create a local view by masking a detected functional group and nearby atoms."""
        if data.x is None:
            return self.global_augmentation(data)

        num_nodes, num_features = data.x.size()
        if num_nodes == 0 or num_features == 0:
            return self.global_augmentation(data)

        selected_atoms = self._select_functional_group_atoms(data, functional_group_options)
        if not selected_atoms:
            return self.masked_local_augmentation(data, node_mask_ratio=node_mask_ratio, feature_mask_ratio=feature_mask_ratio)

        node_mask_indices = set(selected_atoms)
        num_node_mask = self._sample_count(num_nodes, node_mask_ratio)

        if num_node_mask > len(node_mask_indices):
            neighborhood = set()
            for src, dst in data.edge_index.t().tolist():
                if src in node_mask_indices and dst not in node_mask_indices:
                    neighborhood.add(dst)
                elif dst in node_mask_indices and src not in node_mask_indices:
                    neighborhood.add(src)

            candidate_indices = list(neighborhood) if neighborhood else [idx for idx in range(num_nodes) if idx not in node_mask_indices]
            extra_mask_count = min(len(candidate_indices), num_node_mask - len(node_mask_indices))
            if extra_mask_count > 0:
                node_mask_indices.update(random.sample(candidate_indices, extra_mask_count))

        remaining_indices = [idx for idx in range(num_nodes) if idx not in node_mask_indices]

        masked_x = data.x.clone()
        node_reconstruction_mask = torch.zeros(num_nodes, dtype=torch.bool)
        if node_mask_indices:
            masked_node_idx = list(node_mask_indices)
            masked_x[masked_node_idx] = 0
            node_reconstruction_mask[masked_node_idx] = True

        num_feature_mask = self._sample_count(num_features, feature_mask_ratio)
        feature_mask_indices = random.sample(range(num_features), num_feature_mask) if num_feature_mask > 0 else []
        if remaining_indices and feature_mask_indices:
            for node_idx in remaining_indices:
                masked_x[node_idx, feature_mask_indices] = 0

        return Data(
            x=masked_x,
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
            **self._copy_optional_metadata(data),
            view=torch.tensor([0], dtype=torch.long),
            # Same node-order guarantee as masked_local_augmentation above.
            node_reconstruction_mask=node_reconstruction_mask,
        )

    def __call__(self, data):
        """Generate multiple augmented views of the input graph."""
        aug_data_list = []

        # Create two global views
        aug_data_list.append(self.global_augmentation(data))
        aug_data_list.append(self.global_augmentation(data))

        # Per-view mode override (see __init__) -- defaults to using
        # local_augmentation_mode uniformly for every view, exactly as before.
        view_modes = self.local_view_modes if self.local_view_modes is not None else [self.local_augmentation_mode] * self.local_views

        # Functional-group detection re-parses the SMILES and matches 12 SMARTS
        # patterns; compute it once per graph and reuse across all local views
        # instead of redoing it for every view that needs it.
        functional_group_options = None
        if any(mode in ("functional_group_masking", "functional_group_k_hop") for mode in view_modes):
            functional_group_options = self._functional_group_atom_sets(data)

        # Create local views according to each view's configured strategy
        for mode in view_modes:
            if mode == "masking":
                aug_data_list.append(
                    self.masked_local_augmentation(
                        data,
                        node_mask_ratio=self.node_mask_ratio,
                        feature_mask_ratio=self.feature_mask_ratio,
                    )
                )
            elif mode == "functional_group_masking":
                aug_data_list.append(
                    self.functional_group_masked_local_augmentation(
                        data,
                        node_mask_ratio=self.node_mask_ratio,
                        feature_mask_ratio=self.feature_mask_ratio,
                        functional_group_options=functional_group_options,
                    )
                )
            elif mode == "k_hop":
                aug_data_list.append(self.local_augmentation(data, num_hops=self.k_hops))
            elif mode == "functional_group_k_hop":
                aug_data_list.append(
                    self.functional_group_local_augmentation(
                        data, num_hops=self.k_hops, functional_group_options=functional_group_options
                    )
                )
            else:
                raise ValueError(f"Unsupported local_augmentation_mode: {mode}")

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
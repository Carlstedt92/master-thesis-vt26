"""From GNN for CHemists repo"""
from rdkit import Chem
from torch_geometric.data import Data
import torch
import numpy as np

def smiles_to_pygdata(smiles: str):
    """ Convert a SMILES string to a graph representation suitable for GNNs.

    Args:
        smiles (str): The SMILES string representing the molecule.

    Returns:
        Data: A PyTorch Geometric Data object containing node features, edge indices, and edge attributes,
                along with the original SMILES string as metadata.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    mol = Chem.AddHs(mol)

    # Define mapping for bond types to incides for one-hot encoding
    bond_type_mapping = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }

    # Extract atom features
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            atom.GetIsAromatic(),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons()
        ]

        # One hot encode the atom symbol
        atom_types = ['C', 'O', 'N', 'H', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        atom_type_onehot = [1 if atom.GetSymbol() == t else 0 for t in atom_types]
        if atom.GetSymbol() not in atom_types:
            atom_type_onehot.append(1)  # Other atom type
        else:
            atom_type_onehot.append(0)  # Not other atom type

        # Create one hot encoding for hybridization
        hybridization_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
        ]
        hybridization_onehot = [1 if atom.GetHybridization() == h else 0 for h in hybridization_types]
        if atom.GetHybridization() not in hybridization_types:
            hybridization_onehot.append(1)  # Other hybridization type
        else:
            hybridization_onehot.append(0)  # Not other hybridization type

        features = atom_type_onehot + features + hybridization_onehot
        node_features.append(features)

    node_features = torch.tensor(node_features)

    # Extract bond information
    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        # Add edges in both directions for undirected graph
        edge_index.append([start_idx, end_idx])
        edge_index.append([end_idx, start_idx])

        bond_type = bond.GetBondType()
        bond_type_onehot = np.zeros(len(bond_type_mapping))
        if bond_type in bond_type_mapping:
            bond_type_onehot[bond_type_mapping[bond_type]] = 1
        
        # Extract additional properties
        is_conjugated = int(bond.GetIsConjugated())
        is_in_ring = int(bond.IsInRing())

        # Combine bond type one-hot encoding with additional properties
        features = np.concatenate([bond_type_onehot, [is_conjugated, is_in_ring]])

        # Add features for both directions
        edge_features.append(features)
        edge_features.append(features)
    
    # Convert edge features to numpy array, handling empty case
    if edge_features:
        edge_features = np.array(edge_features)
    else:
        edge_features = np.empty((0, len(bond_type_mapping) + 2))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    return Data(x= node_features, edge_index=edge_index, edge_attr=edge_features)


# Example usage:
if __name__ == "__main__":
    smiles = "CCO"
    data = smiles_to_pygdata(smiles)
    print(data)
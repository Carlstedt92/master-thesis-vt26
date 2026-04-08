"""From GNN for CHemists repo"""
from rdkit import Chem
from torch_geometric.data import Data
import torch
import numpy as np
from typing import Optional

def smiles_to_pygdata(smiles: str) -> Optional[Data]:
    """ Convert a SMILES string to a graph representation suitable for GNNs.

    Args:
        smiles (str): The SMILES string representing the molecule.

    Returns:
        Data: A PyTorch Geometric Data object containing 21 node features, edge indices, and 6 edge attributes,
                along with the original SMILES string as metadata.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    
    # Assign stereochemistry (important for GetChiralTag and GetStereo)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Define mapping for bond types to incides for one-hot encoding
    bond_type_mapping = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }
    
    # Define mapping for bond stereo types
    bond_stereo_mapping = {
        Chem.rdchem.BondStereo.STEREONONE: 0,
        Chem.rdchem.BondStereo.STEREOANY: 1,
        Chem.rdchem.BondStereo.STEREOZ: 2,
        Chem.rdchem.BondStereo.STEREOE: 3,
        Chem.rdchem.BondStereo.STEREOCIS: 4,
        Chem.rdchem.BondStereo.STEREOTRANS: 5,
    }

    # Extract atom features
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetIsAromatic(),
            atom.GetNumRadicalElectrons(),
            int(atom.IsInRing()),  # Whether atom is in a ring
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

        # Create one hot encoding for chiral tag
        chiral_tags = [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_OTHER,
        ]
        chiral_tag_onehot = [1 if atom.GetChiralTag() == c else 0 for c in chiral_tags]

        features = atom_type_onehot + features + hybridization_onehot + chiral_tag_onehot
        node_features.append(features)

    # Keep node features as float for stable collation across all molecules.
    if node_features:
        node_features = torch.tensor(node_features, dtype=torch.float)
    else:
        node_features = torch.empty((0, 24), dtype=torch.float)

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
        
        # Create one hot encoding for bond stereochemistry
        bond_stereo = bond.GetStereo()
        bond_stereo_onehot = np.zeros(len(bond_stereo_mapping))
        if bond_stereo in bond_stereo_mapping:
            bond_stereo_onehot[bond_stereo_mapping[bond_stereo]] = 1
        else:
            bond_stereo_onehot[0] = 1  # Default to STEREONONE for unknown types

        # Combine bond type one-hot encoding with additional properties
        features = np.concatenate([bond_type_onehot, [is_conjugated, is_in_ring], bond_stereo_onehot])

        # Add features for both directions
        edge_features.append(features)
        edge_features.append(features)
    
    # Convert edge features to numpy array, handling empty case
    if edge_features:
        edge_features = np.array(edge_features)
    else:
        edge_features = np.empty((0, len(bond_type_mapping) + 2 + len(bond_stereo_mapping)))

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    return Data(x= node_features, edge_index=edge_index, edge_attr=edge_features)


# Example usage:
if __name__ == "__main__":
    smiles = "CCO"
    data = smiles_to_pygdata(smiles)
    if data is not None:
        print(data.x.shape)  # Node features shape
        print(data.edge_index.shape)  # Edge indices shape
        print(data.edge_attr.shape)  # Edge attributes shape
        print(data)
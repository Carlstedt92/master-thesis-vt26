"""From GNN for CHemists repo"""
import math
import os
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from torch_geometric.data import Data
import torch
import numpy as np
from typing import Optional

_ECCENTRICITY_SCALER_PATH = os.path.join(os.path.dirname(__file__), "eccentricity_scaler.joblib")
_eccentricity_scaler = None


def _get_eccentricity_scaler():
    """Lazily load the sklearn StandardScaler fit on the ZINC SSL-training
    distribution (see fit_eccentricity_scaler.py) -- eccentricity is the only
    extended feature with unbounded, molecule-size-dependent raw magnitude
    (unlike the other 25 features, all binary or small bounded values), so it
    gets scaled the same way for both ZINC precompute and live SMILES
    featurization (online eval, knn eval, downstream datasets)."""
    global _eccentricity_scaler
    if _eccentricity_scaler is None:
        import joblib
        _eccentricity_scaler = joblib.load(_ECCENTRICITY_SCALER_PATH)
    return _eccentricity_scaler


def smiles_to_pygdata(
    smiles: str,
    explicit_hydrogens: bool = True,
    encode_hydrogen_count: bool = False,
    use_extended_features: bool = False,
    scale_eccentricity: bool = False,
) -> Optional[Data]:
    """Convert a SMILES string to a graph representation suitable for GNNs.

    Args:
        smiles (str): The SMILES string representing the molecule.
        use_extended_features: append 2 additional node features -- Gasteiger
            partial charge and topological eccentricity (see below). Opt-in
            and off by default so existing 24-feature models/configs are
            unaffected; a model trained with this on needs num_features=26
            (or 27 with encode_hydrogen_count) and cannot load 24-feature
            checkpoints (the first Linear layer's shape depends on it).

    Returns:
        Data: A PyTorch Geometric Data object with:
            - 24 node features per atom (26 with use_extended_features):
                - 11 atom-type flags: C, O, N, H, F, P, S, Cl, Br, I, other
                - 5 atom properties: degree, formal charge, aromaticity, radical electrons, ring membership
                - 4 hybridization flags: SP, SP2, SP3, other
                - 4 chirality flags: unspecified, tetrahedral CCW, tetrahedral CW, other
                - [extended] Gasteiger partial charge: empirical estimate of real
                  (fractional) electron distribution -- unlike formal charge
                  (almost always 0), this varies continuously with each atom's
                  electronegative environment and is relevant to polarity/
                  reactivity-driven properties (lipophilicity, binding, toxicity).
                - [extended] topological eccentricity: this atom's longest
                  shortest-path (bond-count) distance to any other atom in the
                  molecule -- a cheap proxy for "how peripheral vs. central is
                  this atom", giving every atom some awareness of overall
                  molecular size/shape without requiring more message-passing
                  hops than the encoder's fixed depth allows.
            - 12 edge features per bond direction:
                - 4 bond-type flags: single, double, triple, aromatic
                - 2 bond properties: conjugated, in ring
                - 6 bond-stereo flags: none, any, Z, E, cis, trans
            - edge_index describing the bidirectional molecular graph
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)

    # Assign stereochemistry (important for GetChiralTag and GetStereo)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    if use_extended_features:
        # Computed once per molecule (not per atom) -- both are O(atoms) or
        # O(atoms^2) whole-molecule operations.
        rdPartialCharges.ComputeGasteigerCharges(mol)
        # Bond-count shortest-path distance between every atom pair. Disconnected
        # fragments (salts, e.g. "CCO.Cl") get RDKit's large sentinel distance
        # rather than inf; the isfinite guards below catch that same as NaN/inf.
        distance_matrix = Chem.GetDistanceMatrix(mol)
        if scale_eccentricity:
            eccentricity_scaler = _get_eccentricity_scaler()

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
    num_atoms = mol.GetNumAtoms()
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
        if encode_hydrogen_count:
            features.append(float(atom.GetTotalNumHs()))
        if use_extended_features:
            gasteiger_charge = atom.GetDoubleProp("_GasteigerCharge")
            if not math.isfinite(gasteiger_charge):
                gasteiger_charge = 0.0
            # RDKit's distance matrix uses a large finite sentinel (1e8), not
            # inf/NaN, for atom pairs in disconnected fragments (e.g. salts
            # like "CC(=O)O.[Na]") -- isfinite() doesn't catch that. Any real
            # topological distance within a connected fragment is bounded by
            # the molecule's own atom count, so clamp there; this is a no-op
            # for every ordinary (connected) molecule.
            eccentricity = float(distance_matrix[atom.GetIdx()].max())
            if not math.isfinite(eccentricity) or eccentricity > num_atoms:
                eccentricity = float(num_atoms)
            if scale_eccentricity:
                # StandardScaler fit on the ZINC SSL-training distribution
                # (see fit_eccentricity_scaler.py); applied inline as
                # (x - mean) / std -- identical to scaler.transform() but
                # avoids per-atom call overhead across millions of molecules.
                # Gated separately from use_extended_features: checkpoints
                # trained before this scaler existed (e.g.
                # EXTENDED_FEATURES_TEST_60EP) expect raw eccentricity, and
                # must keep getting it at eval time too.
                eccentricity = (eccentricity - eccentricity_scaler.mean_[0]) / eccentricity_scaler.scale_[0]
            features.append(gasteiger_charge)
            features.append(eccentricity)
        node_features.append(features)

    # Keep node features as float for stable collation across all molecules.
    node_dim = 24 + (1 if encode_hydrogen_count else 0) + (2 if use_extended_features else 0)
    if node_features:
        node_features = torch.tensor(node_features, dtype=torch.float)
    else:
        node_features = torch.empty((0, node_dim), dtype=torch.float)

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

    # Keep the raw input SMILES as lightweight metadata for chemistry-aware augmentations.
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, smiles=smiles)


# Example usage:
if __name__ == "__main__":
    smiles = "CCO"
    data = smiles_to_pygdata(smiles)
    if data is not None:
        print(data.x.shape)  # Node features shape
        print(data.edge_index.shape)  # Edge indices shape
        print(data.edge_attr.shape)  # Edge attributes shape
        print(data)
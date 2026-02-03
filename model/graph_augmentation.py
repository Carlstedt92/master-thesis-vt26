
import random
import torch
from chemprop import data
from rdkit import Chem

class GraphAugmentation:
    def __init__(self, local_views=4):
        self.local_views = local_views

    def global_augmentation(self, mol):
        mol = Chem.Mol(mol)  # Create a copy of the molecule
        return mol  # No augmentation for global view yet, discuss what/if any should be implemented

    def local_augmentation(self, mol):
        mol = Chem.Mol(mol)  # Create a copy of the molecule
        # Create augmentated molecule from RDKit molecule by selecting a random atom and its 2hop-neighbors
        atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
        if not atom_indices:
            return mol  # Return original molecule if no atoms are present
        
        center_atom_idx = random.choice(atom_indices)
        aug_indices = set([center_atom_idx])
        for nbr in mol.GetAtomWithIdx(center_atom_idx).GetNeighbors():
            aug_indices.add(nbr.GetIdx())
            for nbr2 in nbr.GetNeighbors():
                aug_indices.add(nbr2.GetIdx())
        aug_indices = list(aug_indices)
        aug_mol = data.MoleculeDatapoint._get_submol(mol, aug_indices)
        return aug_mol

    def __call__(self, mol):
        aug_mol = []

        # Create two global views
        aug_mol.append(self.global_augmentation(mol))
        aug_mol.append(self.global_augmentation(mol))
        # Create local views
        for _ in range(self.local_views):
            aug_mol.append(self.local_augmentation(mol))
        return aug_mol
    
class DINOMoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        mol, _ = self.base_dataset[idx]
        aug_mol = self.transform(mol)
        return aug_mol
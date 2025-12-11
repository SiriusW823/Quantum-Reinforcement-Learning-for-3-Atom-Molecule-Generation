from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def _atom_counts(mol: Chem.Mol, atoms: Sequence[str]) -> np.ndarray:
    counts = np.zeros(len(atoms), dtype=np.float32)
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        if sym in atoms:
            counts[atoms.index(sym)] += 1.0
    return counts


def _bond_counts(mol: Chem.Mol) -> np.ndarray:
    singles = doubles = triples = 0.0
    for b in mol.GetBonds():
        order = b.GetBondType()
        if order == Chem.BondType.SINGLE:
            singles += 1
        elif order == Chem.BondType.DOUBLE:
            doubles += 1
        elif order == Chem.BondType.TRIPLE:
            triples += 1
    return np.array([singles, doubles, triples], dtype=np.float32)


def smiles_to_features(smiles: str, atoms: Sequence[str] | None = None, n_bits: int = 32) -> torch.Tensor:
    """
    Convert SMILES to a normalized feature vector:
    - atom counts (C,N,O default)
    - bond order counts
    - short Morgan fingerprint slice
    """
    atoms = atoms or ["C", "N", "O"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(len(atoms) + 3 + n_bits, dtype=torch.float32)

    atom_vec = _atom_counts(mol, atoms)
    bond_vec = _bond_counts(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)

    # normalize counts by simple max to keep angles in reasonable range
    atom_vec_norm = atom_vec / max(1.0, atom_vec.sum())
    bond_vec_norm = bond_vec / max(1.0, bond_vec.sum())
    feat = np.concatenate([atom_vec_norm, bond_vec_norm, arr.astype(np.float32)], axis=0)
    return torch.tensor(feat, dtype=torch.float32)

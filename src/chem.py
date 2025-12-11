from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import random
import torch

from rdkit import Chem
from rdkit.Chem import rdchem, AllChem

BondType = rdchem.BondType


@dataclass
class SampledMolecule:
    atoms: Tuple[int, ...]
    bonds: Tuple[int, ...]
    smiles: Optional[str]
    valid: float


BOND_TYPES: Sequence[Optional[BondType]] = (
    None,
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
)
BOND_LABELS = {
    None: "NONE",
    BondType.SINGLE: "S",
    BondType.DOUBLE: "D",
    BondType.TRIPLE: "T",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def atoms_bonds_to_smiles(atom_ids: Sequence[int], bond_ids: Sequence[int], allowed_atoms: Sequence[str]) -> Tuple[Optional[str], float]:
    """Build a chain molecule and return canonical SMILES and validity flag."""
    mol = Chem.RWMol()
    try:
        atom_indices = []
        for atom_id in atom_ids:
            symbol = allowed_atoms[atom_id]
            atom_indices.append(mol.AddAtom(Chem.Atom(symbol)))
        for i, bond_id in enumerate(bond_ids):
            a_idx, b_idx = atom_indices[i], atom_indices[i + 1]
            bond_type = BOND_TYPES[bond_id]
            if bond_type is None:
                continue
            mol.AddBond(a_idx, b_idx, bond_type)
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return smiles, 1.0
    except Exception:
        return None, 0.0


def embed_geometry(smiles: str, min_dist2: float = 1e-3) -> Optional[List[Tuple[str, Tuple[float, float, float]]]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol, asMols=True)) > 1:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        return None
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    conf = mol.GetConformer()
    geom: List[Tuple[str, Tuple[float, float, float]]] = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        geom.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))
    coords = [c for _, c in geom]
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dz = coords[i][2] - coords[j][2]
            if dx * dx + dy * dy + dz * dz < min_dist2:
                return None
    return geom


def estimate_qubits(num_atoms: int) -> int:
    return 2 * num_atoms + 2 * (num_atoms * (num_atoms - 1) // 2)

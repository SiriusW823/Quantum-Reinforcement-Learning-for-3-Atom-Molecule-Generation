from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
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
BOND_ORDER = {
    None: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
}
ALLOWED_VALENCE: Dict[str, int] = {"C": 4, "N": 3, "O": 2}


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


def bond_matrix_from_smiles(smiles: str) -> Optional[List[List[int]]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    mat = [[0 for _ in range(n)] for _ in range(n)]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        order = int(bond.GetBondTypeAsDouble())
        mat[i][j] = mat[j][i] = order
    return mat


def valence_usage(smiles: str) -> Optional[List[int]]:
    mat = bond_matrix_from_smiles(smiles)
    if mat is None:
        return None
    n = len(mat)
    usage = [0 for _ in range(n)]
    for i in range(n):
        usage[i] = sum(mat[i])
    return usage


def connectivity_components(smiles: str) -> int:
    mat = bond_matrix_from_smiles(smiles)
    if mat is None:
        return 0
    n = len(mat)
    visited = [False] * n
    comp = 0

    def dfs(u: int) -> None:
        visited[u] = True
        for v, w in enumerate(mat[u]):
            if w > 0 and not visited[v]:
                dfs(v)

    for i in range(n):
        if not visited[i]:
            comp += 1
            dfs(i)
    return comp


def distance_matrix(geom: List[Tuple[str, Tuple[float, float, float]]]) -> List[List[float]]:
    n = len(geom)
    dmat = [[0.0 for _ in range(n)] for _ in range(n)]
    coords = [c for _, c in geom]
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dz = coords[i][2] - coords[j][2]
            d = (dx * dx + dy * dy + dz * dz) ** 0.5
            dmat[i][j] = dmat[j][i] = d
    return dmat

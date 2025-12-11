from dataclasses import dataclass
from typing import List, Tuple, Dict
from rdkit import Chem

ATOM_TYPES = ["NONE", "C", "N", "O"]
BOND_TYPES = ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]
MAX_ATOMS = 5


@dataclass
class StepResult:
    state: List[int]
    done: bool
    info: Dict


class MoleculeEnv:
    """
    Minimal gym-like environment for linear 5-atom generation:
    Atom1 -> Bond1 -> Atom2 -> Bond2 -> Atom3 -> Bond3 -> Atom4 -> Bond4 -> Atom5
    """

    def __init__(self):
        self.reset()

    def reset(self) -> List[int]:
        self.history: List[int] = []  # tokens length up to 9
        self.done = False
        return self.history

    def step(self, action: int) -> StepResult:
        if self.done:
            return StepResult(self.history, True, {})
        self.history.append(action)
        if len(self.history) >= 9:
            self.done = True
        return StepResult(self.history, self.done, {})

    def finalize(self) -> Tuple[str | None, float, float]:
        """
        Build SMILES from the sequence. Returns (smiles, valid, unique_flag_placeholder).
        unique is handled externally; here we return 0.0 as placeholder.
        """
        # Parse history into atoms/bonds, apply NONE truncation
        atoms_ids = [self.history[i] if i < len(self.history) else 0 for i in range(0, 9, 2)]  # positions 0,2,4,6,8
        bonds_ids = [self.history[i] if i < len(self.history) else 0 for i in range(1, 9, 2)]  # positions 1,3,5,7

        # Truncate on NONE atom
        final_atoms: List[int] = []
        final_bonds: List[int] = []
        for idx, a in enumerate(atoms_ids):
            if a == 0:  # NONE
                break
            final_atoms.append(a)
            if idx < len(bonds_ids):
                final_bonds.append(bonds_ids[idx] if a != 0 else 0)
        # Ensure bond list length is atoms-1
        final_bonds = final_bonds[: max(0, len(final_atoms) - 1)]
        if not final_atoms:
            return None, 0.0, 0.0

        smiles, valid = self._atoms_bonds_to_smiles(final_atoms, final_bonds)
        return smiles, valid, 0.0

    def _atoms_bonds_to_smiles(self, atoms: List[int], bonds: List[int]) -> Tuple[str | None, float]:
        mol = Chem.RWMol()
        try:
            atom_indices = []
            for a in atoms:
                atom_indices.append(mol.AddAtom(Chem.Atom(ATOM_TYPES[a])))
            for i, b in enumerate(bonds):
                if b == 0:
                    continue
                bt = {
                    1: Chem.BondType.SINGLE,
                    2: Chem.BondType.DOUBLE,
                    3: Chem.BondType.TRIPLE,
                }[b]
                mol.AddBond(atom_indices[i], atom_indices[i + 1], bt)
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)
            if "." in smiles:
                return None, 0.0
            return smiles, 1.0
        except Exception:
            return None, 0.0

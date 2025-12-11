from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from env.chem import atoms_bonds_to_smiles


class MoleculeEnvironment:
    """
    Simple one-step environment that turns a full atom/bond specification
    into a SMILES string using RDKit utilities and keeps running validity
    and uniqueness statistics.

    This remains classical: it only wraps RDKit-based construction and counters.
    """

    def __init__(self, allowed_atoms: Sequence[str], enforce_connected: bool = True):
        self.allowed_atoms = list(allowed_atoms)
        self.enforce_connected = enforce_connected
        self.reset_counters()

    def reset_counters(self) -> None:
        self.samples = 0
        self.valid_count = 0
        self.unique_valid_count = 0
        self.seen: set[str] = set()

    def build_from_actions(
        self, atoms: Iterable[int], bonds: Iterable[int]
    ) -> Tuple[str | None, float, float]:
        """
        Build molecule from discrete atom/bond choices.
        Returns (smiles, valid_flag, unique_flag).
        """
        smiles, valid = atoms_bonds_to_smiles(tuple(atoms), tuple(bonds), self.allowed_atoms)
        if valid and smiles and self.enforce_connected and "." in smiles:
            valid = 0.0
        unique = 0.0
        self.samples += 1
        if valid:
            self.valid_count += 1
            unique = 1.0 if smiles and smiles not in self.seen else 0.0
            if unique and smiles:
                self.unique_valid_count += 1
                self.seen.add(smiles)
        return smiles, float(valid), float(unique)

    @property
    def valid_ratio(self) -> float:
        return self.valid_count / self.samples if self.samples else 0.0

    @property
    def unique_ratio(self) -> float:
        return self.unique_valid_count / self.samples if self.samples else 0.0

    @property
    def target_metric(self) -> float:
        return self.valid_ratio * self.unique_ratio

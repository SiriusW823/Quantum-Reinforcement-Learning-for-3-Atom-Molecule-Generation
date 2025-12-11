from typing import List, Tuple
import torch
import torch.nn as nn
import pennylane as qml
from quantum.pqc_blocks import make_torch_pqc


class QuantumPolicyNet(nn.Module):
    """
    Quantum policy network using PennyLane QNode.
    - AngleEmbedding on 6 wires (enough for logits projection).
    - StronglyEntanglingLayers as ansatz.
    - A small classical projection head maps expvals to logits for atoms/bonds.
    """

    def __init__(self, n_atoms: int, n_bonds: int, num_atoms_in_chain: int, n_wires: int = 6, layers: int = 2):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_bonds = n_bonds
        self.num_atoms_in_chain = num_atoms_in_chain
        self.num_bonds_in_chain = num_atoms_in_chain - 1
        self.n_wires = n_wires
        self.pqc = make_torch_pqc(n_wires=n_wires, layers=layers)
        self.head_atoms = nn.ModuleList([nn.Linear(self.n_wires, n_atoms) for _ in range(self.num_atoms_in_chain)])
        self.head_bonds = nn.ModuleList([nn.Linear(self.n_wires, n_bonds) for _ in range(self.num_bonds_in_chain)])

    def forward(
        self, angles: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[List[torch.distributions.Categorical], List[torch.distributions.Categorical]]:
        """
        angles: tensor of shape (n_wires,) produced from e.g., dummy constant or time step; here we use zeros.
        """
        expvals = self.pqc(angles).float()  # shape (n_wires,)
        atom_logits = [head(expvals) / temperature for head in self.head_atoms]
        bond_logits = [head(expvals) / temperature for head in self.head_bonds]
        atom_dists = [torch.distributions.Categorical(logits=logits) for logits in atom_logits]
        bond_dists = [torch.distributions.Categorical(logits=logits) for logits in bond_logits]
        return atom_dists, bond_dists


def mask_bond_logits(bond_logits: torch.Tensor, atom_i: int, atom_j: int, allowed_atoms: List[str], max_order_map: dict) -> torch.Tensor:
    """Mask bond logits based on valence (prevent orders above allowed)."""
    max_order = min(max_order_map[allowed_atoms[atom_i]], max_order_map[allowed_atoms[atom_j]])
    mask = torch.zeros_like(bond_logits)
    for idx, order in enumerate([0, 1, 2, 3]):
        if order > max_order:
            mask[idx] = -1e9
    return bond_logits + mask


def sample_action(
    policy: QuantumPolicyNet, allowed_atoms: List[str], temperature: float = 1.0
) -> Tuple[Tuple[int, ...], Tuple[int, ...], torch.Tensor, torch.Tensor]:
    max_order_map = {"C": 3, "N": 3, "O": 2}  # limit bonds: O no triple
    angles = torch.zeros(policy.n_wires, dtype=torch.float32)
    atom_dists, bond_dists = policy.forward(angles, temperature=temperature)
    atoms = []
    bonds = []
    log_probs = []
    entropies = []

    # sample atoms first
    for dist in atom_dists:
        a = dist.sample()
        atoms.append(a.item())
        log_probs.append(dist.log_prob(a))
        entropies.append(dist.entropy())

    # sample bonds with masking
    for idx, dist in enumerate(bond_dists):
        i, j = idx, idx + 1
        logits = dist.logits
        masked_logits = mask_bond_logits(logits, atoms[i], atoms[j], allowed_atoms, max_order_map)
        masked_dist = torch.distributions.Categorical(logits=masked_logits)
        b = masked_dist.sample()
        bonds.append(b.item())
        log_probs.append(masked_dist.log_prob(b))
        entropies.append(masked_dist.entropy())

    total_log_prob = torch.stack(log_probs).sum()
    total_entropy = torch.stack(entropies).sum()
    return tuple(atoms), tuple(bonds), total_log_prob, total_entropy

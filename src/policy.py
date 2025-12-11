from typing import List, Tuple
import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """Factorized policy over atom choices and bond types."""

    def __init__(self, n_atoms: int, n_bonds: int, num_atoms_in_chain: int, hidden: int = 128):
        super().__init__()
        self.num_atoms_in_chain = num_atoms_in_chain
        self.num_bonds_in_chain = num_atoms_in_chain - 1
        self.shared = nn.Sequential(nn.Linear(1, hidden), nn.ReLU())
        self.atom_heads = nn.ModuleList([nn.Linear(hidden, n_atoms) for _ in range(self.num_atoms_in_chain)])
        self.bond_heads = nn.ModuleList([nn.Linear(hidden, n_bonds) for _ in range(self.num_bonds_in_chain)])

    def distributions(
        self, temperature: float = 1.0
    ) -> Tuple[List[torch.distributions.Categorical], List[torch.distributions.Categorical]]:
        x = self.shared(torch.ones(1, 1))
        atom_dists = [torch.distributions.Categorical(logits=head(x) / temperature) for head in self.atom_heads]
        bond_dists = [torch.distributions.Categorical(logits=head(x) / temperature) for head in self.bond_heads]
        return atom_dists, bond_dists


def sample_action(policy: PolicyNet, temperature: float = 1.0) -> Tuple[Tuple[int, ...], Tuple[int, ...], torch.Tensor, torch.Tensor]:
    atom_dists, bond_dists = policy.distributions(temperature=temperature)
    atoms = []
    bonds = []
    log_probs = []
    entropies = []
    for dist in atom_dists:
        atom = dist.sample()
        atoms.append(atom.item())
        log_probs.append(dist.log_prob(atom))
        entropies.append(dist.entropy())
    for dist in bond_dists:
        bond = dist.sample()
        bonds.append(bond.item())
        log_probs.append(dist.log_prob(bond))
        entropies.append(dist.entropy())
    total_log_prob = torch.stack(log_probs).sum()
    total_entropy = torch.stack(entropies).sum()
    return tuple(atoms), tuple(bonds), total_log_prob, total_entropy

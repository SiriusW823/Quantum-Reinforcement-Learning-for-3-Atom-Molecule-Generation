"""
Quantum RL for 3-atom heavy-atom molecules.

- Samples 3 heavy atoms and 2 bonds (1-2, 2-3), converts to SMILES with RDKit.
- Reward = validity * uniqueness * quantum_prior (default PennyLane toy circuit; replace with real chemistry model), targeting 1.0 for valid+novel+good prior.
- Policy optimized with REINFORCE. Quantum prior hook (PennyLane/cudaq/QMG) is pluggable where noted.

Dependencies (install before running):
  conda create -n qmg python=3.11 rdkit -c conda-forge -y
  conda activate qmg
  pip install torch  # and optionally pennylane or cuda-quantum for quantum priors
  pip install pennylane
"""
from __future__ import annotations

import hashlib
import math
import os
import itertools
import random
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np

# Avoid OpenMP runtime conflicts (libomp vs libiomp) often seen on Windows with PyTorch/RDKit.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")  # silence valence warnings in stdout; we handle validity explicitly
except ImportError as exc:
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    raise SystemExit(
        "RDKit is required but not installed for your Python version.\n"
        f"Detected Python {py_ver}. On Windows, install RDKit via conda-forge "
        "with a supported Python (e.g., 3.10/3.11):\n"
        "  conda create -n qmg python=3.11 rdkit -c conda-forge -y\n"
        "  conda activate qmg\n"
        "  pip install torch\n"
        "then re-run this script inside that environment."
    ) from exc

# Allowed heavy atoms; hydrogens are implicit. Kept small for focused training.
ALLOWED_ATOMS: Sequence[str] = (
    "C",
    "N",
    "O",
)
# Bond types considered between atom1-atom2 and atom2-atom3; None means no bond.
BOND_TYPES: Sequence[Optional[rdchem.BondType]] = (
    None,
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
)
BOND_LABELS = {
    None: "NONE",
    rdchem.BondType.SINGLE: "S",
    rdchem.BondType.DOUBLE: "D",
    rdchem.BondType.TRIPLE: "T",
}


def build_qmg_prior() -> Optional[Callable[[str], float]]:
    """
    Optional: real QMG prior using CUDA-Q/QMG workflow.
    Fill this out with your QMG pipeline (see PEESEgroup/QMG and Squirtle007/CUDA-Q notebooks).

    Expected contract: fn(smiles: str) -> float (higher is better, e.g., -energy rescaled >= 0).
    Keep failures safe by returning None so the RL falls back to the default proxy.
    """
    try:
        import cudaq  # type: ignore
        # from qmg import ...  # import your QMG model/energy utilities here
    except Exception:
        return None

    def qmg_prior(smiles: str) -> float:
        # TODO: replace with your QMG circuit construction and energy estimation.
        # Example pseudocode (to be implemented with your assets):
        #   hamiltonian = smiles_to_hamiltonian(smiles)           # your mapping
        #   result = cudaq.observe(your_param_circuit, hamiltonian)
        #   energy = result.expectation()
        #   return max(0.0, -energy)  # lower energy -> higher score
        return 0.0  # placeholder until integrated

    return qmg_prior


def build_pennylane_prior() -> Optional[Callable[[str], float]]:
    """
    PennyLane prior that maps SMILES -> score via a simplified quantum chemistry VQE.
    This uses RDKit -> 3D geom -> PennyLane qchem (PySCF) to build a molecular Hamiltonian
    with a small active space, then runs a short VQE (UCCSD) to obtain an energy proxy.
    NOTE: This is still approximate; adjust active space/steps for your accuracy needs.
    """
    try:
        import pennylane as qml  # type: ignore
        from pennylane import qchem
    except Exception:
        return None

    BASIS = "sto-3g"
    ACTIVE_E = 4       # active electrons (even)
    ACTIVE_ORB = 4     # active orbitals
    VQE_STEPS = 10     # keep small for speed; raise for accuracy
    STEP_SIZE = 0.2

    cache: dict[str, float] = {}
    dev = qml.device("default.qubit", wires=2 * ACTIVE_ORB)

    def smiles_to_geometry(smiles: str) -> Optional[Tuple[List[str], List[Tuple[float, float, float]], int]]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return None
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        conf = mol.GetConformer()
        symbols: List[str] = []
        coords: List[Tuple[float, float, float]] = []
        electrons = 0
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            symbols.append(atom.GetSymbol())
            coords.append((pos.x, pos.y, pos.z))
            electrons += atom.GetAtomicNum()
        return symbols, coords, electrons

    def build_hamiltonian(symbols: List[str], coords: List[Tuple[float, float, float]]):
        h, _ = qchem.molecular_hamiltonian(
            symbols,
            coords,
            basis=BASIS,
            active_electrons=ACTIVE_E,
            active_orbitals=ACTIVE_ORB,
        )
        return h

    def vqe_energy(hamiltonian: qml.Hamiltonian) -> float:
        n_qubits = 2 * ACTIVE_ORB
        wires = list(range(n_qubits))
        n_params = qchem.UCCSD.shape(n_qubits, ACTIVE_E)
        params = np.zeros(n_params)

        @qml.qnode(dev)
        def circuit(p):
            qchem.UCCSD(p, wires=wires)
            return qml.expval(hamiltonian)

        opt = qml.GradientDescentOptimizer(stepsize=STEP_SIZE)
        e = circuit(params)
        p = params
        for _ in range(VQE_STEPS):
            p, e = opt.step_and_cost(circuit, p)
        return float(e)

    def prior_fn(smiles: str) -> float:
        if smiles in cache:
            return cache[smiles]
        geom = smiles_to_geometry(smiles)
        if geom is None:
            return 0.0
        symbols, coords, _ = geom
        try:
            H = build_hamiltonian(symbols, coords)
            energy = vqe_energy(H)  # variational estimate
            score = math.exp(-energy)  # lower energy -> higher score
            score = float(max(0.05, min(score, 5.0)))
            cache[smiles] = score
            return score
        except Exception:
            return 0.0

    return prior_fn


def estimate_qubits(num_atoms: int) -> int:
    """
    Rough qubit budgeting to mirror the sketch:
    - 2 qubits per atom choice (to encode ~4 atom states) * num_atoms
    - 2 qubits per bond choice (to encode ~4 bond states) * C(num_atoms, 2)
    Example: num_atoms=9 -> 2*9 + 2*C(9,2) = 18 + 72 = 90 qubits.
    """
    if num_atoms < 2:
        return 2 * num_atoms
    return 2 * num_atoms + 2 * math.comb(num_atoms, 2)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class SampledMolecule:
    atoms: Tuple[int, int, int]
    bonds: Tuple[int, int]
    smiles: Optional[str]
    valid: float


def atoms_bonds_to_smiles(atom_ids: Sequence[int], bond_ids: Sequence[int]) -> Tuple[Optional[str], float]:
    """Build a 3-atom chain molecule and return canonical SMILES and validity flag."""
    mol = Chem.RWMol()
    try:
        atom_indices = []
        for atom_id in atom_ids:
            symbol = ALLOWED_ATOMS[atom_id]
            atom_indices.append(mol.AddAtom(Chem.Atom(symbol)))

        # Connect as a simple chain: 0-1 and 1-2
        for (a_idx, b_idx, bond_id) in (
            (atom_indices[0], atom_indices[1], bond_ids[0]),
            (atom_indices[1], atom_indices[2], bond_ids[1]),
        ):
            bond_type = BOND_TYPES[bond_id]
            if bond_type is None:
                continue  # skip bond
            mol.AddBond(a_idx, b_idx, bond_type)

        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return smiles, 1.0
    except Exception:
        return None, 0.0


class PolicyNet(nn.Module):
    """Simple factorized policy over atom choices and bond types."""

    def __init__(self, n_atoms: int, n_bonds: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(1, hidden),  # dummy input to keep graph simple
            nn.ReLU(),
        )
        self.atom_heads = nn.ModuleList([nn.Linear(hidden, n_atoms) for _ in range(3)])
        self.bond_heads = nn.ModuleList([nn.Linear(hidden, n_bonds) for _ in range(2)])

    def distributions(self, temperature: float = 1.0) -> Tuple[List[torch.distributions.Categorical], List[torch.distributions.Categorical]]:
        x = self.shared(torch.ones(1, 1))
        atom_dists = [torch.distributions.Categorical(logits=head(x) / temperature) for head in self.atom_heads]
        bond_dists = [torch.distributions.Categorical(logits=head(x) / temperature) for head in self.bond_heads]
        return atom_dists, bond_dists


def sample_action(policy: PolicyNet, temperature: float = 1.0) -> Tuple[Tuple[int, int, int], Tuple[int, int], torch.Tensor, torch.Tensor]:
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
    return (atoms[0], atoms[1], atoms[2]), (bonds[0], bonds[1]), total_log_prob, total_entropy


def quantum_prior_score(smiles: Optional[str], prior_fn: Optional[Callable[[str], float]] = None) -> float:
    """
    Quantum-guided prior hook.
    - If `prior_fn` is provided, it should map SMILES -> non-negative score (higher is better).
      Example: use QMG/pennylane/cudaq to estimate -energy and rescale.
    - Default: deterministic hash-based proxy in [0.5, 1.0] for stable RL training.
    """
    if smiles is None:
        return 0.0
    if prior_fn:
        try:
            val = float(prior_fn(smiles))
            if val > 0:
                return val
        except Exception:
            return 0.0
    # Fallback: hashed proxy to keep reward > 0 for valid+unique cases.
    digest = hashlib.sha256(smiles.encode("utf-8")).digest()
    val = digest[0] / 255.0  # 0-1
    return 0.5 + 0.5 * val  # bias in [0.5,1]


def reinforce_training(
    episodes: int = 1000,
    lr: float = 0.05,
    gamma: float = 0.99,
    device: str = "cpu",
    prior_fn: Optional[Callable[[str], float]] = None,
    entropy_coef: float = 0.01,
    temperature: float = 1.2,
    min_temperature: float = 0.7,
    temp_decay: float = 0.995,
    batch_size: int = 8,
    max_grad_norm: float = 1.0,
) -> List[str]:
    policy = PolicyNet(n_atoms=len(ALLOWED_ATOMS), n_bonds=len(BOND_TYPES)).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    baseline = 0.0
    seen_smiles: set[str] = set()
    best_reward = -math.inf
    best_smiles: Optional[str] = None

    discovered: List[str] = []
    rewards: List[float] = []
    valid_count = 0
    batch_losses: List[torch.Tensor] = []
    for ep in range(1, episodes + 1):
        atoms, bonds, log_prob, entropy = sample_action(policy, temperature=temperature)
        smiles, valid = atoms_bonds_to_smiles(atoms, bonds)
        valid_count += int(valid)
        unique = 1.0 if smiles and smiles not in seen_smiles else 0.0
        if unique:
            seen_smiles.add(smiles)
            discovered.append(smiles)

        quantum_bias = quantum_prior_score(smiles, prior_fn=prior_fn)
        reward = valid * unique * quantum_bias  # encourages valid & novel; bias can be swapped with QMG energy proxy
        rewards.append(reward)

        baseline = 0.9 * baseline + 0.1 * reward
        advantage = reward - baseline
        loss = -(advantage * log_prob + entropy_coef * entropy)
        batch_losses.append(loss)

        if len(batch_losses) >= batch_size:
            optimizer.zero_grad()
            total_loss = torch.stack(batch_losses).mean()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            batch_losses.clear()

        if reward > best_reward:
            best_reward = reward
            best_smiles = smiles

        if ep % 50 == 0:
            print(
                f"[ep {ep:04d}] reward={reward:.3f} valid={valid:.0f} unique={unique:.0f} "
                f"bias={quantum_bias:.3f} atoms={[ALLOWED_ATOMS[a] for a in atoms]} "
                f"bonds={[BOND_LABELS[BOND_TYPES[b]] for b in bonds]} temp={temperature:.2f} "
                f"H={entropy.item():.2f} smiles={smiles}"
            )

        # Cool down temperature to shift from exploration to exploitation.
        temperature = max(min_temperature, temperature * temp_decay)

    # Flush any remaining gradients
    if batch_losses:
        optimizer.zero_grad()
        total_loss = torch.stack(batch_losses).mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

    print("\n===== Final summary =====")
    print(f"Samples (episodes): {episodes}")
    print(f"Valid count: {valid_count} | Unique valid: {len(discovered)}")
    if rewards:
        print(f"Reward max: {max(rewards):.3f} | mean: {sum(rewards)/len(rewards):.3f}")
    print("Top candidate:", best_smiles, "reward=", round(best_reward, 3))
    print("Confirmed unique valid SMILES:")
    for s in discovered:
        print("  ", s)
    return discovered


def main() -> None:
    set_seed(7)
    # Prefer PennyLane prior if available, else QMG CUDA-Q prior, else hashed proxy.
    prior_fn = build_pennylane_prior()
    if prior_fn is None:
        prior_fn = build_qmg_prior()

    print("Estimated qubits for 3 heavy atoms:", estimate_qubits(3))
    # Adjust episodes upward for better exploration; here we run longer with higher entropy.
    reinforce_training(
        episodes=10000,          # stronger training
        lr=0.02,                 # smaller LR for stability with larger batches
        gamma=0.99,
        prior_fn=prior_fn,
        entropy_coef=0.03,       # exploration via entropy
        temperature=1.5,         # start hotter
        min_temperature=0.6,
        temp_decay=0.998,        # slower cooling to keep exploring longer
        batch_size=16,           # mini-batch REINFORCE updates
        max_grad_norm=1.0,       # gradient clipping for stability
    )


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()

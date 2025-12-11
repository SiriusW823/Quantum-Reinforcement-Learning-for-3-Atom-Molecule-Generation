"""
Quantum RL for 5-atom heavy-atom molecules.

- Samples 5 heavy atoms and 4 bonds in a chain, converts to SMILES with RDKit.
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
import matplotlib.pyplot as plt

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
ALLOWED_ATOMS: Sequence[str] = ("C", "N", "O")

# Chain length (number of heavy atoms) and bonds
NUM_ATOMS_IN_CHAIN = 5
NUM_BONDS_IN_CHAIN = NUM_ATOMS_IN_CHAIN - 1
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
    Quantum prior using PySCF -> OpenFermion -> qubit Hamiltonian -> small VQE (PennyLane).
    RDKit 3D geometry -> PySCF RHF -> FermionHamiltonian -> Jordan-Wigner -> qubit Hamiltonian.
    Short VQE run to get energy; mapped to a positive score.
    """
    try:
        import pennylane as qml  # type: ignore
        from openfermionpyscf import run_pyscf  # type: ignore
        from openfermion.transforms import jordan_wigner  # type: ignore
        from openfermion import QubitOperator  # type: ignore
        import numpy as onp
    except Exception:
        return None

    BASIS = "sto-3g"
    cache: dict[str, float] = {}

    def smiles_to_geometry(smiles: str) -> Optional[List[Tuple[str, Tuple[float, float, float]]]]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Skip disconnected fragments to reduce bad embeddings
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
        # Reject geometries with overlapping atoms (singular overlap matrix risk)
        coords = [c for _, c in geom]
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dz = coords[i][2] - coords[j][2]
                if dx * dx + dy * dy + dz * dz < 1e-3:
                    return None
        return geom

    def qubit_hamiltonian(geom: List[Tuple[str, Tuple[float, float, float]]]):
        # Run PySCF RHF via OpenFermion to get Fermion Hamiltonian, then JW to qubits.
        data = run_pyscf(geom, basis=BASIS, multiplicity=1, charge=0, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=False, verbose=0)
        fermion_ham = data.get_molecular_hamiltonian()
        qubit_ham = jordan_wigner(fermion_ham)
        return qubit_ham, data.n_qubits

    def to_pl_hamiltonian(qubit_ham: QubitOperator, n_qubits: int) -> Tuple[List[float], List[qml.ops.PauliWord]]:  # type: ignore
        coeffs: List[float] = []
        ops: List[qml.ops.PauliWord] = []
        for term, coeff in qubit_ham.terms.items():
            if len(term) == 0:
                coeffs.append(coeff.real)
                ops.append(qml.Identity(0))
                continue
            paulis = []
            for idx, pauli_str in term:
                if pauli_str == "X":
                    paulis.append(qml.PauliX(idx))
                elif pauli_str == "Y":
                    paulis.append(qml.PauliY(idx))
                elif pauli_str == "Z":
                    paulis.append(qml.PauliZ(idx))
            op = paulis[0]
            for p in paulis[1:]:
                op = op @ p
            coeffs.append(coeff.real)
            ops.append(op)
        return coeffs, ops

    def vqe_energy(pl_hamiltonian: qml.Hamiltonian, n_qubits: int) -> float:
        dev = qml.device("default.qubit", wires=n_qubits)
        layers = 2
        weights = onp.zeros((layers, n_qubits, 3))

        @qml.qnode(dev)
        def circuit(w):
            qml.StronglyEntanglingLayers(w, wires=range(n_qubits))
            return qml.expval(pl_hamiltonian)

        opt = qml.AdamOptimizer(stepsize=0.1)
        w = weights
        e = circuit(w)
        for _ in range(12):  # small number of steps for speed
            w, e = opt.step_and_cost(circuit, w)
        return float(e)

    def prior_fn(smiles: str) -> float:
        if smiles in cache:
            return cache[smiles]
        geom = smiles_to_geometry(smiles)
        if geom is None:
            return 0.0
        try:
            q_ham, n_qubits = qubit_hamiltonian(geom)
            coeffs, ops = to_pl_hamiltonian(q_ham, n_qubits)
            pl_ham = qml.Hamiltonian(coeffs, ops)
            energy = vqe_energy(pl_ham, n_qubits)
            # Map energy to a bounded positive score; lower energy -> higher score.
            score = max(0.05, min(3.0, math.exp(-energy / 5.0)))
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
    atoms: Tuple[int, ...]
    bonds: Tuple[int, ...]
    smiles: Optional[str]
    valid: float


def atoms_bonds_to_smiles(atom_ids: Sequence[int], bond_ids: Sequence[int]) -> Tuple[Optional[str], float]:
    """Build a chain molecule and return canonical SMILES and validity flag."""
    mol = Chem.RWMol()
    try:
        atom_indices = []
        for atom_id in atom_ids:
            symbol = ALLOWED_ATOMS[atom_id]
            atom_indices.append(mol.AddAtom(Chem.Atom(symbol)))

        # Connect as a simple chain: i-(i+1)
        for i, bond_id in enumerate(bond_ids):
            a_idx, b_idx = atom_indices[i], atom_indices[i + 1]
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
        self.atom_heads = nn.ModuleList([nn.Linear(hidden, n_atoms) for _ in range(NUM_ATOMS_IN_CHAIN)])
        self.bond_heads = nn.ModuleList([nn.Linear(hidden, n_bonds) for _ in range(NUM_BONDS_IN_CHAIN)])

    def distributions(self, temperature: float = 1.0) -> Tuple[List[torch.distributions.Categorical], List[torch.distributions.Categorical]]:
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
    history_steps: List[int] = []
    history_valid_ratio: List[float] = []
    history_unique_ratio: List[float] = []
    history_combo: List[float] = []
    for ep in range(1, episodes + 1):
        atoms, bonds, log_prob, entropy = sample_action(policy, temperature=temperature)
        smiles, valid = atoms_bonds_to_smiles(atoms, bonds)
        valid_count += int(valid)
        # Enforce connectivity: if RDKit built fragments ('.' in SMILES), treat as invalid
        if valid and smiles and "." in smiles:
            valid = 0.0
        unique = 1.0 if smiles and smiles not in seen_smiles else 0.0
        if unique:
            seen_smiles.add(smiles)
            discovered.append(smiles)

        quantum_bias = quantum_prior_score(smiles, prior_fn=prior_fn)
        # Reward: valid + unique -> quantum_bias; duplicates (still valid) get a modest penalty to encourage diversity.
        if valid and unique:
            reward = quantum_bias
        elif valid and not unique:
            reward = -0.01
        else:
            reward = 0.0
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

        # Track ratios for convergence plotting
        history_steps.append(ep)
        history_valid_ratio.append(valid_count / ep)
        history_unique_ratio.append(len(seen_smiles) / ep)
        history_combo.append(history_valid_ratio[-1] * history_unique_ratio[-1])

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
    if history_steps:
        combo_final = history_combo[-1]
        print(f"(Valid/Samples)*(Unique/Samples): {combo_final:.4f}")
    print("Top candidate:", best_smiles, "reward=", round(best_reward, 3))
    print("Confirmed unique valid SMILES:")
    for s in discovered:
        print("  ", s)

    # Convergence plot
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(history_steps, history_valid_ratio, label="valid/samples")
        plt.plot(history_steps, history_unique_ratio, label="unique/samples")
        plt.plot(history_steps, history_combo, label="(valid/samples)*(unique/samples)")
        plt.xlabel("Episode")
        plt.ylabel("Ratio")
        plt.title("Convergence")
        plt.legend()
        plt.tight_layout()
        plt.savefig("convergence.png", dpi=150)
        print("Saved convergence plot to convergence.png")
    except Exception as e:
        print("Plotting failed:", e)

    return discovered


def main() -> None:
    set_seed(7)
    # Prefer PennyLane prior if available, else QMG CUDA-Q prior, else hashed proxy.
    prior_fn = build_pennylane_prior()
    if prior_fn is None:
        prior_fn = build_qmg_prior()

    print("Estimated qubits for 5 heavy atoms:", estimate_qubits(NUM_ATOMS_IN_CHAIN))
    # Adjust episodes upward for better exploration; here we run longer with higher entropy.
    reinforce_training(
        episodes=20000,          # stronger training
        lr=0.02,                 # smaller LR for stability with larger batches
        gamma=0.99,
        prior_fn=prior_fn,
        entropy_coef=0.05,       # slightly stronger exploration via entropy
        temperature=1.7,         # start a bit hotter to diversify
        min_temperature=0.6,
        temp_decay=0.998,        # moderate cooling
        batch_size=16,           # mini-batch REINFORCE updates
        max_grad_norm=1.0,       # gradient clipping for stability
    )


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()

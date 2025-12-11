from dataclasses import dataclass
from typing import Sequence


@dataclass
class RLConfig:
    episodes: int = 20000
    lr: float = 0.02
    gamma: float = 0.99
    entropy_coef: float = 0.05
    temperature: float = 1.7
    min_temperature: float = 0.6
    temp_decay: float = 0.998
    batch_size: int = 16
    max_grad_norm: float = 1.0
    seed: int = 7
    log_interval: int = 50


@dataclass
class QuantumConfig:
    basis: str = "sto-3g"
    target_qubits: int = 8  # active-space qubits (4–8 recommended)
    vqe_layers: int = 2
    vqe_steps: int = 12
    vqe_lr: float = 0.1
    energy_scale: float = 5.0  # used in exp(-E/energy_scale)
    score_min: float = 0.05
    score_max: float = 3.0
    valence_penalty_weight: float = 1.0
    connectivity_penalty_weight: float = 1.0
    geometry_penalty_weight: float = 0.5
    geom_d0: float = 1.2  # reference distance in Angstrom
    geom_sigma: float = 0.4
    lambda_reward_scale: float = 2.0  # λ in reward exp(-E/λ)


@dataclass
class ChemistryConfig:
    allowed_atoms: Sequence[str] = ("C", "N", "O")
    num_atoms_in_chain: int = 5
    allow_disconnected: bool = False
    min_dist2: float = 1e-3  # squared minimum distance to reject overlapping coords


# Global config instance
rl = RLConfig()
quantum = QuantumConfig()
chem = ChemistryConfig()

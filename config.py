from dataclasses import dataclass
from typing import Sequence


@dataclass
class RLConfig:
    episodes: int = 20000
    lr: float = 0.02
    gamma: float = 0.99
    entropy_coef: float = 0.05
    temperature: float = 1.5
    min_temperature: float = 0.6
    temp_decay: float = 0.998
    batch_size: int = 16
    max_grad_norm: float = 1.0
    log_interval: int = 100
    seed: int = 7


@dataclass
class ChemConfig:
    allowed_atoms: Sequence[str] = ("C", "N", "O")
    num_atoms: int = 3  # chain length for Level-5 spec
    min_dist2: float = 1e-3


@dataclass
class QuantumConfig:
    basis: str = "sto-3g"
    target_qubits: int = 8  # active space 4–8 qubits
    vqe_layers: int = 2
    vqe_steps: int = 12
    vqe_lr: float = 0.1
    energy_scale: float = 5.0
    score_min: float = 0.05
    score_max: float = 3.0
    valence_penalty_weight: float = 1.0
    connectivity_penalty_weight: float = 1.0
    geometry_penalty_weight: float = 0.5
    geom_d0: float = 1.2
    geom_sigma: float = 0.4
    lambda_reward: float = 2.0  # reward scale in exp(-E/λ)
    knn_k: int = 5


@dataclass
class EncoderConfig:
    n_qubits: int = 8  # for quantum encoding of SMILES features


rl = RLConfig()
chem = ChemConfig()
quantum = QuantumConfig()
encoder = EncoderConfig()

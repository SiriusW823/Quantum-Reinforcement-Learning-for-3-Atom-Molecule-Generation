from typing import List
import torch
import torch.nn as nn
from quantum.encodings import smiles_to_features
from quantum.pqc_blocks import make_torch_pqc


class QuantumHelper(nn.Module):
    """
    Quantum helper / critic: maps molecule SMILES to a scalar value via PQC.
    - Encodes fingerprint bits with AngleEmbedding.
    - StronglyEntanglingLayers ansatz.
    - Linear head maps expvals to a scalar; tanh squashes to [-1,1].
    """

    def __init__(self, n_wires: int = 6, layers: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.pqc = make_torch_pqc(n_wires=n_wires, layers=layers)
        self.head = nn.Linear(n_wires, 1)

    def forward(self, smiles: str) -> torch.Tensor:
        angles = smiles_to_features(smiles, n_bits=self.n_wires)
        expvals = self.pqc(angles)
        expvals = expvals.float()
        val = self.head(expvals)
        return torch.tanh(val).squeeze()

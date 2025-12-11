from typing import List, Tuple
import numpy as np
import pennylane as qml
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def smiles_to_features(smiles: str, n_bits: int = 64) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def angle_embedding_from_smiles(smiles: str, n_qubits: int) -> Tuple[np.ndarray, List[int]]:
    feats = smiles_to_features(smiles, n_bits=n_qubits)
    angles = feats * np.pi  # map bits to 0 or pi
    wires = list(range(n_qubits))
    return angles, wires


def prepare_state(smiles: str, n_qubits: int) -> qml.QNode:
    angles, wires = angle_embedding_from_smiles(smiles, n_qubits=n_qubits)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.AngleEmbedding(angles, wires=wires)
        return qml.state()

    return circuit

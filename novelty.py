from typing import Dict, List, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


class ClassicalNovelty:
    """V2 classical novelty: hash + kNN on Morgan fingerprints."""

    def __init__(self, k: int = 5):
        self.freq: Dict[str, int] = {}
        self.k = k
        self.fps: List[Tuple[str, np.ndarray]] = []

    def _fp(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(2048, dtype=np.int8)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def update(self, smiles: str) -> None:
        if not smiles:
            return
        self.freq[smiles] = self.freq.get(smiles, 0) + 1
        self.fps.append((smiles, self._fp(smiles)))

    def novelty(self, smiles: str) -> float:
        if not smiles:
            return 0.0
        freq_term = 1.0 / (1.0 + self.freq.get(smiles, 0))
        fp_new = self._fp(smiles)
        if len(self.fps) == 0:
            return freq_term
        dists = []
        for _, fp_old in self.fps:
            hamming = np.count_nonzero(fp_new != fp_old) / fp_new.size
            dists.append(hamming)
        dists = sorted(dists)[: max(1, self.k)]
        knn_term = float(np.mean(dists))
        novelty = 0.5 * freq_term + 0.5 * knn_term
        # normalize to [0,1]
        return max(0.0, min(1.0, novelty))

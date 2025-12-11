import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import pennylane as qml
from openfermion import QubitOperator  # type: ignore
from openfermionpyscf import run_pyscf  # type: ignore
from openfermion.transforms import jordan_wigner  # type: ignore

from .chem import embed_geometry


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


def build_active_space(smiles: str, basis: str, target_qubits: int) -> Optional[Tuple[QubitOperator, int]]:
    geom = embed_geometry(smiles)
    if geom is None:
        return None
    # target_qubits = 2 * active_orbitals
    max_orb = max(2, min(target_qubits // 2, 4))
    # count electrons
    total_e = sum(int(sym != "" and ChemPeriodicTable.get_atomic_number(sym)) for sym, _ in geom)  # type: ignore[name-defined]
    active_e = min(total_e, 2 * max_orb)
    try:
        data = run_pyscf(
            geom,
            basis=basis,
            multiplicity=1,
            charge=0,
            run_mp2=False,
            run_cisd=False,
            run_ccsd=False,
            run_fci=False,
            verbose=0,
            active_space=(active_e, max_orb),
        )
        fermion_ham = data.get_molecular_hamiltonian()
        qubit_ham = jordan_wigner(fermion_ham)
        return qubit_ham, data.n_qubits
    except Exception:
        return None


def build_quantum_prior(basis: str, target_qubits: int, layers: int, steps: int, stepsize: float, energy_scale: float, score_min: float, score_max: float) -> Callable[[str], float]:
    cache: dict[str, float] = {}

    def vqe_energy(pl_hamiltonian: qml.Hamiltonian, n_qubits: int) -> float:
        dev = qml.device("default.qubit", wires=n_qubits)
        weights = np.zeros((layers, n_qubits, 3))

        @qml.qnode(dev)
        def circuit(w):
            qml.StronglyEntanglingLayers(w, wires=range(n_qubits))
            return qml.expval(pl_hamiltonian)

        opt = qml.AdamOptimizer(stepsize=stepsize)
        w = weights
        e = circuit(w)
        for _ in range(steps):
            w, e = opt.step_and_cost(circuit, w)
        return float(e)

    def prior_fn(smiles: str) -> float:
        if smiles in cache:
            return cache[smiles]
        hq = build_active_space(smiles, basis=basis, target_qubits=target_qubits)
        if hq is None:
            return 0.0
        qubit_ham, n_qubits = hq
        try:
            coeffs, ops = to_pl_hamiltonian(qubit_ham, n_qubits)
            pl_ham = qml.Hamiltonian(coeffs, ops)
            energy = vqe_energy(pl_ham, n_qubits)
            score = max(score_min, min(score_max, math.exp(-energy / energy_scale)))
            cache[smiles] = score
            return score
        except Exception:
            return 0.0

    return prior_fn

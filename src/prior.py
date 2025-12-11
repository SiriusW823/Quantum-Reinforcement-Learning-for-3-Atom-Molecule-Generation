import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import pennylane as qml
from rdkit import Chem
from openfermion import QubitOperator  # type: ignore
from openfermionpyscf import run_pyscf  # type: ignore
from openfermion.transforms import jordan_wigner  # type: ignore

from .chem import (
    ALLOWED_VALENCE,
    distance_matrix,
    embed_geometry,
    valence_usage,
    connectivity_components,
)


def to_pl_hamiltonian(qubit_ham: QubitOperator) -> Tuple[List[float], List[qml.operation.Operator]]:
    coeffs: List[float] = []
    ops: List[qml.ops.PauliWord] = []
    for term, coeff in qubit_ham.terms.items():
        if len(term) == 0:
            coeffs.append(coeff.real)
            ops.append(qml.Identity(0))
            continue
        op = None
        for idx, pauli_str in term:
            if pauli_str == "X":
                pauli_op = qml.PauliX(idx)
            elif pauli_str == "Y":
                pauli_op = qml.PauliY(idx)
            elif pauli_str == "Z":
                pauli_op = qml.PauliZ(idx)
            if op is None:
                op = pauli_op
            else:
                op = op @ pauli_op
        if op is None:
            op = qml.Identity(0)
        coeffs.append(coeff.real)
        ops.append(op)
    return coeffs, ops


def build_quantum_chemistry_hamiltonian(geom: List[Tuple[str, Tuple[float, float, float]]], basis: str, target_qubits: int) -> Optional[Tuple[qml.Hamiltonian, int]]:
    # target_qubits = 2 * active_orbitals
    max_orb = max(2, min(target_qubits // 2, 4))
    total_e = sum(_atomic_number(sym) for sym, _ in geom)
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
        coeffs, ops = to_pl_hamiltonian(qubit_ham)
        h_pl = qml.Hamiltonian(coeffs, ops)
        return h_pl, data.n_qubits
    except Exception:
        return None


def build_valence_constraint(smiles: str, weight: float) -> Optional[qml.Hamiltonian]:
    usage = valence_usage(smiles)
    if usage is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    penalties = 0.0
    for idx, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        allowed = ALLOWED_VALENCE.get(symbol, 4)
        diff = usage[idx] - allowed
        penalties += weight * (diff ** 2)
    return qml.Hamiltonian([penalties], [qml.Identity(0)])


def build_connectivity_constraint(smiles: str, weight: float) -> Optional[qml.Hamiltonian]:
    comp = connectivity_components(smiles)
    if comp == 0:
        return None
    penalty = weight * abs(comp - 1)
    return qml.Hamiltonian([penalty], [qml.Identity(0)])


def build_geometry_constraint(geom: List[Tuple[str, Tuple[float, float, float]]], weight: float, d0: float, sigma: float) -> Optional[qml.Hamiltonian]:
    if not geom:
        return None
    dmat = distance_matrix(geom)
    penalties = 0.0
    n = len(dmat)
    for i in range(n):
        for j in range(i + 1, n):
            dij = dmat[i][j]
            penalties += weight * math.exp(-((dij - d0) ** 2) / (sigma ** 2))
    return qml.Hamiltonian([penalties], [qml.Identity(0)])


def assemble_total_hamiltonian(
    h_chem: qml.Hamiltonian,
    h_val: Optional[qml.Hamiltonian],
    h_conn: Optional[qml.Hamiltonian],
    h_geom: Optional[qml.Hamiltonian],
) -> qml.Hamiltonian:
    total_coeffs = list(h_chem.coeffs)
    total_ops = list(h_chem.ops)
    for h in [h_val, h_conn, h_geom]:
        if h is None:
            continue
        total_coeffs.extend(h.coeffs)
        total_ops.extend(h.ops)
    return qml.Hamiltonian(total_coeffs, total_ops)


def vqe_energy(hamiltonian: qml.Hamiltonian, n_qubits: int, layers: int, steps: int, stepsize: float) -> float:
    dev = qml.device("default.qubit", wires=n_qubits)
    weights = np.zeros((layers, n_qubits, 3))

    @qml.qnode(dev)
    def circuit(w):
        qml.StronglyEntanglingLayers(w, wires=range(n_qubits))
        return qml.expval(hamiltonian)

    opt = qml.AdamOptimizer(stepsize=stepsize)
    w = weights
    e = circuit(w)
    for _ in range(steps):
        w, e = opt.step_and_cost(circuit, w)
    return float(e)


def build_quantum_prior(
    basis: str,
    target_qubits: int,
    layers: int,
    steps: int,
    stepsize: float,
    energy_scale: float,
    score_min: float,
    score_max: float,
    valence_weight: float,
    connectivity_weight: float,
    geometry_weight: float,
    geom_d0: float,
    geom_sigma: float,
) -> Callable[[str], float]:
    cache: dict[str, float] = {}

    def prior_fn(smiles: str) -> float:
        if smiles in cache:
            return cache[smiles]
        geom = embed_geometry(smiles)
        if geom is None:
            return 0.0
        h_chem_tuple = build_quantum_chemistry_hamiltonian(geom, basis=basis, target_qubits=target_qubits)
        if h_chem_tuple is None:
            return 0.0
        h_chem, n_qubits = h_chem_tuple
        h_val = build_valence_constraint(smiles, weight=valence_weight)
        h_conn = build_connectivity_constraint(smiles, weight=connectivity_weight)
        h_geom = build_geometry_constraint(geom, weight=geometry_weight, d0=geom_d0, sigma=geom_sigma)
        try:
            h_total = assemble_total_hamiltonian(h_chem, h_val, h_conn, h_geom)
            energy = vqe_energy(h_total, n_qubits=n_qubits, layers=layers, steps=steps, stepsize=stepsize)
            cache[smiles] = energy
            return energy
        except Exception:
            return 0.0

    return prior_fn


def _atomic_number(symbol: str) -> int:
    from rdkit.Chem import GetPeriodicTable
    pt = GetPeriodicTable()
    return pt.GetAtomicNumber(symbol)

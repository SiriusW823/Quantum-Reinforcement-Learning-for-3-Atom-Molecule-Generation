# Quantum RL for 3-Atom Molecule Generation

![Reinforcement Learning Loop](assets/rl-diagram.jpg)

Reinforcement-learning (REINFORCE) agent that samples three heavy atoms and two bonds, builds candidate molecules with RDKit, and scores them with a quantum prior. The default prior uses PySCF + OpenFermion (Jordan–Wigner) to produce a qubit Hamiltonian and runs a short PennyLane VQE to obtain an energy proxy; a CUDA-Q/QMG hook is available for alternative backends. Objective: maximize `reward = validity × uniqueness × quantum_prior`.

## Overview
- **Chemical space**: atoms = (`C, N, O`); bonds = `NONE / Single / Double / Triple`; chain topology 0–1–2.
- **Quantum prior**: PySCF (sto-3g) HF integrals → OpenFermion fermionic Hamiltonian → JW to qubit Hamiltonian → PennyLane VQE (`StronglyEntanglingLayers`) → energy → score.
- **RL**: entropy-regularized REINFORCE with temperature annealing, mini-batch updates, gradient clipping; 20k episodes by default.
- **Filtering**: rejects disconnected SMILES and overlapping geometries; duplicates incur a small penalty to push diversity.
- **Logging**: progress every 50 steps; final summary of samples, validity, uniqueness, reward stats, and all unique valid SMILES.

## Pipeline Details
1. **Sampling**: policy outputs `(atom1, atom2, atom3, bond1, bond2)` with temperature + entropy control.
2. **Build**: RDKit constructs 0–1 and 1–2 bonds; if build fails or SMILES contains '.', it is marked invalid.
3. **Geometry**: RDKit ETKDG + UFF, with overlap check to avoid singular overlap matrices.
4. **Quantum prior**:
   - PySCF RHF (sto-3g) computes integrals.
   - OpenFermion converts to a fermionic Hamiltonian, then JW to qubit Hamiltonian.
   - PennyLane VQE (short run, few steps) estimates energy.
   - Energy mapped to a bounded score (`exp(-E/5)`, clamped to [0.05, 2.5]).
5. **Reward**: valid + unique → quantum score; valid + duplicate → -0.02; invalid → 0.
6. **Update**: mini-batch REINFORCE (batch 16) with gradient clipping.

## Environment & Setup (WSL/Linux recommended)
```bash
conda create -n qmg python=3.10 rdkit pytorch cpuonly numpy=1.26.4 h5py=3.10 -c conda-forge -c pytorch -y
conda activate qmg
pip install --no-cache-dir pennylane pyscf openfermion openfermionpyscf
```
If CUDA-Q/QMG is needed, install your CUDA-Q stack and implement `build_qmg_prior`.

## Run
```bash
conda activate qmg
python "Quantum Reinforcement Learning for 3-Atom Molecule Generation.py"
```

## Key Tunables (in `reinforce_training`)
- `episodes` (default 20000), `lr` (0.02), `entropy_coef` (0.05), `temperature`, `temp_decay`, `min_temperature`
- `batch_size` (16), `max_grad_norm` (1.0)
- Prior scaling in `build_pennylane_prior` (`exp(-E/5)` clamped to [0.05, 2.5])
- Duplicate penalty (-0.02) in reward logic; disconnected SMILES treated as invalid.

## Outputs
- Every 50 steps: reward, valid/unique flags, quantum bias, atoms, bonds, temperature, entropy, SMILES.
- Final summary: sample count, valid count, unique valid count, reward max/mean, best SMILES, and list of unique valid SMILES.

## Customization
- **Atom/bond space**: edit `ALLOWED_ATOMS` and `BOND_TYPES`.
- **Quantum prior**: adjust basis, VQE steps/ansatz, or swap to CUDA-Q/QMG in `build_qmg_prior`.
- **Reward shaping**: change score clamp or duplicate penalty.
- **Exploration**: tweak temperature/entropy schedule and episodes.

## Project Layout
```
Quantum Reinforcement Learning for 3-Atom Molecule Generation.py  # main script
README.md
LICENSE
assets/rl-diagram.jpg
```

## License
All rights reserved. No permission is granted to use, copy, modify, or distribute this work without explicit written consent from the author.

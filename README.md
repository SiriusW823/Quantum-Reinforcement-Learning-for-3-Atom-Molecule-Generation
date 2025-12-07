# Quantum RL for 3-Atom Molecule Generation

![Reinforcement Learning Loop](assets/rl-diagram.jpg)

Reinforcement-learning (REINFORCE) agent that samples 3 heavy atoms and 2 bonds, builds molecules with RDKit, and scores them with a quantum prior. The default prior uses PySCF + OpenFermion (Jordan–Wigner) to produce a qubit Hamiltonian and runs a short PennyLane VQE to obtain an energy proxy; a CUDA-Q/QMG hook remains available. Goal: maximize `reward = validity × uniqueness × quantum_prior`.

## Features
- **Chemical space**: atoms = (`C, N, O`); bonds = `NONE / Single / Double / Triple`.
- **Quantum prior**: PySCF (sto-3g) → OpenFermion JW → PennyLane VQE (`StronglyEntanglingLayers`) → energy mapped to score.
- **RL**: entropy-regularized REINFORCE, temperature annealing, mini-batch updates, gradient clipping; 20k episodes by default.
- **Filtering**: rejects disconnected SMILES and overlapping geometries; invalid/duplicate penalties encourage diversity.
- **Logging**: prints atoms/bonds/SMILES every 50 steps; final summary of samples, validity, uniqueness, reward stats, and unique valid SMILES.

## How It Works
1. Policy samples `(atom1, atom2, atom3, bond1, bond2)` with temperature + entropy control.
2. RDKit builds a chain (0–1, 1–2); disconnected or failed builds → invalid.
3. PySCF HF computes integrals; OpenFermion generates a fermionic Hamiltonian, mapped to qubit Hamiltonian (JW); PennyLane VQE estimates energy; energy → score (`exp(-E/5)` bounded).
4. Reward = quantum score if valid & unique; duplicates penalized; invalid = 0.
5. Mini-batch REINFORCE updates with gradient clipping.

## Environment & Setup (WSL/Linux recommended)
```bash
conda create -n qmg python=3.10 rdkit pytorch cpuonly numpy=1.26.4 h5py=3.10 -c conda-forge -c pytorch -y
conda activate qmg
pip install --no-cache-dir pennylane pyscf openfermion openfermionpyscf
```
If CUDA-Q/QMG is needed, install your CUDA-Q stack and fill the hook in `build_qmg_prior`.

## Run
```bash
conda activate qmg
python "Quantum Reinforcement Learning for 3-Atom Molecule Generation.py"
```

## Key Tunables (see `reinforce_training`)
- `episodes` (default 20000), `lr` (0.02), `entropy_coef` (0.05), `temperature` / `temp_decay` / `min_temperature`
- `batch_size` (16), `max_grad_norm` (1.0)
- Prior scaling in `build_pennylane_prior` (`exp(-E/5)` clamped to [0.05, 2.5]); duplicate penalty (-0.02) in reward logic.

## Outputs
- Every 50 steps: reward, valid/unique flags, quantum bias, atoms, bonds, temp, entropy, SMILES.
- Final: sample count, valid count, unique valid count, reward max/mean, best SMILES, list of unique valid SMILES.

## Project Layout
```
Quantum Reinforcement Learning for 3-Atom Molecule Generation.py  # main script
README.md
LICENSE
assets/rl-diagram.jpg
```

## License
All rights reserved. No permission is granted to use, copy, modify, or distribute this work without explicit written consent from the author.

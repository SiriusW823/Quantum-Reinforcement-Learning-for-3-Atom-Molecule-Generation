# Quantum RL for 3-Atom Molecule Generation

![Reinforcement Learning Loop](assets/rl-diagram.jpg)

Reinforcement-learning (REINFORCE) policy that samples three heavy atoms and two bonds, builds candidate molecules with RDKit, and scores them with a quantum-guided prior. A PennyLane qchem VQE (PySCF backend) provides a simplified quantum-chemistry energy proxy (small active space, short VQE); a CUDA-Q/QMG hook remains for alternative backends. Objective: maximize `reward = validity x uniqueness x quantum_prior`, pushing the product toward 1.

## Key Features
- 3-atom generator: heavy-atom set (`C, N, O`) with bond choices `NONE/Single/Double/Triple`.
- Quantum prior: PennyLane qchem VQE (PySCF, sto-3g, small active space) included; pluggable QMG/CUDA-Q prior hook.
- RL training: entropy-regularized REINFORCE, temperature annealing, mini-batch updates, gradient clipping (10k steps default).
- Reporting: periodic prints of atoms/bonds/SMILES; final summary of samples, validity, uniqueness, reward stats, and unique valid SMILES.

## How It Works
1. Policy samples `(atom1, atom2, atom3, bond1, bond2)` with temperature and entropy regularization.
2. RDKit attempts to build a chain (0-1, 1-2); failure -> `valid=0`.
3. Reward = `valid x uniqueness x quantum_prior`. PennyLane prior maps SMILES -> energy -> positive score; QMG/CUDA-Q hook can replace it.
4. REINFORCE updates parameters in mini-batches with gradient clipping.

## Environment & Setup
```bash
conda create -n qmg python=3.11 rdkit -c conda-forge -y
conda activate qmg
pip install torch pennylane pennylane-qchem pyscf
```
Optional: install CUDA-Q/QMG stack if you plan to use the QMG prior.

## Run
```bash
conda activate qmg
python "Quantum Reinforcement Learning for 3-Atom Molecule Generation.py"
```

## Tuning Knobs (in `reinforce_training`)
- `episodes`: total training steps (default 10000).
- `lr`: learning rate (default 0.02).
- `entropy_coef`: exploration strength (default 0.03).
- `temperature`, `temp_decay`, `min_temperature`: controls exploration->exploitation annealing.
- `batch_size`: mini-batch size for REINFORCE updates (default 16).
- `max_grad_norm`: gradient clipping (default 1.0).

## Quantum Prior Integration
- PennyLane (default, qchem VQE): `build_pennylane_prior` builds a molecular Hamiltonian via PySCF (sto-3g, active_electrons=4, active_orbitals=4) and runs a short UCCSD VQE to get energy; maps to score via `exp(-E)`. Tune active space / VQE steps for accuracy vs speed.
- QMG/CUDA-Q: implement `build_qmg_prior` so `prior_fn(smiles) -> non-negative score` (typical `max(0, -energy)`).

## Outputs
- Console log every 50 steps: reward, valid/unique flags, quantum bias, atoms, bonds, temperature, entropy, SMILES.
- Final summary: sample count, valid count, unique valid count, reward max/mean, best candidate, and full list of unique valid SMILES.

## Project Layout
```
Quantum Reinforcement Learning for 3-Atom Molecule Generation.py  # main script
README.md
LICENSE
assets/rl-diagram.jpg
```

## License
All rights reserved. No permission is granted to use, copy, modify, or distribute this work without explicit written consent from the author.

# Quantum RL for 3-Atom Molecule Generation

![Reinforcement Learning Loop](assets/rl-diagram.jpg)

This repository contains a reinforcement-learning (REINFORCE) policy that samples three heavy atoms and two bonds, builds candidate molecules with RDKit, and uses a quantum-guided prior (PennyLane by default; CUDA-Q/QMG hook included) to bias generation toward low-energy, valid, and unique SMILES. The goal is to maximize `reward = validity × uniqueness × quantum_prior`, driving the product toward 1.

## Highlights
- **3-atom generator**: heavy-atom set (`B,C,N,O,F,Si,P,S,Cl,Br,I`) with bond choices `NONE/Single/Double/Triple`.
- **Quantum prior**: PennyLane two-qubit toy circuit included; swap in your chemistry ansatz or QMG/CUDA-Q energy estimator via the provided hooks.
- **RL training**: entropy-regularized REINFORCE with temperature annealing, mini-batch updates, and gradient clipping for stability.
- **Reporting**: prints atoms/bonds/SMILES during training; final summary of samples, validity, uniqueness, reward stats, and all unique valid SMILES.

## Environment & Dependencies
```bash
conda create -n qmg python=3.11 rdkit -c conda-forge -y
conda activate qmg
pip install torch pennylane
```
CUDA-Q/QMG users: install your CUDA-Q stack and fill in the QMG prior hook.

## How to Run
```bash
conda activate qmg
python "Quantum Reinforcement Learning for 3-Atom Molecule Generation.py"
```
Key tunables are in `reinforce_training` (episodes, lr, entropy_coef, temperature schedule, batch_size).

## Quantum Prior Integration
- PennyLane: edit `build_pennylane_prior` to drop in your ansatz/observable and SMILES→Hamiltonian mapping.
- QMG/CUDA-Q: implement `build_qmg_prior` so `prior_fn(smiles) -> non-negative score` (e.g., `max(0, -energy)`).

## RL Loop (diagram)
Place your RL diagram image at `assets/rl-diagram.png` to render in this README (the provided diagram corresponds to the standard agent–environment loop).

## License
All rights reserved. No permission is granted to use, copy, modify, or distribute this work without explicit written consent from the author.

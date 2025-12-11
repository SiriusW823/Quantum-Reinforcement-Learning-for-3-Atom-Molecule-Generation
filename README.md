# Quantum-Guided RL for Molecule Generation (Level-2 Quantum Environment)

![Reinforcement Learning Loop](assets/rl-diagram.jpg)

Classical REINFORCE policy with a **quantum-informed environment**. The agent samples atom/bond sequences; RDKit builds candidate molecules; the reward is dominated by a PennyLane quantum Hamiltonian that combines:
- **Quantum chemistry energy** (PySCF → OpenFermion → JW → PennyLane VQE, active space 4–8 qubits)
- **Valence penalty** (deviation from allowed valence)
- **Connectivity penalty** (enforce a single component)
- **Geometry penalty** (soft penalty on unphysical distances)

Reward: `reward = exp(-E_total / λ) * rdkit_valid * uniqueness`, driving `(Valid/Samples)*(Unique/Samples)` toward 1.

## Architecture
- **chem.py**: RDKit build/embedding, bond/valence/connectivity utilities, qubit estimate.
- **prior.py**: PySCF integrals → OpenFermion fermionic H → JW → qubit H; adds valence/connectivity/geometry penalty Hamiltonians; short PennyLane VQE returns energy.
- **policy.py**: Classical factorized policy (atoms/bonds) with temperature/entropy control.
- **trainer.py**: REINFORCE loop, mini-batch updates, entropy regularization, temperature annealing, reward as above; convergence plotting.
- **config.py**: RL hyperparameters, chemistry space, quantum/VQE settings, penalty weights.
- **train.py**: Entry point wiring everything together.

## Default Chemical Space
- Chain of 5 heavy atoms (`C, N, O`), bonds `NONE/Single/Double/Triple` along 0–1–2–3–4.

## Quantum Prior (Level-2)
1. RDKit ETKDG + UFF geometry (reject fragments/overlaps).
2. PySCF RHF (sto-3g) → OpenFermion fermionic Hamiltonian.
3. Jordan–Wigner → qubit Hamiltonian (active space capped by `target_qubits`, default 8).
4. Penalties as Pauli sums (Identity terms):
   - Valence: Σ β (valence_i − allowed_i)²
   - Connectivity: δ |components − 1|
   - Geometry: Σ γ exp(−(d_ij − d0)² / σ²)
5. PennyLane VQE (StronglyEntanglingLayers, few steps) on CPU.
6. Energy → reward via `exp(-E_total / λ)`.

## Install (WSL/Linux recommended)
```bash
conda create -n qmg python=3.10 rdkit pytorch cpuonly numpy=1.26.4 h5py=3.10 -c conda-forge -c pytorch -y
conda activate qmg
pip install --no-cache-dir pennylane pyscf openfermion openfermionpyscf matplotlib
```

## Run
```bash
conda activate qmg
python train.py
```

## Key Config (see `src/config.py`)
- RL: `episodes` (20000), `lr`, `entropy_coef`, `temperature`/`temp_decay`/`min_temperature`, `batch_size`, `max_grad_norm`.
- Chemistry: `allowed_atoms`, `num_atoms_in_chain`, `min_dist2`.
- Quantum: `basis`, `target_qubits` (active space), VQE `layers/steps/stepsize`, penalty weights (`valence_penalty_weight`, `connectivity_penalty_weight`, `geometry_penalty_weight`, `geom_d0`, `geom_sigma`), `lambda_reward_scale`.

## Outputs
- Console log every 50 steps (reward, valid/unique, energy, atoms/bonds, temp, entropy, SMILES).
- Final summary: samples, valid, unique, reward stats, `(Valid/Samples)*(Unique/Samples)`, best SMILES, full unique list.
- `convergence.png`: curves for valid/samples, unique/samples, and their product.

## Customization
- Expand atom/bond vocab or chain length via `config.chem`.
- Adjust active space/qubits, basis, VQE steps in `config.quantum`.
- Tune penalties and λ to balance validity vs uniqueness.
- Swap ansatz or add CUDA-Q/QMG by implementing `build_qmg_prior`.

## License
All rights reserved. No permission is granted to use, copy, modify, or distribute this work without explicit written consent from the author.

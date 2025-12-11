# Quantum-Guided RL for Molecule Generation (V2 Classical + V3 Quantum)

![Reinforcement Learning Loop](assets/rl-diagram.jpg)

This project upgrades a classical RL pipeline into a **Level-5 full-quantum environment** using PennyLane, RDKit, PySCF, OpenFermion, and PyTorch. Both policy and prior are quantum-driven: the agent samples atom/bond sequences via a quantum policy network (QPN), builds SMILES, and the reward blends classical novelty (V2) and quantum signals (V3: quantum novelty + quantum energy prior) to maximize `(valid/samples) * (unique/samples) → 1`.

## Goals
- Generate valid, unique molecules via REINFORCE.
- Use full quantum reward:
  - Quantum novelty via state fidelity.
  - VQE energy as quantum prior (PySCF → OpenFermion → JW → PennyLane).
- Combined reward:
  `reward = valid * novelty_classical * novelty_quantum * quantum_energy_prior`

## Modules
- **chem.py**: RDKit molecule builder/validity; utilities for valence/connectivity/geometry; qubit estimate.
- **policy.py**: **Quantum policy network** (PennyLane QNode + StronglyEntanglingLayers) outputs logits for atoms/bonds with masking.
- **novelty.py**: V2 classical novelty (hash frequency + kNN on Morgan fingerprints).
- **encoder.py**: SMILES → features → AngleEmbedding angles; provides quantum state encoder.
- **prior.py**: Quantum module (V3): geometry embedding, quantum Hamiltonian (PySCF→OpenFermion→JW), valence/connectivity/geometry penalties, short VQE, quantum reward.
- **trainer.py**: Quantum policy gradient (REINFORCE-style) with entropy regularization, temperature annealing, mini-batch updates, logging, convergence plots.
- **config.py**: Hyperparameters (RL, chemistry, quantum).
- **train.py**: Entry point wiring everything together.

## Quantum Reward Architecture
1. **Quantum chemistry**: RDKit geometry → PySCF RHF → OpenFermion fermionic H → JW to qubit H → PennyLane VQE (short run).
2. **Valence penalty**: Σ β (valence(i) − allowed_valence(i))² as a Hamiltonian term.
3. **Connectivity penalty**: δ |components − 1| as a Hamiltonian term.
4. **Geometry penalty**: Σ γ exp(−(d_ij − d0)² / σ²) as a Hamiltonian term.
5. **Quantum novelty**: based on quantum state fidelity (state buffer via encoder; novelty = 1 − avg fidelity).
6. **Combined reward**: `reward = valid * novelty_classical * novelty_quantum * quantum_energy_prior`, where `quantum_energy_prior = exp(−E/λ)` (E = VQE energy + penalties).

## Default Chemical Space
- Chain of 5 heavy atoms (`C, N, O`), bonds `NONE/Single/Double/Triple` along 0–1–2–3–4.

## Config (config.py)
- RL: episodes=20000, lr=0.02, entropy=0.05, temperature=1.5, temp_decay=0.998, min_temp=0.6, batch_size=16.
- Chemistry: allowed atoms, chain length, distance filter.
- Quantum: basis (sto-3g), target_qubits (active space 4–8), VQE layers/steps/step-size, penalty weights, λ for reward scaling.

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

## Outputs
- Logs every 100 steps: reward, valid/unique, novelty (classical/quantum), quantum prior, atoms/bonds, temp, entropy, SMILES.
- Final summary: samples, valid, unique, reward stats, (valid/samples)*(unique/samples), best SMILES, unique list.
- `convergence.png`: valid/samples, unique/samples, their product.

## Extending
- Expand atoms/bonds or chain length in `config.py`.
- Implement true quantum novelty via encoder states and fidelity in `prior.py`.
- Adjust active space/qubits, penalties, λ, and VQE settings for speed/accuracy trade-offs.
- Swap ansatz or integrate CUDA-Q/QMG in `prior.py` if available.

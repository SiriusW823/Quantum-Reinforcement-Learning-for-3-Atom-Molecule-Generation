## Quantum RL for 3‑Atom Molecule Generation (Fully Quantum Agents)

This repository implements a **fully quantum actor–critic** pipeline for small-molecule generation. The environment is classical (RDKit) for molecule construction/validation, while both the **generator (actor)** and the **helper (critic/prior)** are parametrized quantum circuits (PQCs) built with PennyLane. The objective is to maximize:

```
valid_ratio = valid_count  / samples
unique_ratio = unique_valid / samples
target_metric = valid_ratio * unique_ratio  → 1
```

### Key Components
- **Environment (classical, RDKit):** builds SMILES from sampled atoms/bonds, enforces connectivity, tracks validity/uniqueness.
- **Quantum Generator (actor):** PQC (AngleEmbedding + StronglyEntanglingLayers) outputs logits for atom and bond choices; trained via policy gradient with parameter-shift.
- **Quantum Helper (critic/prior):** PQC mapping SMILES fingerprints to a scalar value, shaping reward; optional VQE-based chemistry prior (PySCF + OpenFermion + PennyLane) for physical energy bias.
- **Reward:** default `reward = valid ? novelty_classical * novelty_quantum * quantum_prior : -0.02 (if repeated) or 0 (if invalid)`, then scaled by `(1 + critic_value)`. Novelty includes hash + kNN (classical) and a quantum fidelity proxy.

### Project Structure
```
agents/
  quantum_policy.py      # PQC actor
  quantum_helper.py      # PQC critic/prior
configs/
  config.py              # Python config (default)
  default_3atom.yaml     # YAML snapshot of defaults
env/
  chem.py                # RDKit atom/bond utils, geometry, valence helpers
  molecule_env.py        # Environment wrapper with counters
quantum/
  encoder.py             # SMILES→state encoder (AngleEmbedding)
  encodings.py           # Feature encodings (counts + FP)
  pqc_blocks.py          # Reusable PQC builders
  prior.py               # Optional VQE prior (PySCF + OpenFermion + PennyLane)
training/
  loop.py                # Actor–critic training loop
  novelty.py             # Classical novelty (hash + kNN)
  utils.py               # Plotting utilities
scripts/
  train_3atom_full_quantum.py  # Entry point
LICENSE
README.md
```

### Installation (example, conda)
```bash
# Create env (Python ≥3.11)
conda create -n qrl python=3.11 rdkit -c conda-forge -y
conda activate qrl

# Install core packages
pip install torch pennylane pyscf openfermion openfermionpyscf matplotlib
```
> If PySCF/OpenFermion wheels are unavailable on your platform, install via conda-forge or prebuilt wheels as appropriate.

### Training
Run the full quantum actor–critic (defaults for 3 atoms):
```bash
python scripts/train_3atom_full_quantum.py
```
Optional flag to disable the VQE prior:
```bash
python scripts/train_3atom_full_quantum.py --quantum-prior
```
(flag present; defaults to enabled)

Outputs:
- Console logs every `log_interval` episodes.
- `convergence.png` plotting valid/samples, unique/samples, and their product.

### Configuration
- Python defaults: `configs/config.py`
- YAML snapshot: `configs/default_3atom.yaml` (not auto-loaded; provided for reference/versioning)

Key knobs:
- `chem.allowed_atoms`, `chem.num_atoms` (default 3 heavy atoms: C/N/O)
- `rl.*` (episodes, lr, temperature schedule, entropy coef, batch size)
- `quantum.*` (VQE prior settings, penalty weights, target qubits)
- `encoder.n_qubits` (for quantum encoding)

### How It Works (Pipeline)
1. **Actor PQC** samples atoms/bonds → candidate molecule.
2. **Environment (RDKit)** builds and sanitizes SMILES, checks connectivity, updates validity/uniqueness stats.
3. **Novelty & Prior**
   - Classical novelty: frequency-based hash + kNN on fingerprints.
   - Quantum novelty: fidelity proxy on embedded states.
   - Quantum prior: optional VQE energy score from PySCF/OpenFermion → PennyLane.
4. **Critic PQC** evaluates SMILES features → scalar value.
5. **Reward** combines validity, novelty, prior, and critic bonus; gradients flow via parameter-shift to both actor and critic.
6. Metrics tracked: valid_ratio, unique_ratio, target_metric, convergence plot saved.

### Notes and Limits
- Environment stays classical (RDKit) by design; all learning components are quantum PQCs.
- VQE prior is CPU-only and approximate; disable via `--quantum-prior` if too slow.
- Qubit counts are kept modest (4–8) for tractability on CPU simulators.

### License
See `LICENSE` (unchanged).

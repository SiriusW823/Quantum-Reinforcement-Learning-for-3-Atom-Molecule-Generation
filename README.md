## Full Quantum Reinforcement Learning for 5-Atom Molecule Generation

This project implements a **fully quantum actor–critic** (two Codex) system for de novo molecule generation with up to **5 heavy atoms**. All learning components are Variational Quantum Circuits (VQCs) built in PennyLane; no classical neural networks are used for policy or value estimation. RDKit is used only for classical molecule construction and validity checks.

### Chemistry Constraints (hard-coded)
- Max heavy atoms: 5 (Atom1 → Bond1 → Atom2 → Bond2 → Atom3 → Bond3 → Atom4 → Bond4 → Atom5).
- Allowed atoms: `['NONE', 'C', 'N', 'O']` (NONE = padding/stop).
- Allowed bonds: `['NONE', 'SINGLE', 'DOUBLE', 'TRIPLE']`.
- If an atom action is `NONE`, generation terminates (later positions are padded with NONE; bonds beyond termination are forced to NONE).

### Architecture (Two Codex: Quantum Actor–Critic)
- **Actor (Agent A):** PennyLane VQC with AngleEmbedding + StronglyEntanglingLayers (8–10 qubits). Produces logits (via PQC expectation values) for 4-way discrete actions (atom or bond) at each step.
- **Critic (Agent B):** PennyLane VQC estimating state value `V(s)` from the encoded action history to compute an advantage for policy gradient variance reduction.
- **Reward:** Encourages the golden metric `(valid/samples) * (unique_valid/samples) → 1`. Invalid or duplicate molecules receive near-zero/negative reward; valid-unique molecules receive `+1`.
- **Training Loop:** Quantum policy gradient (REINFORCE with value baseline). No classical linear layers are used for “intelligence”; all learnable parameters reside in the quantum circuits.

### Project Layout
```
.
├── README.md
├── requirements.txt
├── setup_git.sh
├── src/
│   ├── __init__.py
│   ├── circuits.py       # actor_qnode, critic_qnode (StronglyEntanglingLayers)
│   ├── embedding.py      # encode 9-step history -> AngleEmbedding angles
│   ├── environment.py    # RDKit-backed environment for 5-atom linear molecules
│   └── agent.py          # QuantumActorCritic (sampling, loss, updates)
└── train.py              # Training loop, golden metric logging
```

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
> RDKit wheels on some platforms may require conda-forge. If `pip install rdkit-pypi` fails, install RDKit via conda and the rest via pip.

### Run Training
```bash
python train.py
```
Outputs:
- Console logs with reward, valid/unique counts, and the golden metric.
- The policy operates purely with VQC parameters; no classical dense layers.

### Notes
- Qubits: default 10 wires for both actor and critic to accommodate the 9-step history plus slack.
- State encoding: each discrete token (0–3) is mapped to an angle in `[0, π]`; padded to wire count.
- RDKit validity: linear chain assembly; fragments or invalid valence are marked invalid. Unique tracking is by canonical SMILES.
- Reward shaping: `+1` for valid & unique, `-0.1` for valid but duplicate, `0` for invalid. Advantage = reward - V(s).

### Golden Metric
The objective is to push:
```
Score = (Valid Count / Total Samples) * (Unique Valid Count / Total Samples)  → 1.0
```
Training is configured to directly reward this target.

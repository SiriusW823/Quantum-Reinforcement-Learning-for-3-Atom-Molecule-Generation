from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.optim as optim

import configs.config as config
from env.chem import BOND_LABELS, BOND_TYPES, set_seed
from env.molecule_env import MoleculeEnvironment
from agents.quantum_policy import QuantumPolicyNet, sample_action
from agents.quantum_helper import QuantumHelper
from quantum.prior import build_quantum_prior
from training.novelty import ClassicalNovelty
from quantum.encoder import prepare_state
from training.utils import plot_convergence


@dataclass
class History:
    steps: List[int]
    valid_ratio: List[float]
    unique_ratio: List[float]
    combo: List[float]


def run(quantum_prior_enabled: bool = True, critic_coef: float = 0.5) -> None:
    rl = config.rl
    chem = config.chem
    qconf = config.quantum
    set_seed(rl.seed)

    env = MoleculeEnvironment(allowed_atoms=chem.allowed_atoms)
    policy = QuantumPolicyNet(
        n_atoms=len(chem.allowed_atoms),
        n_bonds=len(BOND_TYPES),
        num_atoms_in_chain=chem.num_atoms,
    )
    helper = QuantumHelper()
    optimizer = optim.Adam(list(policy.parameters()) + list(helper.parameters()), lr=rl.lr)

    prior_fn = (
        build_quantum_prior(
            basis=qconf.basis,
            target_qubits=qconf.target_qubits,
            layers=qconf.vqe_layers,
            steps=qconf.vqe_steps,
            stepsize=qconf.vqe_lr,
            energy_scale=qconf.energy_scale,
            score_min=qconf.score_min,
            score_max=qconf.score_max,
            valence_weight=qconf.valence_penalty_weight,
            connectivity_weight=qconf.connectivity_penalty_weight,
            geometry_weight=qconf.geometry_penalty_weight,
            geom_d0=qconf.geom_d0,
            geom_sigma=qconf.geom_sigma,
        )
        if quantum_prior_enabled
        else (lambda x: 1.0)
    )

    novelty_cls = ClassicalNovelty(k=qconf.knn_k)
    state_buffer: List[torch.Tensor] = []
    max_states = 200

    history = History([], [], [], [])
    rewards: List[float] = []
    discovered: List[str] = []
    temperature = rl.temperature
    best_reward = -math.inf
    best_smiles: Optional[str] = None

    for ep in range(1, rl.episodes + 1):
        atoms, bonds, log_prob, entropy = sample_action(
            policy, allowed_atoms=list(chem.allowed_atoms), temperature=temperature
        )
        smiles, valid, unique = env.build_from_actions(atoms, bonds)
        novelty_cls.update(smiles if smiles else "")
        novelty_classical = novelty_cls.novelty(smiles if smiles else "")

        novelty_quantum = 1.0
        quantum_prior = 1.0
        if valid and smiles:
            if unique:
                psi = prepare_state(smiles, config.encoder.n_qubits)()
                state_buffer.append(torch.tensor(psi))
                if len(state_buffer) > max_states:
                    state_buffer.pop(0)
            if len(state_buffer) > 1:
                psi_new = state_buffer[-1]
                fid_list = []
                for psi_old in state_buffer[:-1]:
                    fid = torch.abs(torch.dot(psi_new.conj(), psi_old)) ** 2
                    fid_list.append(fid.item())
                avg_fid = sum(fid_list) / len(fid_list) if fid_list else 0.0
                novelty_quantum = 1.0 - avg_fid
            quantum_prior = prior_fn(smiles)

        if not valid:
            reward = 0.0
        elif unique:
            reward = novelty_classical * novelty_quantum * quantum_prior
        else:
            reward = -0.02

        # Critic value
        value = helper(smiles) if smiles else torch.tensor(0.0)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        advantage = reward_tensor - value.detach()

        actor_loss = -(advantage * log_prob + rl.entropy_coef * entropy)
        critic_loss = 0.5 * (reward_tensor - value) ** 2
        total_loss = actor_loss + critic_coef * critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(helper.parameters()), rl.max_grad_norm)
        optimizer.step()

        if unique and valid:
            discovered.append(smiles)
        rewards.append(reward)

        if reward > best_reward:
            best_reward = reward
            best_smiles = smiles

        if ep % rl.log_interval == 0:
            print(
                f"[ep {ep:05d}] reward={reward:.3f} valid={valid:.0f} unique={unique:.0f} "
                f"nov_cls={novelty_classical:.3f} nov_q={novelty_quantum:.3f} prior={quantum_prior:.3f} "
                f"value={value.item():.3f} "
                f"atoms={[chem.allowed_atoms[a] for a in atoms]} bonds={[BOND_LABELS[BOND_TYPES[b]] for b in bonds]} "
                f"temp={temperature:.2f} H={entropy.item():.2f} smiles={smiles}"
            )

        temperature = max(rl.min_temperature, temperature * rl.temp_decay)
        history.steps.append(ep)
        history.valid_ratio.append(env.valid_ratio)
        history.unique_ratio.append(env.unique_ratio)
        history.combo.append(env.target_metric)

    _summarize(rl.episodes, env.valid_count, discovered, rewards, best_smiles, best_reward, history)


def _summarize(
    episodes: int,
    valid_count: int,
    discovered: List[str],
    rewards: List[float],
    best_smiles: Optional[str],
    best_reward: float,
    history: History,
) -> None:
    print("\n===== Final summary =====")
    print(f"Samples (episodes): {episodes}")
    print(f"Valid count: {valid_count} | Unique valid: {len(discovered)}")
    if rewards:
        print(f"Reward max: {max(rewards):.3f} | mean: {sum(rewards)/len(rewards):.3f}")
    if history.steps:
        combo_final = history.combo[-1]
        print(f"(Valid/Samples)*(Unique/Samples): {combo_final:.4f}")
    print("Top candidate:", best_smiles, "reward=", round(best_reward, 3))
    print("Confirmed unique valid SMILES:")
    for s in discovered:
        print("  ", s)
    try:
        plot_convergence(history.steps, history.valid_ratio, history.unique_ratio, history.combo)
        print("Saved convergence plot to convergence.png")
    except Exception as e:
        print("Plotting failed:", e)


def main(args=None) -> None:
    quantum_prior_enabled = True
    if args is not None and hasattr(args, "quantum_prior"):
        quantum_prior_enabled = bool(args.quantum_prior)
    run(quantum_prior_enabled=quantum_prior_enabled)

from dataclasses import dataclass
from typing import List, Optional
import math
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from . import config
from .chem import BOND_LABELS, BOND_TYPES, atoms_bonds_to_smiles, set_seed
from .policy import PolicyNet, sample_action
from .prior import build_quantum_prior


@dataclass
class History:
    steps: List[int]
    valid_ratio: List[float]
    unique_ratio: List[float]
    combo: List[float]


class Trainer:
    def __init__(self, config):
        self.cfg = config
        self.device = "cpu"

    def run(self) -> None:
        rl_cfg = self.cfg.rl
        chem_cfg = self.cfg.chem
        q_cfg = self.cfg.quantum

        set_seed(rl_cfg.seed)
        prior_fn = build_quantum_prior(
            basis=q_cfg.basis,
            target_qubits=q_cfg.target_qubits,
            layers=q_cfg.vqe_layers,
            steps=q_cfg.vqe_steps,
            stepsize=q_cfg.vqe_lr,
            energy_scale=q_cfg.energy_scale,
            score_min=q_cfg.score_min,
            score_max=q_cfg.score_max,
            valence_weight=q_cfg.valence_penalty_weight,
            connectivity_weight=q_cfg.connectivity_penalty_weight,
            geometry_weight=q_cfg.geometry_penalty_weight,
            geom_d0=q_cfg.geom_d0,
            geom_sigma=q_cfg.geom_sigma,
        )

        policy = PolicyNet(
            n_atoms=len(chem_cfg.allowed_atoms),
            n_bonds=len(BOND_TYPES),
            num_atoms_in_chain=chem_cfg.num_atoms_in_chain,
        ).to(self.device)
        optimizer = optim.Adam(policy.parameters(), lr=rl_cfg.lr)
        baseline = 0.0
        seen_smiles: set[str] = set()
        best_reward = -math.inf
        best_smiles: Optional[str] = None

        history = History([], [], [], [])
        rewards: List[float] = []
        discovered: List[str] = []
        valid_count = 0
        batch_losses: List[torch.Tensor] = []
        temperature = rl_cfg.temperature

        for ep in range(1, rl_cfg.episodes + 1):
            atoms, bonds, log_prob, entropy = sample_action(policy, temperature=temperature)
            smiles, valid = atoms_bonds_to_smiles(atoms, bonds, chem_cfg.allowed_atoms)
            if valid and smiles and "." in smiles and not chem_cfg.allow_disconnected:
                valid = 0.0
            valid_count += int(valid)

            unique = 1.0 if smiles and smiles not in seen_smiles else 0.0
            if unique:
                seen_smiles.add(smiles)
                discovered.append(smiles)

            # Quantum energy and reward
            quantum_energy = prior_fn(smiles) if smiles else 0.0
            quantum_score = math.exp(-quantum_energy / q_cfg.lambda_reward_scale) if valid and smiles else 0.0
            if valid and unique:
                reward = quantum_score
            elif valid and not unique:
                reward = -0.01
            else:
                reward = 0.0
            rewards.append(reward)

            baseline = 0.9 * baseline + 0.1 * reward
            advantage = reward - baseline
            loss = -(advantage * log_prob + rl_cfg.entropy_coef * entropy)
            batch_losses.append(loss)

            if len(batch_losses) >= rl_cfg.batch_size:
                optimizer.zero_grad()
                total_loss = torch.stack(batch_losses).mean()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), rl_cfg.max_grad_norm)
                optimizer.step()
                batch_losses.clear()

            if reward > best_reward:
                best_reward = reward
                best_smiles = smiles

            if ep % rl_cfg.log_interval == 0:
                print(
                    f"[ep {ep:04d}] reward={reward:.3f} valid={valid:.0f} unique={unique:.0f} "
                    f"E={quantum_energy:.3f} atoms={[chem_cfg.allowed_atoms[a] for a in atoms]} "
                    f"bonds={[BOND_LABELS[BOND_TYPES[b]] for b in bonds]} temp={temperature:.2f} "
                    f"H={entropy.item():.2f} smiles={smiles}"
                )

            temperature = max(rl_cfg.min_temperature, temperature * rl_cfg.temp_decay)
            history.steps.append(ep)
            history.valid_ratio.append(valid_count / ep)
            history.unique_ratio.append(len(seen_smiles) / ep)
            history.combo.append(history.valid_ratio[-1] * history.unique_ratio[-1])

        if batch_losses:
            optimizer.zero_grad()
            total_loss = torch.stack(batch_losses).mean()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), rl_cfg.max_grad_norm)
            optimizer.step()

        self._summarize(rl_cfg.episodes, valid_count, discovered, rewards, best_smiles, best_reward, history)

    def _summarize(
        self,
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
            plt.figure(figsize=(8, 4))
            plt.plot(history.steps, history.valid_ratio, label="valid/samples")
            plt.plot(history.steps, history.unique_ratio, label="unique/samples")
            plt.plot(history.steps, history.combo, label="(valid/samples)*(unique/samples)")
            plt.xlabel("Episode")
            plt.ylabel("Ratio")
            plt.title("Convergence")
            plt.legend()
            plt.tight_layout()
            plt.savefig("convergence.png", dpi=150)
            print("Saved convergence plot to convergence.png")
        except Exception as e:
            print("Plotting failed:", e)

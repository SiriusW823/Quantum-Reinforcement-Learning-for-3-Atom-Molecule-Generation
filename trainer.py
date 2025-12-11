from dataclasses import dataclass
from typing import List, Optional
import math
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

import config
from chem import BOND_LABELS, BOND_TYPES, atoms_bonds_to_smiles, set_seed
from policy import PolicyNet, sample_action
from prior import build_quantum_prior
from novelty import ClassicalNovelty


@dataclass
class History:
    steps: List[int]
    valid_ratio: List[float]
    unique_ratio: List[float]
    combo: List[float]


class Trainer:
    def __init__(self):
        self.rl = config.rl
        self.chem = config.chem
        self.q = config.quantum
        self.device = "cpu"

    def run(self) -> None:
        set_seed(self.rl.seed)
        policy = PolicyNet(
            n_atoms=len(self.chem.allowed_atoms),
            n_bonds=len(BOND_TYPES),
            num_atoms_in_chain=self.chem.num_atoms,
        ).to(self.device)
        optimizer = optim.Adam(policy.parameters(), lr=self.rl.lr)
        baseline = 0.0

        prior_fn = build_quantum_prior(
            basis=self.q.basis,
            target_qubits=self.q.target_qubits,
            layers=self.q.vqe_layers,
            steps=self.q.vqe_steps,
            stepsize=self.q.vqe_lr,
            energy_scale=self.q.energy_scale,
            score_min=self.q.score_min,
            score_max=self.q.score_max,
            valence_weight=self.q.valence_penalty_weight,
            connectivity_weight=self.q.connectivity_penalty_weight,
            geometry_weight=self.q.geometry_penalty_weight,
            geom_d0=self.q.geom_d0,
            geom_sigma=self.q.geom_sigma,
            lambda_reward=self.q.lambda_reward,
        )
        novelty_cls = ClassicalNovelty(k=self.q.knn_k)

        history = History([], [], [], [])
        rewards: List[float] = []
        discovered: List[str] = []
        seen_smiles: set[str] = set()
        valid_count = 0
        batch_losses: List[torch.Tensor] = []
        temperature = self.rl.temperature
        best_reward = -math.inf
        best_smiles: Optional[str] = None

        for ep in range(1, self.rl.episodes + 1):
            atoms, bonds, log_prob, entropy = sample_action(policy, temperature=temperature)
            smiles, valid = atoms_bonds_to_smiles(atoms, bonds, self.chem.allowed_atoms)
            if valid and smiles and "." in smiles and not self.chem.allow_disconnected:
                valid = 0.0
            valid_count += int(valid)

            unique = 1.0 if smiles and smiles not in seen_smiles else 0.0
            novelty_cls.update(smiles if smiles else "")
            novelty_classical = novelty_cls.novelty(smiles if smiles else "")
            novelty_quantum = 0.0
            quantum_prior = 0.0

            if valid and smiles:
                quantum_energy = prior_fn(smiles)
                quantum_prior = max(self.q.score_min, min(self.q.score_max, math.exp(-quantum_energy / self.q.lambda_reward)))
                # Quantum novelty via placeholder fidelity (extend with encoder states if desired)
                novelty_quantum = novelty_classical  # placeholder proxy

            reward = (
                valid
                * novelty_classical
                * novelty_quantum
                * quantum_prior
            )
            # Handle uniqueness gating
            if unique == 0.0:
                reward *= 0.5
            if unique:
                seen_smiles.add(smiles)
                discovered.append(smiles)
            rewards.append(reward)

            baseline = 0.9 * baseline + 0.1 * reward
            advantage = reward - baseline
            loss = -(advantage * log_prob + self.rl.entropy_coef * entropy)
            batch_losses.append(loss)

            if len(batch_losses) >= self.rl.batch_size:
                optimizer.zero_grad()
                total_loss = torch.stack(batch_losses).mean()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.rl.max_grad_norm)
                optimizer.step()
                batch_losses.clear()

            if reward > best_reward:
                best_reward = reward
                best_smiles = smiles

            if ep % self.rl.log_interval == 0:
                print(
                    f"[ep {ep:05d}] reward={reward:.3f} valid={valid:.0f} unique={unique:.0f} "
                    f"nov_cls={novelty_classical:.3f} nov_q={novelty_quantum:.3f} prior={quantum_prior:.3f} "
                    f"atoms={[self.chem.allowed_atoms[a] for a in atoms]} bonds={[BOND_LABELS[BOND_TYPES[b]] for b in bonds]} "
                    f"temp={temperature:.2f} H={entropy.item():.2f} smiles={smiles}"
                )

            temperature = max(self.rl.min_temperature, temperature * self.rl.temp_decay)
            history.steps.append(ep)
            history.valid_ratio.append(valid_count / ep)
            history.unique_ratio.append(len(seen_smiles) / ep)
            history.combo.append(history.valid_ratio[-1] * history.unique_ratio[-1])

        if batch_losses:
            optimizer.zero_grad()
            total_loss = torch.stack(batch_losses).mean()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), self.rl.max_grad_norm)
            optimizer.step()

        self._summarize(self.rl.episodes, valid_count, discovered, rewards, best_smiles, best_reward, history)

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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import pennylane as qml
import random

from src.circuits import actor_qnode, critic_qnode
from src.embedding import encode_state
from src.environment import MoleculeEnv, ATOM_TYPES, BOND_TYPES


class QuantumActorCritic(nn.Module):
    """
    Full quantum actor-critic.
    - Actor: PQC -> 4 logits (no linear layers).
    - Critic: PQC -> state value scalar.
    """

    def __init__(self, n_wires: int = 10, layers: int = 2, lr: float = 0.02, entropy_coef: float = 0.01):
        super().__init__()
        self.n_wires = n_wires
        self.entropy_coef = entropy_coef

        act_qnode, act_shapes = actor_qnode(n_wires=n_wires, layers=layers)
        self.actor = qml.qnn.TorchLayer(act_qnode, weight_shapes=act_shapes)

        crt_qnode, crt_shapes = critic_qnode(n_wires=n_wires, layers=layers)
        self.critic = qml.qnn.TorchLayer(crt_qnode, weight_shapes=crt_shapes)

        # Small learnable temperature to rescale logits; still quantum-derived
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def policy(self, state_vec: torch.Tensor) -> D.Categorical:
        logits = self.actor(state_vec) * self.logit_scale
        return D.Categorical(logits=logits)

    def value(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.critic(state_vec)

    def act(self, history) -> tuple[int, torch.Tensor, torch.Tensor]:
        state_vec = encode_state(history, n_wires=self.n_wires)
        dist = self.policy(state_vec)
        action = dist.sample()
        logp = dist.log_prob(action)
        ent = dist.entropy()
        return action.item(), logp, ent

    def update(self, logps, ents, rewards, values, gamma=1.0):
        # Monte Carlo return (single-step episode end)
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns = list(reversed(returns))
        returns_t = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.stack(values)
        logps_t = torch.stack(logps)
        ents_t = torch.stack(ents)

        advantages = returns_t - values_t.detach()
        actor_loss = -(advantages * logps_t).mean() - self.entropy_coef * ents_t.mean()
        critic_loss = 0.5 * (returns_t - values_t).pow(2).mean()
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return loss.item(), actor_loss.item(), critic_loss.item()


def run_episode(agent: QuantumActorCritic, env: MoleculeEnv, seen: set[str]) -> tuple[float, float, float, str | None]:
    env.reset()
    logps = []
    ents = []
    values = []
    history = []
    done = False
    epsilon = 0.2  # exploration rate

    while not done:
        action, logp, ent = agent.act(history)
        # epsilon-greedy exploration
        if random.random() < epsilon:
            if len(history) == 0:
                action = random.choice([1, 2, 3])  # first atom cannot be NONE
            else:
                action = random.randint(0, 3)
            state_vec = encode_state(history, n_wires=agent.n_wires)
            dist = agent.policy(state_vec)
            logp = dist.log_prob(torch.tensor(action))
            ent = dist.entropy()
        # enforce first atom is not NONE (0) to avoid trivial invalid episodes
        if len(history) == 0 and action == 0:
            action = 1  # fallback to Carbon
            state_vec = encode_state(history, n_wires=agent.n_wires)
            dist = agent.policy(state_vec)
            logp = dist.log_prob(torch.tensor(action))
            ent = dist.entropy()
        state_vec = encode_state(history, n_wires=agent.n_wires)
        val = agent.value(state_vec)
        history.append(action)
        logps.append(logp)
        ents.append(ent)
        values.append(val)
        if len(history) >= 9 or (len(history) % 2 == 1 and history[-1] == 0):  # stop if 9 steps or NONE atom
            done = True

    smiles, valid, _ = env.finalize()
    unique = 0.0
    if valid and smiles:
        unique = 1.0 if smiles not in seen else 0.0
        if unique:
            seen.add(smiles)

    # Reward shaping aligned with golden metric
    if valid and unique:
        reward = 1.0
    elif valid and not unique:
        reward = -0.1
    else:
        reward = 0.0

    loss_vals = agent.update(logps, ents, [reward], values)
    return reward, valid, unique, smiles

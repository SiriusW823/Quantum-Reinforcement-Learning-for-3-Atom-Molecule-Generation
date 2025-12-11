import os
import random
import torch

from src.environment import MoleculeEnv
from src.agent import QuantumActorCritic, run_episode


def set_seed(seed: int = 7):
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(7)
    env = MoleculeEnv()
    agent = QuantumActorCritic(n_wires=10, layers=2, lr=0.02, entropy_coef=0.02)

    episodes = 20000
    log_interval = 100
    seen: set[str] = set()
    valid_count = 0
    unique_count = 0
    rewards = []

    for ep in range(1, episodes + 1):
        reward, valid, unique, smiles = run_episode(agent, env, seen)
        rewards.append(reward)
        valid_count += int(valid)
        unique_count += int(unique)
        if ep % log_interval == 0:
            valid_ratio = valid_count / ep
            unique_ratio = unique_count / ep
            score = valid_ratio * unique_ratio
            print(
                f"[ep {ep:05d}] reward={reward:.3f} valid={valid:.0f} unique={unique:.0f} "
                f"score={score:.4f} smiles={smiles}"
            )

    final_score = (valid_count / episodes) * (unique_count / episodes)
    print("\n===== Final summary =====")
    print(f"Episodes: {episodes}")
    print(f"Valid: {valid_count} | Unique valid: {unique_count}")
    print(f"Golden metric (Valid/Samples * Unique/Samples): {final_score:.4f}")


if __name__ == "__main__":
    # For OpenMP duplication safety
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()

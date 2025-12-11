import matplotlib.pyplot as plt
from typing import List


def plot_convergence(steps: List[int], valid_ratio: List[float], unique_ratio: List[float], combo: List[float], path: str = "convergence.png") -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(steps, valid_ratio, label="valid/samples")
    plt.plot(steps, unique_ratio, label="unique/samples")
    plt.plot(steps, combo, label="(valid/samples)*(unique/samples)")
    plt.xlabel("Episode")
    plt.ylabel("Ratio")
    plt.title("Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)

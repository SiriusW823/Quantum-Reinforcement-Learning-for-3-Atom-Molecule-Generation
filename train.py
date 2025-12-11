import os
import argparse
import matplotlib

# Use non-GUI backend for plotting on servers/WSL
matplotlib.use("Agg")

from trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum RL Molecule Generator (Level-5)")
    parser.add_argument("--quantum-policy", action="store_true", default=True, help="Use quantum policy network")
    parser.add_argument("--classical-policy", action="store_true", default=False, help="(placeholder) use classical policy")
    parser.add_argument("--quantum-prior", action="store_true", default=True, help="Use quantum VQE prior")
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    trainer = Trainer(use_quantum_policy=args.quantum_policy, use_quantum_prior=args.quantum_prior)
    trainer.run()


if __name__ == "__main__":
    main()

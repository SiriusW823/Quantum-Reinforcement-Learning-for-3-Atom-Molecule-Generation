import os
import argparse
import sys
import matplotlib

# Use non-GUI backend
matplotlib.use("Agg")

# Ensure repository root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from training.loop import main as loop_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-quantum RL for 3-atom molecule generation")
    parser.add_argument("--quantum-prior", action="store_true", default=True, help="Use quantum VQE prior")
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    loop_main(args)


if __name__ == "__main__":
    main()

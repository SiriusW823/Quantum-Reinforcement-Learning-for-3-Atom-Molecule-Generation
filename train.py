import os
import matplotlib

# Use non-GUI backend for plotting on servers/WSL
matplotlib.use("Agg")

from trainer import Trainer


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    trainer = Trainer()
    trainer.run()


if __name__ == "__main__":
    main()

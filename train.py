import os
from src.trainer import Trainer
from src import config


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    trainer = Trainer(config=config)
    trainer.run()


if __name__ == "__main__":
    main()

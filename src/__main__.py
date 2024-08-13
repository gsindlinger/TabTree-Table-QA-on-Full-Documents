import logging
from .config.from_args import Config


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)

    if Config.run.qa:
        print("Using CUDA")
    else:
        print("Not using CUDA")
    print("Helloooooooo World!")


if __name__ == "__main__":
    main()

import logging

from .pipeline import Pipeline
from .config.from_args import Config


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    pipeline = Pipeline.from_config()

    if Config.run.pipeline:
        print("Run pipeline")
        print(pipeline.invoke())
    else:
        print("Not running pipeline")


if __name__ == "__main__":
    main()

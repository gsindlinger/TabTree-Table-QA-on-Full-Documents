import logging
import sys

from .indexing.indexing_service import IndexingService

from .pipeline import Pipeline
from .config.from_args import Config


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)

    if Config.run.indexing:
        indexing_service = IndexingService.from_config()
        indexing_service.embed_documents()

    if Config.run.pipeline:
        pipeline = Pipeline.from_config()

        while True:
            question = input("Please enter your input. To exit, type 'exit': ")
            if question == "exit":
                break
            print(pipeline.invoke(question=question))
    else:
        print("Not running pipeline")


if __name__ == "__main__":
    main()

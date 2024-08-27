import logging
import sys

from .retrieval.qdrant_store import QdrantVectorStore
from .retrieval.indexing_service import IndexingService

from .pipeline import Pipeline
from .config.from_args import Config


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)

    vector_store = QdrantVectorStore.from_config()
    if Config.run.indexing:
        indexing_service = IndexingService(qdrant_store=vector_store)
        indexing_service.embed_documents()

    if Config.run.pipeline:
        pipeline = Pipeline.from_config(vector_store=vector_store)

        while True:
            question = input("Please enter your input. To exit, type 'exit': ")
            if question == "exit":
                break
            print(pipeline.invoke(question=question))
    else:
        print("Not running pipeline")


if __name__ == "__main__":
    main()

import logging
import sys

from .evaluation.evaluator import Evaluator

from .retrieval.qdrant_store import QdrantVectorStore
from .retrieval.indexing_service import IndexingService

from .pipeline import Pipeline
from .config.from_args import Config


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)

    vector_store = QdrantVectorStore.from_config()
    pipeline = Pipeline.from_config(vector_store=vector_store)
    indexing_service = IndexingService(qdrant_store=vector_store)

    if Config.run.analysis:
        indexing_service.analyze_documents()

    if Config.run.indexing:
        indexing_service.embed_documents()

    if Config.run.pipeline:
        while True:
            question = input("Please enter your input. To exit, type 'exit': ")
            if question == "exit":
                break
            print(pipeline.invoke(question=question))

    if Config.run.evaluation:
        evaluator = Evaluator.from_config(pipeline=pipeline)
        evaluator.evaluate()


if __name__ == "__main__":
    main()

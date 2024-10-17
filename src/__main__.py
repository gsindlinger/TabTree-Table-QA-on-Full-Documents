import logging
import sys
from typing import Tuple

from .evaluation.evaluation_results import EvaluationResults
from .retrieval.document_splitters.document_splitter import DocumentSplitter
from .retrieval.embeddings.custom_embeddings import CustomEmbeddings
from .retrieval.document_preprocessors.preprocess_config import PreprocessConfig
from .evaluation.evaluator import Evaluator

from .retrieval.qdrant_store import QdrantVectorStore
from .retrieval.indexing_service import IndexingService

from .pipeline import Pipeline
from .config.from_args import Config


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)

    preprocess_config = PreprocessConfig.from_config()
    embedding_model = CustomEmbeddings.from_config()
    chunking_model = DocumentSplitter.from_config(
        embeddings=embedding_model, preprocess_config=preprocess_config
    )
    dataset = Config.run.dataset
    retriever_num_documents = Config.run.retriever_num_documents

    pipeline, indexing_service = setup_run_config(
        preprocess_config=preprocess_config,
        embedding_model=embedding_model,
        chunking_model=chunking_model,
        dataset=dataset,
        retriever_num_documents=retriever_num_documents,
    )

    if Config.run.indexing:
        indexing_service.embed_documents(
            preprocess_config=preprocess_config,
            overwrite_existing_collection=True,
        )

    if Config.run.pipeline:
        while True:
            question = input("Please enter your input. To exit, type 'exit': ")
            if question == "exit":
                break
            print(pipeline.invoke(question=question))

    if Config.run.evaluation:
        evaluation_single = Evaluator.run_single_evaluation(
            indexing_service=indexing_service,
            preprocess_config=preprocess_config,
            dataset=dataset,
            pipeline=pipeline,
            retriever_num_documents=retriever_num_documents,
        )

        if evaluation_single.evaluation_results:
            Evaluator.save_evaluation_results(
                evaluation_results=[evaluation_single.evaluation_results],
                retriever_num_documents=retriever_num_documents,
                names=[preprocess_config.name],
            )

    if Config.run.evaluation_multi:
        preprocess_configs = PreprocessConfig.from_config_multi()
        for retriever_num_documents in [3]:
            evaluation_results = []
            for preprocess_config in preprocess_configs:
                chunking_model = DocumentSplitter.from_config(
                    embeddings=embedding_model, preprocess_config=preprocess_config
                )
                pipeline, indexing_service = setup_run_config(
                    preprocess_config=preprocess_config,
                    embedding_model=embedding_model,
                    chunking_model=chunking_model,
                    dataset=dataset,
                    retriever_num_documents=retriever_num_documents,
                )

                single_evaluation = Evaluator.run_single_evaluation(
                    indexing_service=indexing_service,
                    preprocess_config=preprocess_config,
                    dataset=dataset,
                    pipeline=pipeline,
                    retriever_num_documents=retriever_num_documents,
                )

                evaluation_results.append(single_evaluation.evaluation_results)

            Evaluator.save_evaluation_results(
                evaluation_results=evaluation_results,
                retriever_num_documents=retriever_num_documents,
                names=[
                    preprocess_config.name for preprocess_config in preprocess_configs
                ],
            )


def setup_run_config(
    preprocess_config: PreprocessConfig,
    embedding_model: CustomEmbeddings,
    chunking_model: DocumentSplitter,
    dataset: str,
    retriever_num_documents: int,
) -> Tuple[Pipeline, IndexingService]:
    collection_name = generate_collection_name(
        dataset=dataset,
        embedding_model=embedding_model.get_model_name_stripped(),
        chunking_strategy=chunking_model.name,
        preprocess_mode=preprocess_config.name,
    )

    vector_store = QdrantVectorStore.from_config(
        embedding_model=embedding_model,
        collection_name=collection_name,
    )
    pipeline = Pipeline.from_config(
        vector_store=vector_store, retriever_num_documents=retriever_num_documents
    )
    indexing_service = IndexingService(vector_store=vector_store)

    return pipeline, indexing_service


def generate_collection_name(
    dataset: str, embedding_model: str, chunking_strategy: str, preprocess_mode: str
) -> str:
    return f"{dataset}-{embedding_model}-{chunking_strategy}-{preprocess_mode}"


if __name__ == "__main__":
    main()

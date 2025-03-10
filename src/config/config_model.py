from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from pydantic import BaseModel

from ..retrieval.qdrant_store import QdrantVectorStore
from ..pipeline import AbstractPipeline, RAGPipeline, TableQAPipeline
from ..retrieval.indexing_service import IndexingService
from .config import Config
from ..retrieval.document_splitters.document_splitter import DocumentSplitter
from ..retrieval.document_preprocessors.table_serializer import TableSerializer
from ..retrieval.embeddings.custom_embeddings import CustomEmbeddings
from ..retrieval.document_preprocessors.preprocess_config import PreprocessConfig


class RunSetup(ABC, BaseModel):
    pipeline: AbstractPipeline

    @classmethod
    @abstractmethod
    def run_setup(cls, LLMConfig) -> RunSetup:
        pass


class RunSetupTableQA(RunSetup):
    pipeline: TableQAPipeline

    @classmethod
    def run_setup(cls) -> RunSetupTableQA:
        pipeline = TableQAPipeline.from_config()
        return cls(pipeline=pipeline)


class RunSetupRAG(RunSetup):
    indexing_service: IndexingService
    pipeline: RAGPipeline

    @classmethod
    def run_setup(cls, rag_config: RAGConfig) -> RunSetupRAG:
        collection_name = rag_config.generate_collection_name()

        vector_store = QdrantVectorStore.from_config(
            embedding_model=rag_config.embedding_model,
            collection_name=collection_name,
        )
        pipeline = RAGPipeline.from_config(
            vector_store=vector_store,
            retriever_num_documents=rag_config.retriever_num_documents,
        )
        indexing_service = IndexingService(vector_store=vector_store)

        return cls(
            pipeline=pipeline,
            indexing_service=indexing_service,
        )


class LLMConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    preprocess_config: PreprocessConfig
    table_serializer: TableSerializer | None
    dataset: str

    @classmethod
    def from_config(
        cls, preprocess_config: Optional[PreprocessConfig] = None
    ) -> LLMConfig:
        if not preprocess_config:
            preprocess_config = PreprocessConfig.from_config()
        table_serializer = TableSerializer.from_preprocess_config(preprocess_config)
        dataset = Config.run.dataset
        return cls(
            preprocess_config=preprocess_config,
            table_serializer=table_serializer,
            dataset=dataset,
        )


class RAGConfig(LLMConfig):
    embedding_model: CustomEmbeddings
    chunking_model: DocumentSplitter
    retriever_num_documents: int

    @classmethod
    def from_config(
        cls, preprocess_config: Optional[PreprocessConfig] = None
    ) -> RAGConfig:
        llm_config = LLMConfig.from_config(preprocess_config=preprocess_config)
        embedding_model = CustomEmbeddings.from_config()
        chunking_model = DocumentSplitter.from_config(
            embeddings=embedding_model,
            preprocess_config=llm_config.preprocess_config,
        )
        retriever_num_documents = Config.run.retriever_num_documents
        if isinstance(retriever_num_documents, list) or isinstance(
            retriever_num_documents, tuple
        ):
            retriever_num_documents = retriever_num_documents[0]

        return cls(
            preprocess_config=llm_config.preprocess_config,
            embedding_model=embedding_model,
            table_serializer=llm_config.table_serializer,
            chunking_model=chunking_model,
            dataset=llm_config.dataset,
            retriever_num_documents=retriever_num_documents,
        )

    def generate_collection_name(self) -> str:
        return f"{self.dataset}-{self.embedding_model.get_model_name_stripped()}-{self.chunking_model.name}-{self.preprocess_config.name}"

    def update_by_preprocess_config(
        self, preprocess_config: PreprocessConfig
    ) -> RAGConfig:
        self.table_serializer = TableSerializer.from_preprocess_config(
            preprocess_config
        )
        self.chunking_model = DocumentSplitter.from_config(
            embeddings=self.embedding_model,
            preprocess_config=preprocess_config,
        )
        return self

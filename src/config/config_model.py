from __future__ import annotations
from pydantic import BaseModel

from ..retrieval.qdrant_store import QdrantVectorStore
from ..pipeline import Pipeline
from ..retrieval.indexing_service import IndexingService
from .config import Config
from ..retrieval.document_splitters.document_splitter import DocumentSplitter
from ..retrieval.document_preprocessors.table_serializer import TableSerializer
from ..retrieval.embeddings.custom_embeddings import CustomEmbeddings
from ..retrieval.document_preprocessors.preprocess_config import PreprocessConfig


class RunConfig(BaseModel):
    pipeline: Pipeline
    indexing_service: IndexingService


class GeneralConfig(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    preprocess_config: PreprocessConfig
    embedding_model: CustomEmbeddings
    table_serializer: TableSerializer | None
    chunking_model: DocumentSplitter
    dataset: str
    retriever_num_documents: int

    @classmethod
    def from_config(cls) -> GeneralConfig:
        preprocess_config = PreprocessConfig.from_config()
        embedding_model = CustomEmbeddings.from_config()
        table_serializer = TableSerializer.from_preprocess_config(preprocess_config)
        chunking_model = DocumentSplitter.from_config(
            embeddings=embedding_model,
            preprocess_config=preprocess_config,
        )
        dataset = Config.run.dataset  # dataset name

        retriever_num_documents = Config.run.retriever_num_documents

        if isinstance(retriever_num_documents, list) or isinstance(
            retriever_num_documents, tuple
        ):
            retriever_num_documents = retriever_num_documents[0]

        return cls(
            preprocess_config=preprocess_config,
            embedding_model=embedding_model,
            table_serializer=table_serializer,
            chunking_model=chunking_model,
            dataset=dataset,
            retriever_num_documents=retriever_num_documents,
        )

    def setup_run_config(
        self,
    ) -> RunConfig:
        collection_name = self.generate_collection_name()

        vector_store = QdrantVectorStore.from_config(
            embedding_model=self.embedding_model,
            collection_name=collection_name,
        )
        pipeline = Pipeline.from_config(
            vector_store=vector_store,
            retriever_num_documents=self.retriever_num_documents,
        )
        indexing_service = IndexingService(vector_store=vector_store)

        return RunConfig(
            pipeline=pipeline,
            indexing_service=indexing_service,
        )

    def generate_collection_name(self) -> str:
        return f"{self.dataset}-{self.embedding_model.get_model_name_stripped()}-{self.chunking_model.name}-{self.preprocess_config.name}"

    def update_by_preprocess_config(
        self, preprocess_config: PreprocessConfig
    ) -> GeneralConfig:
        self.table_serializer = TableSerializer.from_preprocess_config(
            preprocess_config
        )
        self.chunking_model = DocumentSplitter.from_config(
            embeddings=self.embedding_model,
            preprocess_config=preprocess_config,
        )
        return self

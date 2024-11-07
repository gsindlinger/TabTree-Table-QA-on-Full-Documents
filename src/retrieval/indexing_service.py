import logging
from typing import List
from pydantic import BaseModel

from .document_preprocessors.preprocess_config import PreprocessConfig
from .document_preprocessors.document_preprocessor import DocumentPreprocessor

from .document_splitters.document_splitter import DocumentSplitter
from .document_loaders.document_loader import DocumentLoader
from .full_document_storage.local_store import LocalStore
from .qdrant_store import QdrantVectorStore


class IndexingService(BaseModel):
    vector_store: QdrantVectorStore

    class Config:
        arbitrary_types_allowed = True

    def embed_documents(
        self,
        preprocess_config: PreprocessConfig,
        overwrite_existing_collection: bool = False,
    ) -> None:
        # Check if collection exists, if so overwrite / delete it or return depending the overwrite_existing_collection flag
        if self.vector_store.client.is_populated(
            collection_name=self.vector_store.collection_name, accept_empty=False
        ):
            if overwrite_existing_collection:
                self.vector_store.client.delete_collection(
                    collection_name=self.vector_store.collection_name
                )
                logging.info(
                    f"Deleted existing collection {self.vector_store.collection_name}"
                )
            else:
                logging.info(
                    f"Collection {self.vector_store.collection_name} already exists, skipping indexing"
                )
                return

        # Load & Preprocess documents
        document_loader = DocumentLoader.from_config()
        documents = [document_loader.load_single_document()]

        # documents = document_loader.load_documents(
        #     num_of_documents=10
        # )

        # Perform preprocessing
        document_preprocessor = DocumentPreprocessor.from_config(
            preprocess_config=preprocess_config
        )
        documents = document_preprocessor.preprocess_multiple_documents(documents)

        # Store full documents
        local_store = LocalStore.from_preprocess_config(
            preprocess_config=preprocess_config
        )
        initial_number_of_documents = local_store.store_full_documents(
            documents=documents
        )

        # Split documents
        document_splitter = DocumentSplitter.from_config(
            embeddings=self.vector_store.embeddings,
            preprocess_config=preprocess_config,
            table_serializer=document_preprocessor.table_serializer,
        )
        documents = document_splitter.split_documents(
            documents=documents,
            ignore_tables_for_embeddings=(
                preprocess_config.ignore_tables_for_embeddings
                if preprocess_config.ignore_tables_for_embeddings
                else False
            ),
        )

        # Embed documents
        vectors = self.vector_store.embeddings.embed_documents(
            texts=[doc.page_content for doc in documents]
        )
        logging.info(
            f"Generated {len(vectors)} chunks for {initial_number_of_documents} documents"
        )

        # Store documents using Qdrant client
        self.vector_store.client.add_documents(
            documents=documents,
            collection_name=self.vector_store.collection_name,
            vectors=vectors,
        )

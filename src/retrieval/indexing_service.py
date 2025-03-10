import logging
from typing import List, Optional
from pydantic import BaseModel

from ..model.custom_document import CustomDocument
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
        self.check_collection_existant_handler(overwrite_existing_collection)

        # Load & Preprocess documents
        documents = [self.load_and_preprocess_document(preprocess_config)]

        # Store full documents locally for debugging purposes
        initial_number_of_documents = self.store_documents(documents, preprocess_config)

        # Split documents into chunks
        documents = self.split_documents(documents, preprocess_config)

        # Generate embeddings for each chunk
        vectors = self.generate_embeddings(documents)
        logging.info(
            f"Generated {len(vectors)} chunks for {initial_number_of_documents} documents"
        )

        # Store documents in Vector store
        self.store_data_in_vectorstore(documents, vectors)

    @staticmethod
    def load_and_preprocess_document(
        preprocess_config: PreprocessConfig,
    ) -> CustomDocument:
        # Load & Preprocess documents
        document_loader = DocumentLoader.from_config()
        document = document_loader.load_single_document()

        # Perform preprocessing
        document_preprocessor = DocumentPreprocessor.from_config(
            preprocess_config=preprocess_config
        )
        document = document_preprocessor.preprocess_document(document)
        return document

    @staticmethod
    def load_and_preprocess_documents(
        preprocess_config: PreprocessConfig,
        num_of_documents: Optional[int] = None,
        id_list: Optional[List[str]] = None,
    ) -> List[CustomDocument]:
        # Load & Preprocess documents
        document_loader = DocumentLoader.from_config()

        if id_list:
            documents = [document_loader.load_single_document(id=id) for id in id_list]
            if num_of_documents:
                documents = documents[:num_of_documents]
        else:
            documents = document_loader.load_documents(
                num_of_documents=num_of_documents
            )

        # Perform preprocessing
        document_preprocessor = DocumentPreprocessor.from_config(
            preprocess_config=preprocess_config
        )
        documents = document_preprocessor.preprocess_multiple_documents(documents)
        return documents

    def store_documents(
        self, documents: List[CustomDocument], preprocess_config: PreprocessConfig
    ) -> int:
        # Store full documents
        local_store = LocalStore.from_preprocess_config(
            preprocess_config=preprocess_config
        )
        initial_number_of_documents = local_store.store_full_documents(
            documents=documents  # type: ignore
        )
        local_store.store_tables(documents=documents)  # type: ignore
        return initial_number_of_documents

    def split_documents(
        self, documents: List[CustomDocument], preprocess_config: PreprocessConfig
    ) -> List[CustomDocument]:
        # Split documents
        document_splitter = DocumentSplitter.from_config(
            embeddings=self.vector_store.embeddings,  # type: ignore
            preprocess_config=preprocess_config,
        )
        documents = document_splitter.split_documents(
            documents=documents,
            ignore_tables_for_embeddings=(
                preprocess_config.ignore_tables_for_embeddings
                if preprocess_config.ignore_tables_for_embeddings
                else False
            ),
        )
        return documents

    def generate_embeddings(self, documents: List[CustomDocument]) -> List[List[float]]:
        # Embed documents
        vectors = self.vector_store.embeddings.embed_documents(
            texts=[doc.page_content for doc in documents]
        )
        return vectors

    def store_data_in_vectorstore(
        self, documents: List[CustomDocument], vectors: List[List[float]]
    ) -> None:
        # Store documents using Vector store client
        self.vector_store.client.add_documents(
            documents=documents,
            collection_name=self.vector_store.collection_name,
            vectors=vectors,
        )

    def check_collection_existant_handler(
        self, overwrite_existing_collection: bool
    ) -> None:
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

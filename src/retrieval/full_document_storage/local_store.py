from __future__ import annotations
import json
import logging
import os
from typing import List, Optional

from pydantic import Field

from ..document_preprocessors.preprocess_config import PreprocessConfig
from ...config import Config
from ...model.custom_document import (
    CustomDocument,
    CustomDocumentWithMetadata,
    FullMetadata,
)
from .document_store import FullDocumentStore


class LocalStore(FullDocumentStore):
    file_ending: str = Field(default="test.html")
    path: str

    @classmethod
    def from_preprocess_config(cls, preprocess_config: PreprocessConfig) -> LocalStore:
        base_path = f"{Config.indexing.full_document_storage_path}{Config.run.dataset}"
        return cls(path=f"{base_path}/{preprocess_config.name}/", file_ending="html")

    def store_tables(self, documents: List[CustomDocumentWithMetadata]):
        # Ensure the directory exists
        table_path = os.path.join(self.path, "tables/")
        os.makedirs(table_path, exist_ok=True)

        for document in documents:
            if not document.splitted_content:
                continue

            table_data = [
                chunk.content
                for chunk in document.splitted_content
                if chunk.type == "table"
            ]
            table_string = "\n\n-------------------\n\n".join(table_data)

        try:
            file_path = os.path.join(table_path, document.metadata.doc_id + ".txt")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(table_string)
                logging.info(f"Stored tables locally")
        except Exception as e:
            logging.error(f"Failed to store tables: {e}")
            
    def store_splits(self, chunks: List[CustomDocument]):
        """ Stores the document splits locally in the specified directory.
        Attention: Input is a flattened list of all chunks of all documents.
        """
        # Ensure the directory exists
        split_path = os.path.join(self.path, "document_splits/")
        os.makedirs(split_path, exist_ok=True)
        
        doc_ids = set([doc.metadata.doc_id for doc in chunks]) # type: ignore
        for doc_id in doc_ids:
            chunks = [doc for doc in chunks if doc.metadata.doc_id == doc_id] # type: ignore
            split_data = [{i: chunk.page_content} for i, chunk in enumerate(chunks)]            
            try:
                file_path = os.path.join(split_path, doc_id + ".json") # type: ignore ; use first chunk id as file name
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(split_data, f, indent=4)
                logging.info(f"Stored splits locally for document {doc_id}")
            except Exception as e:
                logging.error(f"Failed to store splits: {e}")

    def store_full_documents(
        self,
        documents: List[CustomDocumentWithMetadata],
        file_ending: Optional[str] = None,
    ) -> int:
        """Stores the documents locally in the specified directory.

        Args:
            documents (List[CustomDocument]): List of documents to store.
            file_ending (str, optional): File ending to use for the stored documents. Defaults to "html".

        Returns:
            int: Number of documents stored.
        """
        if not file_ending:
            file_ending = self.file_ending

        # Ensure the directory exists
        os.makedirs(self.path, exist_ok=True)

        for document in documents:
            file_path = os.path.join(
                self.path, document.metadata.doc_id + "." + file_ending
            )
            try:
                page_content = document.page_content
                logging.info(f"Document character count: {len(page_content)}")
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(page_content)
                    logging.info(f"Stored document {document.metadata.doc_id} locally")
            except Exception as e:
                logging.error(
                    f"Failed to store document {document.metadata.doc_id}: {e}"
                )

        return len(documents)

    def get_document_by_id(
        self, id: str, file_ending: Optional[str] = None
    ) -> CustomDocument:
        if not file_ending:
            file_ending = self.file_ending

        with open(self.path + id + "." + file_ending, "r", encoding="utf-8") as file:
            html_content = file.read()
            return CustomDocument(
                page_content=html_content,
                metadata=FullMetadata(doc_id=id),
            )

    def get_unique_documents(
        self, documents: List[CustomDocumentWithMetadata]
    ) -> List[CustomDocument]:

        unique_docs = {doc.metadata.doc_id for doc in documents}
        return [self.get_document_by_id(doc_id) for doc_id in unique_docs]

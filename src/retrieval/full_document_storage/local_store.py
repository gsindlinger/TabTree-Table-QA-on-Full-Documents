from __future__ import annotations
import logging
import os
from typing import List, Optional

from pydantic import Field

from ..document_preprocessors.preprocess_config import PreprocessConfig
from ...config import Config
from ...model.custom_document import CustomDocument, FullMetadata
from .document_store import FullDocumentStore


class LocalStore(FullDocumentStore):
    file_ending: str = Field(default="test.html")
    path: str

    @classmethod
    def from_preprocess_config(cls, preprocess_config: PreprocessConfig) -> LocalStore:
        base_path = f"{Config.indexing.full_document_storage_path}/{Config.run.dataset}"
        return cls(path=f"{base_path}/{preprocess_config.name}/", file_ending="html")

    def store_full_documents(
        self, documents: List[CustomDocument], file_ending: Optional[str] = None
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
                logging.info(f"Document character count: {len(document.page_content)}")
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(document.page_content)
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
        self, documents: List[CustomDocument]
    ) -> List[CustomDocument]:

        unique_docs = {doc.metadata.doc_id for doc in documents}
        return [self.get_document_by_id(doc_id) for doc_id in unique_docs]

import logging
import os
from typing import List

from pydantic import Field

from ...config import Config
from ...model.custom_document import CustomDocument, FullMetadata
from .document_store import FullDocumentStore


class LocalStore(FullDocumentStore):
    path: str = Field(default_factory=lambda: Config.full_document_storage.local_path)


class LocalStore(FullDocumentStore):
    path: str = Field(default_factory=lambda: Config.full_document_storage.local_path)

    def store_full_documents(
        self, documents: List[CustomDocument], file_ending: str = "html_1"
    ) -> int:
        """Stores the documents locally in the specified directory.

        Args:
            documents (List[CustomDocument]): List of documents to store.
            file_ending (str, optional): File ending to use for the stored documents. Defaults to "html".

        Returns:
            int: Number of documents stored.
        """
        # Ensure the directory exists
        os.makedirs(self.path, exist_ok=True)

        for document in documents:
            file_path = os.path.join(
                self.path, document.metadata.doc_id + "." + file_ending
            )
            try:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(document.page_content)
                    logging.info(f"Stored document {document.metadata.doc_id} locally")
            except Exception as e:
                logging.error(
                    f"Failed to store document {document.metadata.doc_id}: {e}"
                )

        return len(documents)

    def get_document_by_id(
        self, id: str, file_ending: str = "html_1"
    ) -> CustomDocument:
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

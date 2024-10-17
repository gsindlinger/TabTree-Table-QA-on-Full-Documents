import glob
import logging
import os
import re
from typing import List, Optional, Tuple
import unicodedata

from bs4 import BeautifulSoup
from pydantic import Field

from .document_loader import DocumentLoader
from ...config.config import Config
from ...model.custom_document import CustomDocument, FullMetadata


class SECFilingLoader(DocumentLoader):
    folder_path: str = Field(default_factory=lambda: Config.sec_filings.data_path)
    file_encoding: str = Field(default="ascii")

    def load_single_document(self) -> CustomDocument:
        return self.load_single_document_with_file_path(
            file_path=Config.sec_filings.data_path_single
        )

    def load_single_document_with_file_path(
        self,
        file_path: Optional[str] = None,
    ) -> CustomDocument:
        """Loads a single document (in this case from local storage) and calls preprocessing method."""

        if not file_path:
            file_path = glob.glob(self.folder_path + "*")[0]

        with open(file_path, "r", encoding=self.file_encoding) as file:
            html_content = file.read()
            filing = CustomDocument(
                page_content=html_content,
                metadata=FullMetadata(doc_id=os.path.basename(file_path)),
            )
            logging.info(f"Loaded document {filing.metadata.doc_id}")
            return filing

    def load_documents(
        self,
        num_of_documents: Optional[int] = None,
    ) -> List[CustomDocument]:

        documents = []
        count = 0
        for file_path in glob.glob(self.folder_path + "*"):
            if num_of_documents and count == num_of_documents:
                break
            with open(file_path, "r", encoding="utf-8") as file:
                html_content = file.read()
                filing = CustomDocument(
                    page_content=html_content,
                    metadata=FullMetadata(doc_id=os.path.basename(file_path)),
                )

            # Append to documents
            documents.append(filing)
            count += 1
        return documents

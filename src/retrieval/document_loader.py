from abc import ABC, abstractmethod
import glob
import logging
import os
from typing import Generator, List

from ..model.custom_document import CustomDocument, FullMetadata
from ..model.sec_filing import SECFiling


class DocumentLoader(ABC):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    @abstractmethod
    def load_documents(self) -> Generator[CustomDocument, None, None]:
        pass


class SECFilingLoader:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def load_documents(self) -> List[CustomDocument]:
        documents = []

        # Load only 5 documents for testing
        count = 0
        for file_path in glob.glob(self.folder_path + "*"):
            if count == 2:
                break
            with open(file_path, "r", encoding="utf-8") as file:
                html_content = file.read()

                filing = CustomDocument(
                    page_content=html_content,
                    metadata=FullMetadata(doc_id=os.path.basename(file_path)),
                )
                documents.append(filing)
                logging.info(f"Loaded document {filing.metadata.doc_id}")
            count += 1
        return documents

from abc import ABC, abstractmethod
import glob
import logging
import os
from typing import Generator, List

from ..model.document import Document
from ..model.sec_filing import SECFiling


class DocumentLoader(ABC):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    @abstractmethod
    def load_documents(self) -> Generator[Document, None, None]:
        pass


class SECFilingLoader:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def load_documents(self) -> List[SECFiling]:
        documents = []

        # Load only 5 documents for testing
        count = 0
        for file_path in glob.glob(self.folder_path + "*"):
            if count == 5:
                break
            with open(file_path, "r", encoding="utf-8") as file:
                html_content = file.read()
                filing = SECFiling(
                    text=html_content, doc_id=os.path.basename(file_path)
                )
                documents.append(filing)
                logging.info(f"Loaded document {filing.doc_id}")
            count += 1
        return documents

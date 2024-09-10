from abc import ABC, abstractmethod
import logging
from typing import List, Optional
from langchain_text_splitters import TextSplitter
from pydantic import BaseModel, Field

from ...config.config import Config
from ...model.custom_document import CustomDocument


class DocumentSplitter(ABC, BaseModel):
    text_splitter: Optional[TextSplitter] = None
    chunk_size: int = Field(6000)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config(cls):
        if Config.run.mode == "sec-filings":
            from .sec_filing_splitter import SECFilingSplitter

            return SECFilingSplitter()

        elif Config.run.mode == "sec-filings-html-splitter":
            from .sec_filing_splitter import SECFilingSplitterHTML

            return SECFilingSplitterHTML()

        else:
            raise ValueError(f"Unknown mode: {Config.run.mode}")

    def split_document(self, document: CustomDocument) -> List[CustomDocument]:
        splitted_text = self.text_splitter.split_text(document.page_content)
        return [
            CustomDocument(
                page_content=chunk,
                metadata=document.extend_metadata(chunk_id=i),
            )
            for i, chunk in enumerate(splitted_text)
        ]

    def split_documents(self, documents: List[CustomDocument]) -> List[CustomDocument]:
        splitted_documents = []
        for document in documents:
            splitted_documents.extend(self.split_document(document))

        logging.info(
            f"Split {len(documents)} documents into {len(splitted_documents)} chunks"
        )
        return splitted_documents

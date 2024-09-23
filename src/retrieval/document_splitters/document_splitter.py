from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Tuple
from langchain_text_splitters import TextSplitter
from pydantic import BaseModel

from ...config.config import Config
from ...model.custom_document import CustomDocument


class DocumentSplitter(ABC, BaseModel):
    text_splitter: Optional[TextSplitter | Tuple[TextSplitter]] = None
    chunk_size: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config(cls, embeddings) -> DocumentSplitter:
        match Config.indexing.chunking_strategy:
            case "recursive-character":
                from .sec_filing_splitter import SECFilingSplitter

                return SECFilingSplitter(chunk_size=500)
            case "recursive-character-html":
                from .sec_filing_splitter import SECFilingSplitterHTML

                return SECFilingSplitterHTML(chunk_size=500)

            case "semantic":
                from .sec_filing_splitter import SECFilingSplitterSemantic

                return SECFilingSplitterSemantic(embeddings=embeddings)

            case _:
                raise ValueError(f"Unknown mode: {Config.indexing.chunking_strategy}")

    def split_document(self, document: CustomDocument) -> List[CustomDocument]:
        if document.splitted_content:
            splitted_text = [
                self.text_splitter.split_text(section)
                for section in document.splitted_content
            ]
            # flatten the list
            splitted_text = [item for sublist in splitted_text for item in sublist]
        else:
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

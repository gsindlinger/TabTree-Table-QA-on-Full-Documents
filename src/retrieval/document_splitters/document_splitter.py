from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Tuple
from langchain_text_splitters import TextSplitter
from pydantic import BaseModel

from ..document_preprocessors.preprocess_config import PreprocessConfig
from ..embeddings.custom_embeddings import CustomEmbeddings
from ...config.config import Config
from ...model.custom_document import CustomDocument, SplitContent
from langchain_core.embeddings import Embeddings


class DocumentSplitter(ABC, BaseModel):
    text_splitter: Optional[TextSplitter] = None
    chunk_size: Optional[int] = None
    name: str

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config(
        cls,
        embeddings: Embeddings,
        preprocess_config: PreprocessConfig,
    ) -> DocumentSplitter:
        match Config.indexing.chunking_strategy:
            case "recursive-character":
                from .sec_filing_splitter import SECFilingSplitter

                return SECFilingSplitter(
                    name=Config.indexing.chunking_strategy, chunk_size=500
                )
            case "recursive-character-html":
                from .sec_filing_splitter import SECFilingSplitterHTML

                return SECFilingSplitterHTML(
                    name=Config.indexing.chunking_strategy, chunk_size=8000
                )

            case "semantic":
                from .sec_filing_splitter import SECFilingSplitterSemantic

                return SECFilingSplitterSemantic(
                    name=Config.indexing.chunking_strategy,
                    embeddings=embeddings,
                    chunk_size=10000,
                    ignore_tables_for_embeddings=preprocess_config.ignore_tables_for_embeddings,
                )

            case _:
                raise ValueError(f"Unknown mode: {Config.indexing.chunking_strategy}")

    def split_document(
        self, document: CustomDocument, ignore_tables_for_embeddings: str = False
    ) -> List[CustomDocument]:
        if ignore_tables_for_embeddings:
            splitted_text = self.split_text(document.splitted_content)
        else:
            splitted_text = self.split_text(document.page_content)
        return [
            CustomDocument(
                page_content=chunk,
                metadata=document.extend_metadata(chunk_id=i),
            )
            for i, chunk in enumerate(splitted_text)
        ]

    def split_documents(
        self,
        documents: List[CustomDocument],
        ignore_tables_for_embeddings: bool = False,
    ) -> List[CustomDocument]:
        splitted_documents = []
        for document in documents:
            splitted_documents.extend(
                self.split_document(
                    document=document,
                    ignore_tables_for_embeddings=ignore_tables_for_embeddings,
                )
            )

        logging.info(
            f"Split {len(documents)} documents into {len(splitted_documents)} chunks"
        )
        return splitted_documents

    def split_text(self, text: str | List[SplitContent]) -> List[str]:
        return self.text_splitter.split_text(text)

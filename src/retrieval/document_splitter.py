from abc import ABC, abstractmethod
import logging
from typing import List
from langchain_text_splitters import Language
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..model.custom_document import CustomDocument, FullMetadata
from ..model.sec_filing import SECFiling


class DocumentSplitter(ABC, BaseModel):

    @abstractmethod
    def split_document(self, document: CustomDocument) -> List[CustomDocument]:
        pass

    def split_documents(self, documents: List[CustomDocument]) -> List[CustomDocument]:
        splitted_documents = []
        for document in documents:
            splitted_documents.extend(self.split_document(document))

        logging.info(
            f"Split {len(documents)} documents into {len(splitted_documents)} chunks"
        )
        return splitted_documents


class SECFilingSplitter(DocumentSplitter):
    def split_document(self, document: CustomDocument) -> List[CustomDocument]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splitted_text = text_splitter.split_text(document.page_content)
        return [
            CustomDocument(
                page_content=chunk,
                metadata=document.extend_metadata(chunk_id=i),
            )
            for i, chunk in enumerate(splitted_text)
        ]

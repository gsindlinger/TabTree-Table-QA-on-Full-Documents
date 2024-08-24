from abc import ABC, abstractmethod
import logging
from typing import List
from langchain_text_splitters import Language
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..model.document import Document
from ..model.sec_filing import SECFiling


class DocumentSplitter(ABC, BaseModel):

    @abstractmethod
    def split_document(self, document: Document) -> List[Document]:
        pass

    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitted_documents = []
        for document in documents:
            splitted_documents.extend(self.split_document(document))

        logging.info(
            f"Split {len(documents)} documents into {len(splitted_documents)} chunks"
        )
        return splitted_documents


class SECFilingSplitter(DocumentSplitter):
    def split_document(self, document: Document) -> List[SECFiling]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splitted_text = text_splitter.split_text(document.text)
        return [
            SECFiling(
                text=chunk,
                metadata=document.metadata,
                doc_id=document.doc_id,
                chunk_id=i + 1,
            )
            for i, chunk in enumerate(splitted_text)
        ]

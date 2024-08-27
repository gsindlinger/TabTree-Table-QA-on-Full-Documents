from __future__ import annotations
from typing import Dict, List, Optional
from langchain_core.documents.base import Document
from pydantic import BaseModel


class FullMetadata(BaseModel):
    doc_id: str
    chunk_id: Optional[str] = None
    additional_metadata: Optional[Dict[str, str]] = None

    def get_full_metadata(self) -> Dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": str(self.chunk_id),
            **(self.additional_metadata or {}),
        }


class CustomDocument(Document):
    metadata: Optional[FullMetadata] = None
    vector: Optional[List[float]] = None

    class Config:
        arbitrary_types_allowed = True

    def add_embedding(self, vector: List[float]) -> CustomDocument:
        self.vector = vector
        return self

    def extend_metadata(
        self, chunk_id: int, additional_metadata: Optional[Dict[str, str]] = None
    ) -> FullMetadata:
        self.metadata.chunk_id = chunk_id
        self.metadata.additional_metadata = additional_metadata
        return self.metadata

    def get_full_metadata(self) -> Dict[str, str]:
        if self.metadata is None:
            return None
        else:
            return self.metadata.get_full_metadata()

    def to_payload(self) -> Dict[str, str]:
        if self.metadata is None:
            return {
                "page_content": self.page_content,
            }
        else:
            return {
                "page_content": self.page_content,
                "metadata": self.get_full_metadata(),
            }

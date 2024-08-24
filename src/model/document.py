from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pydantic import BaseModel


from .qdrant_document import QdrantDocument


class Document(ABC, BaseModel):
    text: str
    metadata: Optional[Dict[str, str]] = None
    doc_id: Optional[str] = None
    chunk_id: Optional[int] = None
    vector: Optional[List[float]] = None

    @abstractmethod
    def to_qdrant_document(self, generate_vector: bool = False) -> QdrantDocument:
        pass

    @abstractmethod
    def add_embedding(self, vector: List[float]) -> Document:
        pass

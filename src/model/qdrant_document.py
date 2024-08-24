from pydantic import BaseModel, field_validator
from typing import List, Dict, Optional


class QdrantPayload(BaseModel):
    text: str
    text_id: Optional[str] = None
    chunk_id: Optional[int] = None
    metadata: Optional[Dict[str, str]] = None


class QdrantDocument(BaseModel):
    vector: Optional[List[float]] = None
    payload: QdrantPayload


class QdrantDocumentCollection(BaseModel):
    documents: List[QdrantDocument]

    @field_validator("documents")
    def documents_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("Documents list must not be empty")
        return v

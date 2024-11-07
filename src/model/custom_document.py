from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from langchain_core.documents.base import Document
from pydantic import BaseModel, Field


class FullMetadata(BaseModel):
    doc_id: str
    chunk_id: Optional[str] = Field(default=None)
    additional_metadata: Optional[Dict[str, Any]] = Field(default=None)

    def get_full_metadata(self) -> Dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id or "",  # Ensure chunk_id is a string
            **(self.additional_metadata or {}),
        }


class FullMetadataRetrieval(FullMetadata):
    similarity_score: Optional[float] = Field(default=None)


class SplitContent(BaseModel):
    type: Literal["table", "text"]
    content: str
    position: Optional[int] = Field(default=None)
    visited: Optional[bool] = Field(default=None)


class CustomDocument(Document):
    metadata: Optional[Union[FullMetadata, FullMetadataRetrieval]] = Field(default=None)
    vector: Optional[List[float]] = Field(default=None)
    splitted_content: Optional[List[SplitContent]] = Field(default=None)
    original_content: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def add_embedding(self, vector: List[float]) -> CustomDocument:
        self.vector = vector
        return self

    def extend_metadata(
        self,
        chunk_id: Optional[str],
        additional_metadata: Optional[Dict[str, str]] = None,
    ) -> FullMetadata | FullMetadataRetrieval | None:
        if self.metadata:
            self.metadata.chunk_id = chunk_id
            self.metadata.additional_metadata = additional_metadata
        return self.metadata

    def get_full_metadata(self) -> Optional[Dict[str, str]]:
        return self.metadata.get_full_metadata() if self.metadata else None

    def to_payload(self) -> Dict[str, Any]:
        payload = {"page_content": self.page_content}
        if self.metadata:
            full_metadata = self.get_full_metadata()
            if full_metadata:
                payload["metadata"] = self.get_full_metadata()  # type: ignore
        return payload

    @staticmethod
    def doc_to_custom_doc(doc: Document) -> CustomDocument:
        metadata_dict = doc.metadata or {}
        chunk_id = metadata_dict.get("chunk_id")
        additional_metadata = metadata_dict.get("additional_metadata")
        similarity_score = metadata_dict.get("similarity_score")

        metadata = FullMetadataRetrieval(
            doc_id=metadata_dict["doc_id"],
            chunk_id=str(chunk_id),
            additional_metadata=additional_metadata,
            similarity_score=similarity_score,
        )

        return CustomDocument(
            id=doc.id,
            page_content=doc.page_content,
            metadata=metadata,
        )

    @staticmethod
    def docs_to_custom_docs(docs: List[Document]) -> List[CustomDocument]:
        return [CustomDocument.doc_to_custom_doc(doc) for doc in docs]


class CustomDocumentWithMetadata(CustomDocument):
    metadata: Union[FullMetadata, FullMetadataRetrieval] = Field(default=None)

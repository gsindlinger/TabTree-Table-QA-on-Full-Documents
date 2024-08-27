from __future__ import annotations

from .custom_document import CustomDocument
from .qdrant_document import QdrantDocument, QdrantPayload


class SECFiling(CustomDocument):
    def to_qdrant_document(self) -> QdrantDocument:
        return QdrantDocument(
            vector=self.vector,
            payload=QdrantPayload(
                text=self.text,
                text_id=self.doc_id,
                chunk_id=self.chunk_id,
                metadata=self.metadata,
            ),
        )

    def add_embedding(self, vector: list[float]) -> SECFiling:
        self.vector = vector
        return self

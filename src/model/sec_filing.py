from __future__ import annotations
import logging

from .document import Document
from ..indexing.embeddings.huggingface_embeddings import HuggingFaceEmbeddings
from .qdrant_document import QdrantDocument, QdrantPayload


class SECFiling(Document):
    def to_qdrant_document(self, generate_vector: bool = False) -> QdrantDocument:
        if generate_vector:
            huggingface_embeddings = HuggingFaceEmbeddings()
            self.vector = huggingface_embeddings.embed_documents([self.text])[0]
            logging.info(f"Generated vector for {self.doc_id}")

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

from __future__ import annotations
from langchain_qdrant import QdrantVectorStore as _QdrantVectorStore

from .qdrant_client import QdrantClient
from langchain_core.embeddings import Embeddings


class QdrantVectorStore(_QdrantVectorStore):
    client: QdrantClient

    @classmethod
    def from_config(
        cls,
        embedding_model: Embeddings,
        collection_name: str,
    ) -> QdrantVectorStore:
        """Create a QdrantVectorStore instance from the config"""

        client = QdrantClient()
        vector_size = len(embedding_model.embed_documents(["Hello World!"])[0])
        if not client.is_populated(collection_name=collection_name):
            client.create_index(
                collection_name=collection_name,
                vector_size=vector_size,
            )

        return cls(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )

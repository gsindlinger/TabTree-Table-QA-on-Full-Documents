from langchain_qdrant import QdrantVectorStore as _QdrantVectorStore

from .embeddings.huggingface_embeddings import HuggingFaceEmbeddings
from ..config import Config
from .qdrant_client import QdrantClient


class QdrantVectorStore(_QdrantVectorStore):
    client: QdrantClient

    @classmethod
    def from_config(cls):
        embedding_tool = Config.pipeline.embedding_tool
        if embedding_tool == "HuggingFaceEmbeddings":
            embedding_model = HuggingFaceEmbeddings(show_progress=True)
            vector_size = len(embedding_model.embed_documents(["Hello World!"])[0])
        else:
            raise ValueError(f"Unknown embedding tool: {embedding_tool}")

        client = QdrantClient()
        collection_name = f"{Config.qdrant.index_name}-{Config.pipeline.embedding_tool}"
        if not client.is_populated(collection=collection_name):
            client.create_index(
                collection_name=collection_name,
                vector_size=vector_size,
            )

        return cls(
            client=QdrantClient(),
            collection_name=collection_name,
            embedding=embedding_model,
        )

from langchain_qdrant import QdrantVectorStore as _QdrantVectorStore

from .embeddings.openai_embeddings import OpenAIEmbeddings
from .embeddings.nomic_embeddings import NomicEmbeddings
from .embeddings.huggingface_embeddings import HuggingFaceEmbeddings
from ..config import Config
from .qdrant_client import QdrantClient


class QdrantVectorStore(_QdrantVectorStore):
    client: QdrantClient

    @classmethod
    def from_config(cls):
        match Config.indexing.embedding_method:
            case "huggingface":
                embedding_model = HuggingFaceEmbeddings()
            case "nomic":
                embedding_model = NomicEmbeddings()
            case "openai":
                embedding_model = OpenAIEmbeddings()
            case _:
                raise ValueError(
                    f"Unknown embedding tool: {Config.indexing.embedding_method}"
                )

        vector_size = len(embedding_model.embed_documents(["Hello World!"])[0])
        client = QdrantClient()
        embedding_model_name = embedding_model.get_model_name()
        embedding_model_name = embedding_model_name[
            embedding_model_name.rfind("/") + 1 :
        ]
        match Config.run.dataset:
            case "sec-filings":
                collection_name = f"{Config.run.dataset}-{embedding_model_name}-{Config.indexing.chunking_strategy}-{Config.sec_filings.preprocess_mode_index_name}"
            case _:
                collection_name = f"{Config.run.dataset}-{embedding_model_name}-{Config.indexing.chunking_strategy}"

        if not client.is_populated(collection_name=collection_name):
            client.create_index(
                collection_name=collection_name,
                vector_size=vector_size,
            )

        return cls(
            client=QdrantClient(),
            collection_name=collection_name,
            embedding=embedding_model,
        )

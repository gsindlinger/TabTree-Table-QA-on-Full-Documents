import logging
from pydantic import BaseModel

from .embeddings.sentence_transformers_embeddings import SentenceTransformersEmbeddings
from .embeddings.huggingface_embeddings import HuggingFaceEmbeddings
from .document_splitter import SECFilingSplitter
from .document_loader import SECFilingLoader
from ..config import Config
from .qdrant import QdrantClient


class IndexingService(BaseModel):
    qdrant: QdrantClient
    index_name: str

    class Config:
        arbitrary_types_allowed = True

    def create_index(self):
        self.qdrant.create_index(collection_name=self.index_name)

    def embed_documents(self):
        # Load documents
        document_path = Config.data.path_local
        document_loader = SECFilingLoader(document_path)
        documents = document_loader.load_documents()

        # Split documents
        document_splitter = SECFilingSplitter()
        documents = document_splitter.split_documents(documents)

        # Embed documents
        # embeddings = HuggingFaceEmbeddings()
        embeddings = SentenceTransformersEmbeddings.from_config()
        vectors = embeddings.embed_documents(texts=[doc.text for doc in documents])
        logging.info(f"Generated {len(vectors)} using HuggingFace")

        # Add embeddings to documents
        documents = [
            doc.add_embedding(embedding) for doc, embedding in zip(documents, vectors)
        ]

        # Transform to Qdrant documents
        documents = [doc.to_qdrant_document() for doc in documents]

        # Store documents
        self.qdrant.add_documents(documents=documents, collection_name=self.index_name)

    @classmethod
    def from_config(cls):
        return cls(
            qdrant=QdrantClient(),
            index_name=Config.qdrant.index_name,
        )

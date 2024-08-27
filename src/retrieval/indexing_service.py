import logging
from pydantic import BaseModel


from .document_splitter import SECFilingSplitter
from .document_loader import SECFilingLoader
from ..config import Config
from .qdrant_store import QdrantVectorStore


class IndexingService(BaseModel):
    qdrant_store: QdrantVectorStore

    class Config:
        arbitrary_types_allowed = True

    def create_index(self):
        self.qdrant_store.client.create_index(
            collection_name=self.qdrant_store.collection_name
        )

    def embed_documents(self):
        # Load documents
        document_path = Config.data.path_local
        document_loader = SECFilingLoader(document_path)
        documents = document_loader.load_documents()

        # Split documents
        document_splitter = SECFilingSplitter()
        documents = document_splitter.split_documents(documents)

        # Embed documents
        vectors = self.qdrant_store.embeddings.embed_documents(
            texts=[doc.page_content for doc in documents]
        )
        logging.info(f"Generated {len(vectors)}")

        # Store documents using Qdrant client
        self.qdrant_store.client.add_documents(
            documents=documents,
            collection_name=self.qdrant_store.collection_name,
            vectors=vectors,
        )

        # Store documents
        # uploaded_documents = self.qdrant_store.add_documents(documents=documents)

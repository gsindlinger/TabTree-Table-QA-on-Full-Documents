from pydantic import BaseModel
from .qdrant_document import QdrantDocumentCollection
from .qdrant import QdrantClient


class IndexingService(BaseModel):
    qdrant: QdrantClient
    index_name: str
    documents: QdrantDocumentCollection

    def create_index(self):
        self.qdrant.create_index(collection_name=self.index_name)

    def add_documents(self, texts, metadata, ids):
        self.documents.add_multiple(texts)
        self.qdrant.add_documents(
            collection_name=self.index_name,
            documents=self.documents,
            metadata=metadata,
            ids=ids,
        )

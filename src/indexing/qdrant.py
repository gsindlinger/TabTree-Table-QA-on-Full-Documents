from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient as _QdrantClient

from ..model.qdrant_document import QdrantDocument, QdrantDocumentCollection
from ..config import Config
from itertools import count
from time import sleep
from typing import Optional, Self
import logging
import uuid


class QdrantClient(_QdrantClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **(self._default_kwargs() | kwargs))

    @staticmethod
    def _default_kwargs() -> dict:
        return {
            "host": Config.qdrant.host,
            "port": Config.qdrant.port,
            "grpc_port": Config.qdrant.grpc_port,
        }

    def wait(self, timeout: float | None = None, ping_interval: float = 1) -> Self:
        """wait until qdrant becomes available via HTTP"""
        for t in count():
            try:
                self.get_collections()
                return self
            except Exception as e:
                if timeout is not None and t >= timeout:
                    break
                logging.info(f"waiting for {self.transport.hosts} to become available")
                sleep(ping_interval)
        raise TimeoutError

    def is_populated(self, collection: str, accept_empty: bool = False) -> bool:
        """check if collection exists and is not empty"""
        if not self.collection_exists(collection_name=collection):
            logging.info(f"index {collection} does not exist")
            return False
        if not accept_empty:
            count = self.count(collection_name=collection)
            if count:
                logging.info(f"index {collection} contains {count} documents")
                return True
            else:
                logging.info(f"index {collection} exists but is empty")
                return False
        else:
            logging.info(f"index {collection} exists")
            return True

    def create_index(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: Distance = Distance.COSINE,
    ) -> None:
        if not self.collection_exists(collection_name):
            self.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_metric,
                ),
            )
            logging.info(f"index {collection_name} created")
        else:
            logging.info(f"index {collection_name} already exists")

    def add_documents(
        self,
        collection_name: str,
        documents: QdrantDocumentCollection,
        vector_size: Optional[int] = None,
    ) -> None:
        if not self.is_populated(collection=collection_name, accept_empty=True):
            if not vector_size:
                vector_size = len(documents[0].vector)

            self.create_index(
                collection_name=collection_name,
                vector_size=vector_size,
            )
        points = [
            PointStruct(
                id=str(uuid.uuid4()), vector=doc.vector, payload=doc.payload.__dict__
            )
            for doc in documents
        ]
        self.upload_points(collection_name=collection_name, points=points)
        logging.info(f"added {len(documents)} documents to {collection_name}")

    def add_document(self, collection_name: str, document: QdrantDocument) -> None:
        if not self.is_populated(collection=collection_name, accept_empty=True):
            self.create_index(
                collection_name=collection_name,
                vector_size=document.vector_size,
            )
        points = [
            PointStruct(
                id=document.id, vector=document.vector, payload=document.payload
            )
        ]
        self.add_documents(collection_name=collection_name, points=points)

    def query(self, collection_name: str, query_text: str, top_k: int = 4) -> dict:
        return super().query(
            collection_name=collection_name,
            query=query_text,
            top=top_k,
        )

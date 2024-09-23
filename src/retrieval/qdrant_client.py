from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient as _QdrantClient

from ..model.custom_document import CustomDocument
from ..config import Config
from itertools import count
from time import sleep
from typing import List, Self
import logging
import uuid


class QdrantClient(_QdrantClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **(self._default_kwargs() | kwargs))

    @staticmethod
    def _default_kwargs() -> dict:
        return {
            "host": Config.env_variables.QDRANT_HOST,
            "port": Config.env_variables.QDRANT_API_PORT,
            "grpc_port": Config.env_variables.QDRANT_GRPC_PORT,
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

    def is_populated(self, collection_name: str, accept_empty: bool = False) -> bool:
        """check if collection exists and is not empty"""
        if not self.collection_exists(collection_name=collection_name):
            logging.info(f"index {collection_name} does not exist")
            return False
        if not accept_empty:
            count = self.count(collection_name=collection_name)
            if count:
                logging.info(f"index {collection_name} contains {count} documents")
                return True
            else:
                logging.info(f"index {collection_name} exists but is empty")
                return False
        else:
            logging.info(f"index {collection_name} exists")
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
        documents: List[CustomDocument],
        vectors: List[List[float]],
    ) -> None:

        assert len(documents) == len(vectors)

        if not self.is_populated(collection_name=collection_name, accept_empty=True):
            self.create_index(
                collection_name=collection_name,
                vector_size=len(vectors[0]),
            )
        else:
            logging.info(
                f"Index {collection_name} already exists, adding documents to existing index"
            )
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=doc.to_payload(),
            )
            for doc, vec in zip(documents, vectors)
        ]
        self.upload_points(collection_name=collection_name, points=points)
        logging.info(f"added {len(documents)} documents to {collection_name}")

import json
import os
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient as _QdrantClient
from langchain_core.documents.base import Document

from ..model.custom_document import CustomDocument, FullMetadata
from ..config import Config
from itertools import count
from time import sleep
from typing import List, Self
import logging
import uuid


class QdrantClient(_QdrantClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **(self._default_kwargs() | kwargs))
        
        payload_folder_path = Config.qdrant.payload_folder_path
        if payload_folder_path[-1] != "/":
            payload_folder_path += "/"
        
        self.payload_folder_path = Config.qdrant.payload_folder_path


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
            if count and count.count > 0:
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
            
        ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        points = [PointStruct(id=id, vector=vec, payload = CustomDocument(id=id, page_content="", metadata=FullMetadata(doc_id="", additional_metadata={"manual_id": id})).to_payload()) for (vec, id) in zip(vectors, ids)]
        self.upload_points(collection_name=collection_name, points=points)
        
        
        points_with_payloads = [
            PointStruct(
                id=id,
                vector=vec,
                payload=doc.to_payload(),
            ) for doc, vec, id in zip(documents, vectors, ids)]
        self.store_payloads(documents=points_with_payloads, collection_name=collection_name)
        logging.info(f"added {len(documents)} documents to {collection_name}")

        
    def store_payloads(self, documents: List[PointStruct], collection_name: str) -> None:
        
        # check if directory with file path + collection name exists and create it if not
        if not os.path.exists(self.payload_folder_path):
            os.makedirs(self.payload_folder_path)
            
        payloads = {}
        for doc in documents:
            payloads[doc.id] = doc.payload
            
        # store payloads in json file
        with open(f"{self.payload_folder_path}{collection_name}.json", "w") as file:
            json.dump(payloads, file, indent=4)
            
            
    def load_payloads(self, collection_name: str) -> dict:
        try:
            with open(f"{self.payload_folder_path}{collection_name}.json", "r") as file:
                payloads = json.load(file)
            return payloads
        except FileNotFoundError:
            raise FileNotFoundError(f"Payload file for collection {collection_name} not found")
    
    @staticmethod
    def extend_docs_with_payload(docs: List[Document]) -> List[Document]:
        if len(docs) == 0:
            return docs
        
        collection_name = docs[0].metadata['_collection_name']
        client = QdrantClient()
        payloads = client.load_payloads(collection_name=collection_name)
        
        final_payloads = []
        for doc in docs:
            doc_id = doc.metadata['manual_id']
            payload = payloads[doc_id]
            doc.page_content = payload['page_content']
            doc.metadata.update(payload['metadata'])
            final_payloads.append(doc)
        
        
        return final_payloads    
    
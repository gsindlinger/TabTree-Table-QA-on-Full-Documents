from typing import List
from sentence_transformers import SentenceTransformer

from .abstract_embeddings import Embeddings
from ...config import Config


class SentenceTransformersEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.connector.encode(texts, show_progress_bar=True).tolist()

    @classmethod
    def from_config(cls):
        connector = SentenceTransformer(Config.sentence_transformers.embedding_model)
        return cls(connector=connector)

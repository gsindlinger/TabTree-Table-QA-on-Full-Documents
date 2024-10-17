import logging
from fastembed import TextEmbedding
from langchain_community.embeddings.fastembed import (
    FastEmbedEmbeddings as _FastEmbedEmbeddings,
)
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from .custom_embeddings import CustomEmbeddings
from ...config import Config


class FastEmbedEmbeddings(BaseModel, Embeddings, CustomEmbeddings):
    embedding_model_name: str
    model: TextEmbedding

    class Config:
        arbitrary_types_allowed = True

    def __init__(self) -> None:
        model_name = Config.fast_embed.embedding_model
        super().__init__(
            model=TextEmbedding(model_name), embedding_model_name=model_name
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.passage_embed(texts)
        logging.info(f"Embeddings: {embeddings}")
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.query_embed(text)
        return list(embedding)

    @staticmethod
    def _default_kwargs() -> dict:
        return {
            "model_name": Config.fast_embed.embedding_model,
        }

    def get_model_name(self) -> str:
        return self.model_name

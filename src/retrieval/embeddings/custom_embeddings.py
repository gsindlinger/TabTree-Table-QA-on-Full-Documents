from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, Union, runtime_checkable


from ...config.config import Config
from langchain_core.embeddings import Embeddings


@runtime_checkable
class CustomEmbeddings(Protocol):
    @classmethod
    def from_config(cls) -> CustomEmbeddings:
        from .fast_embed_embeddings import FastEmbedEmbeddings
        from .huggingface_embeddings import HuggingFaceEmbeddings
        from .nomic_embeddings import NomicEmbeddings
        from .openai_embeddings import OpenAIEmbeddings

        match Config.indexing.embedding_method:
            case "huggingface":
                embedding_model = HuggingFaceEmbeddings()
            case "nomic":
                embedding_model = NomicEmbeddings()
            case "openai":
                embedding_model = OpenAIEmbeddings()
            case "fast_embed":
                embedding_model = FastEmbedEmbeddings()
            case _:
                embedding_model = None

        if isinstance(embedding_model, CustomEmbeddings):
            return embedding_model
        else:
            raise ValueError(
                f"Unknown embedding method: {Config.indexing.embedding_method}"
            )

    def get_model_name_stripped(self):
        embedding_model_name = self.get_model_name()
        embedding_model_name = embedding_model_name[
            embedding_model_name.rfind("/") + 1 :
        ]
        return embedding_model_name

    @abstractmethod
    def get_model_name(self) -> str:
        pass

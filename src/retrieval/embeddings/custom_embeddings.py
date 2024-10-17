from __future__ import annotations
from abc import ABC, abstractmethod


from ...config.config import Config


class CustomEmbeddings(ABC):
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
                raise ValueError(
                    f"Unknown embedding tool: {Config.indexing.embedding_method}"
                )
        return embedding_model

    def get_model_name_stripped(self):
        embedding_model_name = self.get_model_name()
        embedding_model_name = embedding_model_name[
            embedding_model_name.rfind("/") + 1 :
        ]
        return embedding_model_name

    @abstractmethod
    def get_model_name(self) -> str:
        pass

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
                embedding_model = None

        if embedding_model is not None:
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
        """
        Generate a unique model name for this embedding model.
        """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """

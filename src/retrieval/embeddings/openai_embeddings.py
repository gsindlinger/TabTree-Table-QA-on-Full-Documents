from langchain_openai import OpenAIEmbeddings as _OpenAIEmbeddings

from .custom_embeddings import CustomEmbeddings
from ...config import Config


class OpenAIEmbeddings(_OpenAIEmbeddings, CustomEmbeddings):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **(self._default_kwargs() | kwargs))

    @staticmethod
    def _default_kwargs() -> dict:
        return {
            "model": Config.openai.embedding_model,
        }

    def get_model_name(self) -> str:
        return self.model

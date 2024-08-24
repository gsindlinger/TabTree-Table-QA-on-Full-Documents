from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from .abstract_embeddings import Embeddings
from ...config import Config


class HuggingFaceEmbeddings(Embeddings):
    @classmethod
    def from_config(cls):
        connector = HuggingFaceInferenceAPIEmbeddings(
            api_key=Config.huggingface.api_key,
            model_name=Config.huggingface.embedding_model,
        )
        return cls(connector=connector)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self.connector.embed_documents(documents)

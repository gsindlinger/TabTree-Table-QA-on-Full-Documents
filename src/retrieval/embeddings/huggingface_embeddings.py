from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
from ...config import Config


class HuggingFaceEmbeddings(_HuggingFaceEmbeddings):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **(self._default_kwargs() | kwargs))

    @staticmethod
    def _default_kwargs() -> dict:
        model_kwargs = {"device": "cpu", "trust_remote_code": True}
        encode_kwargs = {"normalize_embeddings": True}
        return {
            "model_name": Config.huggingface.embedding_model,
            "model_kwargs": model_kwargs,
            "encode_kwargs": encode_kwargs,
            "show_progress": True,
        }

    def get_model_name(self) -> str:
        return self.model_name

from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
from ...config import Config


class HuggingFaceEmbeddings(_HuggingFaceEmbeddings):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **(self._default_kwargs() | kwargs))

    @staticmethod
    def _default_kwargs() -> dict:
        return {"model_name": Config.huggingface.embedding_model}

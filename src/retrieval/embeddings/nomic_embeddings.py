from langchain_nomic import NomicEmbeddings as _NomicEmbeddings

from ...config.config import Config


class NomicEmbeddings(_NomicEmbeddings):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **(self._default_kwargs() | kwargs))

    @staticmethod
    def _default_kwargs() -> dict:
        return {
            "model": Config.nomic.embedding_model,
            "nomic_api_key": Config.env_variables.NOMIC_API_KEY,
        }

    def get_model_name(self) -> str:
        return self.model

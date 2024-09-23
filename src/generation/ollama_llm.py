from langchain_ollama import ChatOllama
from ..config import Config


class OllamaLLM(ChatOllama):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **(self._default_kwargs() | kwargs))

    @staticmethod
    def _default_kwargs() -> dict:
        return {
            "base_url": f"http://{Config.env_variables.OLLAMA_HOST}:{Config.env_variables.OLLAMA_PORT}",
            "model": Config.ollama.model,
            "temperature": 0,
        }

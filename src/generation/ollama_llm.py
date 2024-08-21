from langchain_ollama import ChatOllama
from ..config import Config


class OllamaLLM(ChatOllama):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **(self._default_kwargs() | kwargs))

    @staticmethod
    def _default_kwargs() -> dict:
        return {
            "base_url": f"http://{Config.ollama.host}:{Config.ollama.port}",
            "model": Config.ollama.model,
            "temperature": Config.ollama.temperature,
        }

from ..config import Config
from langchain_openai import ChatOpenAI


class OpenAILLM(ChatOpenAI):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **(self._default_kwargs() | kwargs))

    @staticmethod
    def _default_kwargs() -> dict:
        return {
            "model": Config.openai.generation_model,
            "max_tokens": Config.openai.max_tokens,
        }

from langchain_huggingface import HuggingFaceEndpoint
from ..config import Config
from langchain_community.chat_models.openai import ChatOpenAI


class HuggingFaceLLM(HuggingFaceEndpoint):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **(self._default_kwargs() | kwargs))

    @staticmethod
    def _default_kwargs() -> dict:
        return {
            "repo_id": Config.huggingface.generation_model,
            "huggingfacehub_api_token": Config.env_variables.HUGGINGFACE_API_KEY,
        }

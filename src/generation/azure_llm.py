import logging
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from ..config.config import Config


class AzureLLM(AzureAIChatCompletionsModel):
    def __init__(self, *args, **kwargs) -> None:
        logging.getLogger("azure").setLevel(logging.WARNING)
        super().__init__(*args, **(self._default_kwargs() | kwargs))

    @staticmethod
    def _default_kwargs() -> dict:
        return {
            "endpoint": Config.env_variables.AZURE_INFERENCE_ENDPOINT,
            "credential": Config.env_variables.AZURE_INFERENCE_CREDENTIAL,
            "model_name": Config.azure.model_name_generation,
            "max_tokens": Config.azure.max_tokens_generation,
            "temperature": Config.azure.temperature_generation,
        }

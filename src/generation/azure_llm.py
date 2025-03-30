import logging
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from ..config.config import Config


class AzureLLM(AzureAIChatCompletionsModel):
    def __init__(self, *args, **kwargs) -> None:
        logging.getLogger("azure").setLevel(logging.WARNING)
        super().__init__(*args, **(self._default_kwargs() | kwargs))

    @staticmethod
    def _default_kwargs() -> dict:
        model = Config.azure.model_name_generation
        
        match model:
            case "Phi-4":
                endpoint = Config.env_variables.AZURE_INFERENCE_ENDPOINT_PHI_4
                credential = Config.env_variables.AZURE_INFERENCE_CREDENTIAL_PHI_4
            case "Llama-3.3-70B-Instruct":
                endpoint = Config.env_variables.AZURE_INFERENCE_ENDPOINT_LLAMA
                credential = Config.env_variables.AZURE_INFERENCE_CREDENTIAL_LLAMA
            case "Phi-4-multimodal-instruct":
                endpoint = Config.env_variables.AZURE_INFERENCE_ENDPOINT_LLAMA
                credential = Config.env_variables.AZURE_INFERENCE_CREDENTIAL_LLAMA
            case _:
                raise ValueError(f"Model {model} not supported")
        
        return {
                "endpoint": endpoint,
                "credential": credential,
                "model_name": Config.azure.model_name_generation,
                "max_tokens": Config.azure.max_tokens_generation,
                "temperature": Config.azure.temperature_generation,
            }
        
        
        return {
            "endpoint": Config.env_variables.AZURE_INFERENCE_ENDPOINT,
            "credential": Config.env_variables.AZURE_INFERENCE_CREDENTIAL,
            "model_name": Config.azure.model_name_generation,
            "max_tokens": Config.azure.max_tokens_generation,
            "temperature": Config.azure.temperature_generation,
        }

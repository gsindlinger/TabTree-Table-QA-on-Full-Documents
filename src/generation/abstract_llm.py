from abc import ABC, abstractmethod
from typing import List

from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel

from ..config import Config


class LLM(BaseModel, ABC):
    @abstractmethod
    def generate_text(
        self, prompt: str, max_tokens: int = 100, stop: Optional[List[str]] = None
    ) -> str:
        pass

    @classmethod
    def from_config(cls):
        from .openai_llm import OpenAILLM
        from .huggingface_llm import HuggingFaceLLM
        from .ollama_llm import OllamaLLM

        if Config.text_generation.method == "ollama":
            return OllamaLLM()
        elif Config.text_generation.method == "huggingface":
            return HuggingFaceLLM()
        elif Config.text_generation.method == "openai":
            return OpenAILLM()
        else:
            raise ValueError(f"Unknown LLM method: {Config.text_generation.method}")

    @classmethod
    def from_tabtree_config(cls):
        from .openai_llm import OpenAILLM
        from .huggingface_llm import HuggingFaceLLM
        from .ollama_llm import OllamaLLM

        model_name = Config.tabtree.llm_model
        max_tokens = Config.tabtree.llm_max_tokens

        if Config.tabtree.llm_method == "ollama":
            return OllamaLLM(model=model_name, num_predict=max_tokens)
        elif Config.tabtree.llm_method == "huggingface":
            return HuggingFaceLLM(repo_id=model_name, max_new_tokens=max_tokens)
        elif Config.tabtree.llm_method == "openai":
            return OpenAILLM(model=model_name, max_tokens=max_tokens)
        else:
            raise ValueError(f"Unknown LLM method: {Config.text_generation.method}")

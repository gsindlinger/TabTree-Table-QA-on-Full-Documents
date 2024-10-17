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

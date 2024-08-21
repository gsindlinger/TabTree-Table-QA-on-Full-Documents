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
        """
        Generate text based on the given prompt.

        Args:
            prompt (str): The input text prompt to generate from.
            max_tokens (int): The maximum number of tokens to generate.
            stop (Optional[List[str]]): A list of stop words or sequences where generation should stop.

        Returns:
            str: The generated text.
        """
        pass

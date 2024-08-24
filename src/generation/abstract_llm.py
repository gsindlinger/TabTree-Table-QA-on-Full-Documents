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

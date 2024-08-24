from abc import ABC, abstractmethod
from typing import Any, List, Optional


from pydantic import BaseModel


class Embeddings(ABC, BaseModel):
    connector: Optional[Any] = None

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

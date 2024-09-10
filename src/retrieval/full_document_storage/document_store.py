from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel
from ...model.custom_document import CustomDocument


class FullDocumentStore(ABC, BaseModel):

    @abstractmethod
    def store_full_documents(self, documents: List[CustomDocument]) -> None:
        pass

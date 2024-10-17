from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel

from ...config.config import Config
from ...model.custom_document import CustomDocument


class DocumentLoader(ABC, BaseModel):

    @classmethod
    def from_config(cls) -> DocumentLoader:
        mode = Config.run.dataset
        match mode:
            case "wiki-table-questions":
                from .wiki_table_questions_loader import WikiTableQuestionsLoader

                return WikiTableQuestionsLoader()
            case "sec-filings":
                from .sec_filing_loader import SECFilingLoader

                return SECFilingLoader()
            case _:
                raise ValueError(f"Unknown mode: {mode}")

    @abstractmethod
    def load_documents(
        self, num_of_documents: Optional[int] = None
    ) -> List[CustomDocument]:
        pass

    @abstractmethod
    def load_single_document(
        self,
    ) -> CustomDocument:
        pass

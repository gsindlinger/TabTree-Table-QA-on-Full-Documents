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
        if "sec-filings" in mode:
            from .sec_filing_loader import SECFilingLoader

            return SECFilingLoader()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @abstractmethod
    def load_documents(
        self, preprocess_mode: str = "default", num_of_documents: Optional[int] = None
    ) -> List[CustomDocument]:
        pass

    @abstractmethod
    def load_single_document(self, preprocess_mode: str = "default") -> CustomDocument:
        pass

    def preprocess_document(
        self, document: CustomDocument, preprocess_mode: str = "default"
    ) -> CustomDocument:
        """Loads documents and calls preprocessing method.

        Args:
            preprocess_mode (str, optional): _description_. Defaults to "default".
                                                            Options are "none", "remove_attributes", "remove_xbrl", "all", "default". "All" and "default" are equivalent.
            num_of_documents (Optional[int], optional): _description_. Defaults to None.

        Returns:
            List[CustomDocument]: _description_
        """
        return document

    def store_full_document(self, document: CustomDocument) -> None:
        pass

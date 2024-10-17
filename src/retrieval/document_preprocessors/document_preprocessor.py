from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel

from .table_serializer import TableSerializer
from ..document_preprocessors.preprocess_config import PreprocessConfig
from ...config.config import Config
from ...model.custom_document import CustomDocument


class DocumentPreprocessor(ABC, BaseModel):
    preprocess_config: Optional[PreprocessConfig] = None
    table_serializer: Optional[TableSerializer] = None

    @classmethod
    def from_config(cls, preprocess_config: PreprocessConfig) -> DocumentPreprocessor:
        mode = Config.run.dataset
        table_serializer = TableSerializer.from_preprocess_config(preprocess_config)

        if mode in ["wiki-table-questions", "sec-filings"]:

            from .html_preprocessor import HTMLPreprocessor

            return HTMLPreprocessor(
                preprocess_config=preprocess_config, table_serializer=table_serializer
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @abstractmethod
    def preprocess_document(self, document: CustomDocument) -> CustomDocument:
        pass

    def preprocess_multiple_documents(
        self, documents: List[CustomDocument]
    ) -> List[CustomDocument]:
        return [self.preprocess_document(document) for document in documents]

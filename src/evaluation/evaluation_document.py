from __future__ import annotations
from typing import List
import pandas as pd
from pydantic import BaseModel

from ..config.config import Config


class EvaluationDocument(BaseModel):
    doc_id: str
    question: str
    answer: str
    search_reference: str

    @staticmethod
    def filter_documents_by_id(documents: List[EvaluationDocument], id: str):
        return [doc for doc in documents if doc.doc_id == id]

    @staticmethod
    def to_csv(documents: List[EvaluationDocument], file_path: str):
        df = pd.DataFrame([item.model_dump() for item in documents])
        df.to_csv(path_or_buf=file_path, sep=";", index=False)

    @staticmethod
    def from_csv(file_path: str) -> List[EvaluationDocument]:
        df = pd.read_csv(filepath_or_buffer=file_path, sep=";")
        df["answer"] = df["answer"].astype(str)
        df["search_reference"] = df["search_reference"].astype(str)
        return [EvaluationDocument(**row) for _, row in df.iterrows()]

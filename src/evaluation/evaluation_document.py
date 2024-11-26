from __future__ import annotations
import ast
import os
from typing import List, Optional
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
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        df = pd.read_csv(filepath_or_buffer=file_path, sep=";")
        df["answer"] = df["answer"].astype(str)
        df["search_reference"] = df["search_reference"].astype(str)
        return [EvaluationDocument(**row) for _, row in df.iterrows()]


class HeaderEvaluationDocument(BaseModel):
    position: int
    columns: list[int]
    rows: list[int]
    search_regexp: str
    best_orientation: Optional[str] = None
    note: Optional[str] = None
    html_data: Optional[str] = None

    @staticmethod
    def to_csv(documents: List[HeaderEvaluationDocument], file_path: str):
        df = pd.DataFrame([item.model_dump() for item in documents])
        df.to_csv(path_or_buf=file_path, sep=";", index=False)

    @staticmethod
    def from_csv(file_path: str) -> List[HeaderEvaluationDocument]:
        df = pd.read_csv(filepath_or_buffer=file_path, sep=";")
        df["position"] = df["position"].astype(int)
        df["search regexp"] = df["search regexp"].astype(str)
        df["columns"] = df["columns"].apply(lambda x: list(ast.literal_eval(x)))
        df["rows"] = df["rows"].apply(lambda x: list(ast.literal_eval(x)))
        df["best orientation"] = df["best orientation"].astype(str)
        df["note"] = df["note"].astype(str)
        return [
            HeaderEvaluationDocument(
                position=row["position"],
                search_regexp=row["search regexp"],
                columns=row["columns"],
                rows=row["rows"],
            )
            for _, row in df.iterrows()
        ]

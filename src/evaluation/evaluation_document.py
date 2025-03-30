from __future__ import annotations
import ast
import os
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel

from ..config.config import Config


class EvaluationDocument(BaseModel):
    doc_id: str
    question_id: Optional[str] = None
    question: str
    answer: str
    search_reference: str
    category: Optional[str] = None

    @staticmethod
    def filter_documents_by_id(documents: List[EvaluationDocument], id: str):
        return [doc for doc in documents if doc.doc_id == id]

    @staticmethod
    def to_csv(documents: List[EvaluationDocument], file_path: str):
        df = pd.DataFrame([item.model_dump() for item in documents])
        # make dir if not exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(path_or_buf=file_path, sep=";", index=False)

    @staticmethod
    def from_csv(file_path: str) -> List[EvaluationDocument]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        df = pd.read_csv(filepath_or_buffer=file_path, sep=";")
        df.fillna("", inplace=True)
        
        df["answer"] = df["answer"].astype(str)
        df["search_reference"] = df["search_reference"].astype(str)
        return [EvaluationDocument(**row) for _, row in df.iterrows()]


class EvaluationDocumentWithTable(EvaluationDocument):
    html_table: str
    max_column_header_row: int | None = None
    max_row_label_column: int | None = None
    table_title: str | None = None
    
class HeaderEvaluationDocumentReduced(BaseModel):
    html_table: str
    row_label_columns: list[int]
    column_header_rows: list[int]


class HeaderEvaluationDocument(BaseModel):
    position: Optional[int] = None
    row_label_columns: list[int]
    column_header_rows: list[int]
    search_regexp: str | None = None
    best_orientation: Optional[str] = None 
    note: Optional[str] = None
    html_data: Optional[str] = None
    doc_id: str | None = None

    @staticmethod
    def to_csv(documents: List[HeaderEvaluationDocument], file_path: str):
        df = pd.DataFrame([item.model_dump() for item in documents])
        df.to_csv(path_or_buf=file_path, sep=";", index=False)
        
    @staticmethod
    def from_wiki_table_questions_csv(file_path: str) -> List[HeaderEvaluationDocument]:
        df = pd.read_csv(filepath_or_buffer=file_path, sep=";")
        df["column_header_rows"] = df["column_header_rows"].astype(str)
        df["row_label_columns"] = df["row_label_columns"].astype(str)
        df["column_header_rows"] = df["column_header_rows"].apply(
           lambda x: [int(i) for i in x.split(',') if i.strip().isdigit()]
        )
        df["row_label_columns"] = df["row_label_columns"].apply(
            lambda x: [int(i) for i in x.split(',') if i.strip().isdigit()]        
        )
        df["doc-id"] = df["doc-id"].astype(str)
        
        return [HeaderEvaluationDocument(
            doc_id = row["doc-id"],
            row_label_columns=row["row_label_columns"],
            column_header_rows=row["column_header_rows"],
        ) for _, row in df.iterrows()]

    @staticmethod
    def from_sec_filing_csv(file_path: str) -> List[HeaderEvaluationDocument]:
        df = pd.read_csv(filepath_or_buffer=file_path, sep=";")
        df["position"] = df["position"].astype(int)
        df["search regexp"] = df["search regexp"].astype(str)
        df["column_header_rows"] = df["column_header_rows"].astype(str)
        df["row_label_columns"] = df["row_label_columns"].astype(str)
        df["column_header_rows"] = df["column_header_rows"].apply(
           lambda x: [int(i) for i in x.split(',') if i.strip().isdigit()]
        )
        df["row_label_columns"] = df["row_label_columns"].apply(
            lambda x: [int(i) for i in x.split(',') if i.strip().isdigit()]        
        )
        df["best orientation"] = df["best orientation"].astype(str)
        df["note"] = df["note"].astype(str)
        return [
            HeaderEvaluationDocument(
                position=row["position"],
                search_regexp=row["search regexp"],
                row_label_columns=row["row_label_columns"],
                column_header_rows=row["column_header_rows"],
            )
            for _, row in df.iterrows()
        ]
        
    

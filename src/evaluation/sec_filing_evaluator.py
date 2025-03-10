import os
import re
from typing import List, Optional, override
from pandas import DataFrame

from ..retrieval.document_loaders.sec_filing_loader import SECFilingLoader
from .evaluation_document import (
    EvaluationDocument,
    EvaluationDocumentWithTable,
    HeaderEvaluationDocument,
)
from ..config.config import Config
from .evaluator import Evaluator
import pandas as pd


class SECFilingEvaluator(Evaluator):
    def get_evaluation_docs(self) -> List[EvaluationDocumentWithTable]:
        df = self.get_evaluation_docs_by_path_name(
            Config.sec_filings.evaluation_data_path
        )
        questions = [
            EvaluationDocument(
                doc_id=row["file"],
                question=row["question"],
                question_id=row["question"],
                answer=row["answer"],
                search_reference=row["search reference"],
                category=row["category"],
            )
            for _, row in df.iterrows()
        ]

        questions = SECFilingLoader.add_tables_to_questions(questions)
        self.store_sample_documents([questions])

        if self.evaluation_num_documents:
            questions = questions[: self.evaluation_num_documents]

        return questions

    def get_evaluation_docs_list(self) -> List[List[EvaluationDocument]]:
        raise ValueError("For SEC Filings, only single evaluation is supported")

    def get_evaluation_docs_by_path_name(
        self,
        file_path: str = Config.sec_filings.evaluation_data_path,
    ) -> DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        df = pd.read_csv(file_path, sep=";", dtype=str, keep_default_na=False)
        return df

    @override
    def store_sample_documents(
        self,
        questions: List[List[EvaluationDocumentWithTable]],
        mapper_path: str = "./data/sec_filings/0_id_mapper_questions.json",
    ):
        return super().store_sample_documents(questions, mapper_path)

    def get_tabtree_header_evaluation_data(self) -> List[HeaderEvaluationDocument]:
        file_path: str = Config.sec_filings.evaluation_get_header_data_path
        eval_data = HeaderEvaluationDocument.from_csv(file_path=file_path)
        return eval_data

    @staticmethod
    def search_table_by_regexp(string_to_search: str, chunk: str) -> bool:
        if re.search(string_to_search, chunk, flags=re.DOTALL):
            return True
        return False

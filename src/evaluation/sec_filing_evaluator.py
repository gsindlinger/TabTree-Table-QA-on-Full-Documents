import os
from typing import List, Optional
from pandas import DataFrame

from .evaluation_document import EvaluationDocument
from ..config.config import Config
from .evaluator import Evaluator
import pandas as pd


class SECFilingEvaluator(Evaluator):
    def get_evaluation_docs(self) -> List[EvaluationDocument]:

        df = self.get_evaluation_docs_by_path_name(
            Config.sec_filings.evaluation_data_path
        )
        questions = [
            EvaluationDocument(
                doc_id=row["file"],
                question=row["question"],
                answer=row["answer"],
                search_reference=row["search reference"],
            )
            for _, row in df.iterrows()
        ]

        if self.evaluation_num_documents:
            questions = questions[: self.evaluation_num_documents]

        return questions

    def get_evaluation_docs_by_path_name(
        self,
        file_path: str = Config.sec_filings.evaluation_data_path,
    ) -> DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        df = pd.read_csv(file_path, sep=";")
        return df

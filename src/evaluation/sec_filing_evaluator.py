import time
from pandas import DataFrame

from config.config import Config
from evaluation.evaluation_results import EvaluationResults
from .evaluator import Evaluator
import pandas as pd


class SECFilingEvaluator(Evaluator):
    def get_evaluation_docs(
        self, file_path: str = Config.data.path_local_evaluation
    ) -> DataFrame:
        df = pd.read_csv(file_path)
        df = df[["question", "answer", "doc_id"]]
        return df

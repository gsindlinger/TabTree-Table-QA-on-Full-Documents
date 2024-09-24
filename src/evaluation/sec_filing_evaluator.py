import os
from pandas import DataFrame

from ..config.config import Config
from .evaluator import Evaluator
import pandas as pd


class SECFilingEvaluator(Evaluator):
    def get_evaluation_docs(
        self, file_path: str = Config.sec_filings.evaluation_data_path
    ) -> DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        df = pd.read_csv(file_path, sep=";")

        # select first 5 rows
        # df = df.head(5)
        return df

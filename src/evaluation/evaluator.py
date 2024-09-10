import time

from pandas import DataFrame
from ..pipeline import Pipeline
from .evaluation_results import EvaluationResults
from ..config.config import Config
from abc import ABC, abstractmethod


class Evaluator(ABC):
    pipeline: Pipeline

    @classmethod
    def from_config(cls, pipeline):
        mode = Config.run.mode
        if mode == "sec-filings":

            from .sec_filing_evaluator import SECFilingEvaluator

            return SECFilingEvaluator(pipeline=pipeline)

    def evaluate(self) -> EvaluationResults:
        evaluation_data = self.get_evaluation_docs()
        # check if evaluation data contains columns "question" and "answer"
        if not all(col in evaluation_data.columns for col in ["question", "answer"]):
            raise ValueError(
                "Evaluation data must contain columns 'question' and 'answer'"
            )

        predictions = []
        ground_truths = []

        # For each entry of evaluation data apply RAG model
        for _, row in evaluation_data.iterrows():
            question = row["question"]
            answer = row["answer"]

            llm_response = self.pipeline.invoke(question=question)
            predictions.append(llm_response)
            ground_truths.append(answer)

            # Sleep to avoid rate limiting
            time.sleep(1)

        # Calculate accuracy and F1 score
        evaluation_results = EvaluationResults.calculate_metrics(
            predictions, ground_truths
        )
        evaluation_results.write_to_json_file()
        return evaluation_results

    @abstractmethod
    def get_evaluation_docs(self) -> DataFrame:
        pass

import time

from pandas import DataFrame
from pydantic import BaseModel
from ..pipeline import Pipeline
from .evaluation_results import EvaluationResults, IRResults, QAResults
from ..config.config import Config
from abc import ABC, abstractmethod


class Evaluator(ABC, BaseModel):
    pipeline: Pipeline

    @classmethod
    def from_config(cls, pipeline):
        mode = Config.run.dataset
        if "sec-filings" in mode:

            from .sec_filing_evaluator import SECFilingEvaluator

            return SECFilingEvaluator(pipeline=pipeline)

    def evaluate(
        self, evaluate_qa: bool = True, evaluate_ir: bool = True
    ) -> EvaluationResults:
        if evaluate_qa:
            qa_results = self.evaluate_qa()
        if evaluate_ir:
            ir_results = self.evaluate_ir()

        full_results = EvaluationResults(qa_results=qa_results, ir_results=ir_results)
        full_results.write_to_json_file()
        return full_results

    def evaluate_ir(self) -> IRResults:
        evaluation_data = self.get_evaluation_docs()
        predictions_doc_id = []
        ground_truths_doc_id = []
        predictions_text = []
        ground_truths_search_string = []

        for _, row in evaluation_data.iterrows():
            question = row["question"]
            doc_id = row["file"]
            search_string = row["search reference"]

            retriever_response = self.pipeline.retrieve(question=question)
            predictions_text.append([doc.page_content for doc in retriever_response])
            predictions_doc_id.append(
                [doc.metadata.doc_id for doc in retriever_response]
            )
            ground_truths_doc_id.append(doc_id)
            ground_truths_search_string.append(search_string)

        return IRResults.calculate_metrics(
            predictions_doc_id,
            ground_truths_doc_id,
            predictions_text,
            ground_truths_search_string,
        )

    def evaluate_qa(self) -> QAResults:
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
            predictions.append(llm_response.strip())
            ground_truths.append(answer)

            # Sleep to avoid rate limiting
            time.sleep(1)

        # Calculate accuracy and F1 score
        evaluation_results = QAResults.calculate_metrics(predictions, ground_truths)

        # For debugging purposes, store predictions and ground truths
        evaluation_results.data = {
            "predictions": predictions,
            "ground_truths": ground_truths,
        }

        return evaluation_results

    @abstractmethod
    def get_evaluation_docs(self) -> DataFrame:
        pass

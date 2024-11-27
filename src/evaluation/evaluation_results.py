from __future__ import annotations
from abc import ABC
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


class EvaluationResults(BaseModel):
    qa_results: Optional[QAResults] = None
    ir_results: Optional[IRResults] = None
    token_counts: Optional[TokenCountsResults] = None

    @staticmethod
    def list_to_json(lst: List[EvaluationResults], only_qa_and_ir: bool = False) -> str:
        return json.dumps(
            [
                ob.model_dump(exclude={"token_counts"} if only_qa_and_ir else {})
                for ob in lst
            ],
            indent=4,
        )

    @staticmethod
    def calculate_accuracy(predictions: List[Any], ground_truths: List[Any]) -> float:
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length.")

        correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        total = len(predictions)

        if total == 0:
            return 0.0  # If no predictions are made, return accuracy as 0.0

        accuracy = correct / total
        return accuracy

    @staticmethod
    def calculate_f1_score(predictions: List[Any], ground_truths: List[Any]) -> float:
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length.")

        true_positive = sum(
            1 for p, g in zip(predictions, ground_truths) if p == g and p is not None
        )
        false_positive = sum(
            1 for p, g in zip(predictions, ground_truths) if p != g and p is not None
        )
        false_negative = sum(
            1 for p, g in zip(predictions, ground_truths) if p != g and p is None
        )

        if true_positive == 0:
            return 0.0  # Avoid division by zero, return 0 when no positive predictions are correct.

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score


class IRResults(BaseModel):
    document_recall: Optional[float] = None
    document_precision: Optional[float] = None
    document_mrr: Optional[float] = None

    chunk_recall: Optional[float] = None
    chunk_precision: Optional[float] = None
    chunk_mrr: Optional[float] = None
    similarity_scores: Optional[List[List[float]]] = None

    @staticmethod
    def calculate_metrics(
        predictions_doc_id: List[List[str] | str],
        ground_truths_doc_id: List[str],
        predictions_text: List[List[str] | str],
        ground_truths_search_string: List[str],
        similarity_scores: List[List[float]],
        retriever_num_documents: int,
    ) -> IRResults:
        document_metrics = IRResults.calculate_document_metrics(
            predictions=predictions_doc_id,
            ground_truths=ground_truths_doc_id,
            retriever_num_documents=retriever_num_documents,
        )

        chunk_metrics = IRResults.calculate_chunk_metrics(
            predictions_doc_id=predictions_doc_id,
            ground_truths_doc_id=ground_truths_doc_id,
            predictions_text=predictions_text,
            ground_truths_search_string=ground_truths_search_string,
            retriever_num_documents=retriever_num_documents,
        )

        return IRResults(
            document_recall=document_metrics.document_recall,
            document_precision=document_metrics.document_precision,
            document_mrr=document_metrics.document_mrr,
            chunk_recall=chunk_metrics.chunk_recall,
            chunk_precision=chunk_metrics.chunk_precision,
            chunk_mrr=chunk_metrics.chunk_mrr,
            similarity_scores=similarity_scores,
        )

    @staticmethod
    def calculate_document_metrics(
        predictions: List[List[str] | str],
        ground_truths: List[str],
        retriever_num_documents: int,
    ) -> IRResults:

        document_recall_count = 0
        document_precisions_count = 0
        document_mrr_count = 0

        for pos, (pred, truth) in enumerate(zip(predictions, ground_truths)):
            if truth in pred:
                document_recall_count += 1
                document_precisions_count += 1 / retriever_num_documents
                document_mrr_count += 1 / (pos + 1)

        return IRResults(
            document_recall=document_recall_count / len(predictions),
            document_precision=document_precisions_count / len(predictions),
            document_mrr=document_mrr_count / len(predictions),
        )

    @staticmethod
    def calculate_chunk_metrics(
        predictions_doc_id: List[List[str] | str],
        ground_truths_doc_id: List[str],
        predictions_text: List[List[str] | str],
        ground_truths_search_string: List[str],
        retriever_num_documents: int,
    ) -> IRResults:

        chunk_recall_count = 0
        chunk_precisions_count = 0
        chunk_mrr_count = 0

        all_documents = zip(
            predictions_doc_id,
            ground_truths_doc_id,
            predictions_text,
            ground_truths_search_string,
        )
        for (
            predictions_doc_id_temp,
            ground_truths_doc_id_temp,
            predictions_text_temp,
            ground_truths_search_string_temp,
        ) in all_documents:
            if ground_truths_doc_id_temp in predictions_doc_id_temp:
                # search ground truth string in retrieved text as regex
                for pos, predictions_text_single_temp in enumerate(
                    predictions_text_temp
                ):
                    if re.search(
                        ground_truths_search_string_temp,
                        predictions_text_single_temp,
                        flags=re.DOTALL,
                    ):
                        chunk_recall_count += 1
                        chunk_precisions_count += 1 / retriever_num_documents
                        chunk_mrr_count += 1 / (pos + 1)
                        break

        return IRResults(
            chunk_recall=chunk_recall_count / len(predictions_doc_id),
            chunk_precision=chunk_precisions_count / len(predictions_doc_id),
            chunk_mrr=chunk_mrr_count / len(predictions_doc_id),
        )


class QAResults(BaseModel):
    accuracy: float
    f1_score: float
    data: Optional[Dict] = None

    @staticmethod
    def calculate_metrics(
        predictions: List[str], ground_truths: List[str]
    ) -> QAResults:
        accuracy = EvaluationResults.calculate_accuracy(predictions, ground_truths)
        f1_score = EvaluationResults.calculate_f1_score(predictions, ground_truths)
        return QAResults(accuracy=accuracy, f1_score=f1_score)


class TokenCountsResults(BaseModel):
    preprocess_mode: str
    ids: List[str]
    num_characters: List[int]
    num_tokens: List[int]
    avg: float
    min: int
    max: int
    std: float

    @staticmethod
    def list_to_json(lst: List[TokenCountsResults]) -> str:
        return json.dumps([ob.model_dump() for ob in lst], indent=4)


class HeaderDetectionResults(BaseModel):
    accuracy_rows: Optional[float] = None
    accuracy_columns: Optional[float] = None
    predictions_rows: List[List[int]] = []
    ground_truth_rows: List[List[int]] = []
    predictions_columns: List[List[int]] = []
    ground_truth_columns: List[List[int]] = []
    advanced_analysis: Optional[Dict[str, Any]] = None

    @staticmethod
    def calculate_advanced_metrics(
        predictions: List[List[int]], ground_truths: List[List[int]]
    ) -> Tuple[List[float], List[float]]:
        advanced_accuracy = []
        advanced_f1_score = []

        # Check if the number of predictions and ground truths are the same
        # if not extend the smaller list with nones
        for pred, gt in zip(predictions, ground_truths):
            pred_mod = pred.copy()
            gt_mod = gt.copy()
            if len(pred) < len(gt):
                pred_mod.extend([None] * (len(gt) - len(pred)))  # type: ignore
            elif len(pred) > len(gt):
                gt_mod.extend([None] * (len(pred) - len(gt)))  # type: ignore
            advanced_accuracy.append(
                EvaluationResults.calculate_accuracy(pred_mod, gt_mod)
            )
            advanced_f1_score.append(
                EvaluationResults.calculate_f1_score(pred_mod, gt_mod)
            )

        return advanced_accuracy, advanced_f1_score

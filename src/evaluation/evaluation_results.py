from __future__ import annotations
from abc import ABC
import json
import re
import time
from typing import Any, Dict, List, Optional
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
        correct = 0
        for pred, truth in zip(predictions, ground_truths):
            if pred == truth:
                correct += 1
        return correct / len(predictions)

    @staticmethod
    def calculate_f1_score(predictions: List[Any], ground_truths: List[Any]) -> float:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for pred, truth in zip(predictions, ground_truths):
            if pred == truth:
                true_positives += 1
            else:
                if pred:
                    false_positives += 1
                else:
                    false_negatives += 1

        # avoid division by zero
        if true_positives == 0:
            return 0

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        return 2 * precision * recall / (precision + recall)


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

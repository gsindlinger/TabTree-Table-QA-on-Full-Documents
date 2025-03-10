from __future__ import annotations
from abc import ABC, abstractmethod
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel

from ..retrieval.document_preprocessors.table_parser.custom_html_parser import (
    HTMLTableParser,
)

from ..model.custom_document import CustomDocument
from .wiki_tables_evaluator.official_evaluator import OfficialEvaluator


class EvaluationResults(BaseModel):
    qa_results: Optional[QAResults] = None
    ir_results: Optional[IRResults] = None
    dataset_summary: Optional[DatasetSummaryResults | List[DatasetSummaryResults]] = (
        None
    )
    qa_only_results: Optional[QAOnlyResults] = None

    @staticmethod
    def list_to_json(lst: List[EvaluationResults], only_qa_and_ir: bool = False) -> str:
        return json.dumps(
            [
                ob.model_dump(exclude={"dataset_summary"} if only_qa_and_ir else {})
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

        from .sec_filing_evaluator import SECFilingEvaluator

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
                    if SECFilingEvaluator.search_table_by_regexp(
                        string_to_search=ground_truths_doc_id_temp,
                        chunk=predictions_text_single_temp,
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


class QAOnlyResults(BaseModel):
    accuracy: float
    std_dev_accuracy: Optional[float] = None
    official_accuracy: Optional[float] = None
    std_dev_official_accuracy: Optional[float] = None
    qa_results: List[QAResults]

    @staticmethod
    def combine_list_of_results(results: List[QAResults]) -> QAOnlyResults:
        accuracy = sum([result.accuracy for result in results]) / len(results)
        official_accuracy = sum(
            [result.official_accuracy for result in results if result.official_accuracy]
        ) / len(results)

        std_accuracy = np.std([result.accuracy for result in results])
        std_official_accuracy = np.std(
            [result.official_accuracy for result in results if result.official_accuracy]
        )

        return QAOnlyResults(
            accuracy=accuracy,
            std_dev_accuracy=std_accuracy,  # type: ignore
            official_accuracy=official_accuracy,
            std_dev_official_accuracy=std_official_accuracy,  # type: ignore
            qa_results=results,
        )


class QAResults(BaseModel):
    accuracy: float
    f1_score: float
    official_accuracy: Optional[float] = None
    data: Optional[Dict] = None

    @staticmethod
    def calculate_metrics(
        predictions: List[str], ground_truths: List[str]
    ) -> QAResults:
        accuracy = EvaluationResults.calculate_accuracy(predictions, ground_truths)
        official_accuracy = QAResults.calculate_official_accuracy(
            predictions, ground_truths
        )
        f1_score = EvaluationResults.calculate_f1_score(predictions, ground_truths)
        return QAResults(
            accuracy=accuracy, f1_score=f1_score, official_accuracy=official_accuracy
        )

    @staticmethod
    def calculate_official_accuracy(
        predictions: List[str], ground_truths: List[str]
    ) -> float:
        assert len(predictions) == len(
            ground_truths
        ), "Predictions and ground truths must have the same length."

        num_correct = 0
        num_examples = len(predictions)

        for pred, gt in zip(predictions, ground_truths):
            pred_values = OfficialEvaluator.to_value_list([pred])
            gt_values = OfficialEvaluator.to_value_list([gt])

            if len(gt_values) > 1:
                print(
                    "Warning: Multiple ground truth values detected. Using the first one."
                )
            if OfficialEvaluator.check_denotation(gt_values, pred_values):
                num_correct += 1

        accuracy = num_correct / num_examples if num_examples > 0 else 0.0
        return accuracy


class Statistics(BaseModel):
    avg: float
    std: float
    min: int
    max: int

    @staticmethod
    def calculate_statistics(values: List[int | float]) -> Statistics:
        return Statistics(
            avg=np.mean(values),  # type: ignore
            std=np.std(values),  # type: ignore
            min=np.min(values),
            max=np.max(values),
        )


class DatasetSummary(BaseModel):
    document_list: List[DocumentStatisticsSingle]
    token_statistics: Statistics
    character_statistics: Statistics
    table_list: Optional[List[TableSummary]] = (
        None  # table statistics for each document
    )
    table_statistics: Optional[TableSummary] = (
        None  # table statistics for all documents
    )

    @classmethod
    def get_summary(cls, documents: List[CustomDocument]) -> DatasetSummary:
        # get unique documents
        document_set = {doc.metadata.doc_id for doc in documents}  # type: ignore
        document_list: List[DocumentStatisticsSingle] = []
        for doc_id in document_set:
            document_list.append(
                DocumentStatisticsSingle.get_document_summary(
                    next(doc for doc in documents if doc.metadata.doc_id == doc_id)  # type: ignore
                )
            )
        token_statistics = Statistics.calculate_statistics(
            [doc.num_tokens for doc in document_list]
        )
        character_statistics = Statistics.calculate_statistics(
            [doc.num_characters for doc in document_list]
        )
        table_list = [TableSummary.get_summary(doc) for doc in documents]
        table_statistics = TableSummary.get_summary_all(table_list)

        return DatasetSummary(
            document_list=document_list,
            token_statistics=token_statistics,
            character_statistics=character_statistics,
            table_list=table_list,
            table_statistics=table_statistics,
        )


class DocumentStatisticsSingle(BaseModel):
    id: str
    num_tokens: int
    num_characters: int

    @classmethod
    def get_document_summary(cls, document: CustomDocument) -> DocumentStatisticsSingle:
        from .evaluator import Evaluator

        num_characters = len(document.page_content)
        num_tokens = Evaluator.num_tokens_from_string(document.page_content)

        return DocumentStatisticsSingle(
            id=document.metadata.doc_id if document.metadata is not None else "",
            num_tokens=num_tokens,
            num_characters=num_characters,
        )


class TableSummary(BaseModel):
    num_tokens: List[int]
    token_statistics: Statistics
    num_rows: List[int]
    rows_statistics: Statistics
    num_columns: List[int]
    columns_statistics: Statistics
    domains: Optional[List[str]] = None
    doc_id: Optional[str] = None

    @classmethod
    def get_summary_all(cls, table_summaries: List[TableSummary]) -> TableSummary:
        num_tokens: List[int] = [
            num for summary in table_summaries for num in summary.num_tokens
        ]
        num_rows = [num for summary in table_summaries for num in summary.num_rows]
        num_columns = [
            num for summary in table_summaries for num in summary.num_columns
        ]

        return TableSummary(
            doc_id="all tables",
            num_tokens=num_tokens,
            token_statistics=Statistics.calculate_statistics(num_tokens),  # type: ignore
            num_rows=num_rows,
            rows_statistics=Statistics.calculate_statistics(num_rows),  # type: ignore
            num_columns=num_columns,
            columns_statistics=Statistics.calculate_statistics(num_columns),  # type: ignore
        )

    @classmethod
    def get_summary(cls, document: CustomDocument) -> TableSummary:
        from .evaluator import Evaluator

        if document.splitted_content is None:
            raise ValueError(
                "Document must be splitted before calculating table summary."
            )
        tables_str = [
            table
            for table in document.splitted_content
            if table.type == "table" and table.original_content is not None
        ]

        num_rows = []
        num_columns = []
        num_tokens = []

        for table in tables_str:
            if table.original_content is None:
                raise ValueError("Table must have original content.")
            custom_parsed_table = HTMLTableParser.parse_and_clean_table(
                table.original_content
            )
            if custom_parsed_table is None:
                raise ValueError("Table couldn't be parsed.")
            num_rows.append(custom_parsed_table.rows)
            num_columns.append(custom_parsed_table.columns)
            num_tokens.append(Evaluator.num_tokens_from_string(table.content))

        return TableSummary(
            doc_id=document.metadata.doc_id if document.metadata is not None else "",
            num_tokens=num_tokens,
            token_statistics=Statistics.calculate_statistics(num_tokens),
            num_rows=num_rows,
            rows_statistics=Statistics.calculate_statistics(num_rows),
            num_columns=num_columns,
            columns_statistics=Statistics.calculate_statistics(num_columns),
        )


class DatasetSummaryResults(BaseModel):
    preprocess_mode: str
    table_serialization_mode: str
    iterations: int
    num_documents_per_iteration: int | None
    dataset_summary: DatasetSummary

    @staticmethod
    def list_to_json(lst: List[DatasetSummaryResults]) -> str:
        return json.dumps([ob.model_dump() for ob in lst], indent=4)


class HeaderDetectionResults(BaseModel):
    accuracy_rows: Optional[float] = None
    accuracy_columns: Optional[float] = None
    predictions_column_header_rows: List[List[int]] = []
    ground_truth_column_header_rows: List[List[int]] = []
    predictions_row_label_columns: List[List[int]] = []
    ground_truth_row_label_columns: List[List[int]] = []
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

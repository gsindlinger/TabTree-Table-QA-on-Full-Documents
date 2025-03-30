from __future__ import annotations
from abc import ABC, abstractmethod
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel
from sklearn.metrics import f1_score

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
    error_count: Optional[int] = None

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
    def calculate_f1_score_list(predictions: List[Any], ground_truths: List[Any]) -> float:
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length.")
    
        return f1_score(ground_truths, predictions, average='macro', zero_division=0)

        

    # calculate f1 score based on word 
    @staticmethod
    def calculate_f1_score(predictions: List[str], ground_truths: List[str]) -> float:
        """ Calculates the macro F1 score for a list of predictions and ground truths."""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length.")
        
        f1 = 0

        for pred_item, gt_item in zip(predictions, ground_truths):
            # split both on '|' with or without spaces
            pred_item = pred_item.lower().split("|")
            gt_item = gt_item.lower().split("|")
            
            pred_item.sort()
            gt_item.sort()
            
            pred_items = " ".join(pred_item).split()
            gt_items = " ".join(gt_item).split()
            
            # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
            if len(pred_items) == 0 or len(gt_items) == 0:
                f1 += int(pred_items == gt_items)
            
            common_items = set(pred_items) & set(gt_items)
    
            # if there are no common tokens then f1 = 0
            if len(common_items) == 0:
                continue
            
            prec = len(common_items) / len(pred_items)
            rec = len(common_items) / len(gt_items)
            
            f1 += 2 * (prec * rec) / (prec + rec)
        return f1 / len(predictions)

class IRResults(BaseModel):
    document_precision: Optional[float] = None
    document_mrr: Optional[float] = None

    chunk_recall_1_detailed: Optional[float] = None
    chunk_recall_3_detailed: Optional[float] = None
    chunk_recall_5_detailed: Optional[float] = None
    chunk_mrr_5_detailed: Optional[float] = None
    
    chunk_recall_1_coarse: Optional[float] = None
    chunk_recall_3_coarse: Optional[float] = None
    chunk_recall_5_coarse: Optional[float] = None
    chunk_mrr_5_coarse: Optional[float] = None
    
    similarity_scores: Optional[List[List[float]]] = None
    match_list: Optional[List[int]] = None

    @staticmethod
    def calculate_metrics(
        predictions_doc_id: List[List[str] | str],
        ground_truths_doc_id: List[str],
        predictions_text: List[List[str] | str],
        predictions_table_string: List[List[str] | str],
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
            predictions_table_string=predictions_table_string,
            ground_truths_search_string=ground_truths_search_string,
            retriever_num_documents=retriever_num_documents,
        )

        return IRResults(
            document_precision=document_metrics.document_precision,
            document_mrr=document_metrics.document_mrr,
            chunk_recall_1_detailed=chunk_metrics.chunk_recall_1_detailed,
            chunk_recall_3_detailed=chunk_metrics.chunk_recall_3_detailed,
            chunk_recall_5_detailed=chunk_metrics.chunk_recall_5_detailed,
            chunk_mrr_5_detailed=chunk_metrics.chunk_mrr_5_detailed,
            chunk_recall_1_coarse=chunk_metrics.chunk_recall_1_coarse,
            chunk_recall_3_coarse=chunk_metrics.chunk_recall_3_coarse,
            chunk_recall_5_coarse=chunk_metrics.chunk_recall_5_coarse,
            chunk_mrr_5_coarse=chunk_metrics.chunk_mrr_5_coarse,
            similarity_scores=similarity_scores,
            match_list=chunk_metrics.match_list,
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

        for (pred, truth) in zip(predictions, ground_truths):
            if truth in pred:
                document_recall_count += 1
                
            for pos, doc_id in enumerate(pred):
                if doc_id == truth:
                    document_precisions_count += 1 / retriever_num_documents
                    
            for pos, doc_id in enumerate(pred):
                if doc_id == truth:
                    document_mrr_count += 1 / (pos + 1)
                    break

        return IRResults(
            document_precision=document_precisions_count / len(predictions),
            document_mrr=document_mrr_count / len(predictions),
        )

    @staticmethod
    def calculate_chunk_metrics(
        predictions_doc_id: List[List[str] | str],
        ground_truths_doc_id: List[str],
        predictions_text: List[List[str] | str],
        predictions_table_string: List[List[str] | str],
        ground_truths_search_string: List[str],
        retriever_num_documents: int,
    ) -> IRResults:

        from .sec_filing_evaluator import SECFilingEvaluator
        
        match_list = []

        chunk_recall_count_detailed_1 = 0.0        
        chunk_recall_count_detailed_3 = 0.0
        chunk_recall_count_detailed_5 = 0.0
        chunk_mrr_count_detailed_5 = 0.0
        
        detailed_results_mapper = {
            1: [chunk_recall_count_detailed_1], 
            3: [chunk_recall_count_detailed_3], 
            5: [chunk_recall_count_detailed_5, chunk_mrr_count_detailed_5]
            }
        
        chunk_recall_count_coarse_1 = 0.0
        chunk_recall_count_coarse_3 = 0.0
        chunk_recall_count_coarse_5 = 0.0
        chunk_mrr_count_coarse_5 = 0.0
        
        normal_coarse_mismatch_counter = 0
        
        coarse_results_mapper = {
            1: [chunk_recall_count_coarse_1], 
            3: [chunk_recall_count_coarse_3], 
            5: [chunk_recall_count_coarse_5, chunk_mrr_count_coarse_5]
            }
        
        

        all_documents = zip(
            predictions_doc_id,
            ground_truths_doc_id,
            predictions_text,
            predictions_table_string,
            ground_truths_search_string,
            
        )
        for (
            predictions_doc_id_temp,
            ground_truths_doc_id_temp,
            predictions_text_temp,
            predictions_table_string_temp,
            ground_truths_search_string_temp,
        ) in all_documents:
            # search ground truth string in retrieved text as regex with normal search
            detailed_match = False
            coarse_match = False
            match_list_checker = True
            for i in [1, 3, 5]:
                predictions_text_temp_reduced = predictions_text_temp[:i]
            
                for pos, predictions_text_single_temp in enumerate(
                    predictions_text_temp_reduced
                ):
                    if ground_truths_doc_id_temp != predictions_doc_id_temp[pos]:
                        continue
                    
                    if SECFilingEvaluator.search_table_by_regexp(
                        string_to_search=ground_truths_search_string_temp,
                        chunk=predictions_text_single_temp,
                        chunk_table_string=predictions_table_string_temp[pos],
                        mode="detailed"
                    ):
                        detailed_results_mapper[i][0] += 1 # recall
                        if i == 5:
                            detailed_match = True
                            detailed_results_mapper[5][1] += (1 / (pos + 1))# mrr
                        break
                    
                # search ground truth string in retrieved text as regex with detailed mode
                for pos, predictions_text_single_temp in enumerate(
                    predictions_text_temp_reduced
                ):
                    if ground_truths_doc_id_temp != predictions_doc_id_temp[pos]:
                        continue
                    
                    
                    if SECFilingEvaluator.search_table_by_regexp(
                        string_to_search=ground_truths_search_string_temp,
                        chunk=predictions_text_single_temp,
                        chunk_table_string=predictions_table_string_temp[pos],
                        mode="normal"
                    ):
                        coarse_results_mapper[i][0] += 1
                        if match_list_checker:
                            match_list.append(i)
                            match_list_checker = False
                        if i == 5:
                            coarse_match = True
                            coarse_results_mapper[5][1] += (1 / (pos + 1))
                        break
                    
                        
                if i == 5:
                    if not detailed_match == coarse_match:
                        normal_coarse_mismatch_counter += 1
                        
                    if not coarse_match:
                        match_list.append(-1)
                        
            
        logging.info(f"Number of normal-coarse mismatches: {normal_coarse_mismatch_counter}")
        
        return IRResults(
            chunk_recall_1_detailed=detailed_results_mapper[1][0]/len(predictions_doc_id),
            chunk_recall_3_detailed=detailed_results_mapper[3][0]/len(predictions_doc_id),
            chunk_recall_5_detailed=detailed_results_mapper[5][0]/len(predictions_doc_id),
            chunk_mrr_5_detailed=detailed_results_mapper[5][1]/len(predictions_doc_id),
            chunk_recall_1_coarse=coarse_results_mapper[1][0]/len(predictions_doc_id),
            chunk_recall_3_coarse=coarse_results_mapper[3][0]/len(predictions_doc_id),
            chunk_recall_5_coarse=coarse_results_mapper[5][0]/len(predictions_doc_id),
            chunk_mrr_5_coarse=coarse_results_mapper[5][1]/len(predictions_doc_id),
            match_list=match_list,
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
    def postprocess_qa_results(predictions: List[str], ground_truths: List[str]):
        for i in range(len(predictions)):
            predictions[i] = re.sub(r"\s+", " ", predictions[i])
            ground_truths[i] = re.sub(r"\s+", " ", ground_truths[i])
            
            if "%" in predictions[i] and "%" in ground_truths[i]:
                predictions[i] = predictions[i].replace("%", "").strip()
                
            # equalize thousands separators if answer contains only digits
            if predictions[i].replace(",", "").replace(".","").isdigit() or ground_truths[i].replace(",", "").replace(".","").isdigit():
                predictions[i] = predictions[i].replace(",", "")
                ground_truths[i] = ground_truths[i].replace(",", "")
            # if ground truth is a number with decimals, ensure that prediction is rounded to the same number of decimal places
            if ground_truths[i].replace(".", "").isdigit():
                if "." in ground_truths[i]:
                    decimals = len(ground_truths[i].split(".")[1])
                    predictions[i] = f"{float(predictions[i]):.{decimals}f}"
        return predictions, ground_truths

    @staticmethod
    def calculate_metrics(
        predictions: List[str], ground_truths: List[str]
    ) -> QAResults:
        
        predictions, ground_truths = QAResults.postprocess_qa_results(predictions, ground_truths)
        
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
                pred_mod.extend([-1] * (len(gt) - len(pred)))  # type: ignore
            elif len(pred) > len(gt):
                gt_mod.extend([-1] * (len(pred) - len(gt)))  # type: ignore
            advanced_accuracy.append(
                EvaluationResults.calculate_accuracy(pred_mod, gt_mod)
            )
            advanced_f1_score.append(
                EvaluationResults.calculate_f1_score_list(pred_mod, gt_mod)
            )

        return advanced_accuracy, advanced_f1_score

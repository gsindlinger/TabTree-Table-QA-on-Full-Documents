from __future__ import annotations
from abc import ABC
import re
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class EvaluationResults(ABC, BaseModel):
    qa_results: Optional[QAResults] = None
    ir_results: Optional[IRResults] = None
    
    def write_to_json_file(
        self, folder_path: str = "./data/evaluation"
    ):
        
        # add timestamp to file_path in format yyyy-mm-dd-hh-mm-ss
        file_path = f"{folder_path}/evaluation_results-{time.strftime("%Y-%m-%d-%H-%M-%S")}.json"
        json_evaluation_results = self.model_dump_json(indent=4)
        # write json to file
        with open(file_path, "w") as file:
            file.write(json_evaluation_results)
            
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

class IRResults(EvaluationResults):
    document_accuracy: float
    chunk_accuracy: float
    
    @staticmethod
    def calculate_metrics(
        predictions_doc_id: List[List[str] | str],
        ground_truths_doc_id: List[str],
        predictions_text: List[List[str] | str],
        ground_truths_search_string: List[str],
    ) -> IRResults:
        document_accuracy = IRResults.calculate_accuracy(
            predictions_doc_id, ground_truths_doc_id
        )
        
        chunk_accuracy = IRResults.calculate_chunk_accuracy(
            predictions_doc_id, ground_truths_doc_id, predictions_text, ground_truths_search_string
        )
         
        return IRResults(document_accuracy=document_accuracy, chunk_accuracy=chunk_accuracy)
    
    @staticmethod
    def calculate_accuracy(predictions: List[List[str] | str], ground_truths: List[str]) -> float:
        correct = 0
        for pred, truth in zip(predictions, ground_truths):
            if truth in pred:
                correct += 1
                
        return correct / len(predictions)
    
    
    @staticmethod
    def calculate_chunk_accuracy(
        predictions_doc_id: List[List[str] | str],
        ground_truths_doc_id: List[str],
        predictions_text: List[List[str] | str],
        ground_truths_search_string: List[str]) -> float:
        
        chunk_accuracy_count = 0
        
        chunk_accuracy_count = 0
        all_documents = zip(predictions_doc_id, ground_truths_doc_id, predictions_text, ground_truths_search_string)
        for predictions_doc_id, ground_truths_doc_id, predictions_text, ground_truths_search_string in all_documents:
            if ground_truths_doc_id in predictions_doc_id:
                # search ground truth string in retrieved text as regex
                for predictions_text_single in predictions_text:
                    if re.search(ground_truths_search_string, predictions_text_single):
                        chunk_accuracy_count += 1
                        break
                    
        return chunk_accuracy_count / len(predictions_doc_id)  
        

class QAResults(EvaluationResults):
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

    

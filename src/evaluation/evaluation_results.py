from __future__ import annotations
import time
from typing import List
from pydantic import BaseModel


class EvaluationResults(BaseModel):
    accuracy: float
    f1_score: float

    def write_to_json_file(
        self, folder_path: str = "./data/evaluation"
    ):
        
        # add timestamp to file_path in format yyyy-mm-dd-hh-mm-ss
        file_path = f"{folder_path}/evaluation_results-{time.strftime("%Y-%m-%d-%H-%M-%S")}.json"
        self.model_dump_json(file_path, indent=4)

    @staticmethod
    def calculate_metrics(
        predictions: List[str], ground_truths: List[str]
    ) -> EvaluationResults:
        accuracy = EvaluationResults.calculate_accuracy(predictions, ground_truths)
        f1_score = EvaluationResults.calculate_f1_score(predictions, ground_truths)
        return EvaluationResults(accuracy=accuracy, f1_score=f1_score)

    @staticmethod
    def calculate_accuracy(predictions, ground_truths):
        correct = 0
        for pred, truth in zip(predictions, ground_truths):
            if pred == truth:
                correct += 1
        return correct / len(predictions)

    @staticmethod
    def calculate_f1_score(predictions, ground_truths):
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
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        return 2 * precision * recall / (precision + recall)

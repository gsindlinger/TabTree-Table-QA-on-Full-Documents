from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional


from matplotlib import pyplot as plt
import numpy as np
from pydantic import BaseModel
import tiktoken

from ..config.config_model import GeneralConfig, RunConfig
from .evaluation_document import EvaluationDocument
from ..config.config import Config
from ..retrieval.document_preprocessors.document_preprocessor import (
    DocumentPreprocessor,
)

from ..retrieval.document_loaders.document_loader import DocumentLoader
from ..retrieval.document_preprocessors.preprocess_config import PreprocessConfig
from ..pipeline import Pipeline
from .evaluation_results import (
    EvaluationResults,
    IRResults,
    QAResults,
    TokenCountsResults,
)
from abc import ABC, abstractmethod


class Evaluator(ABC, BaseModel):
    pipeline: Pipeline
    preprocess_config: PreprocessConfig
    evaluate_qa: bool = False
    evaluate_ir: bool = False
    run_document_analysis: bool = False
    retriever_num_documents: int
    evaluation_num_documents: Optional[int] = None
    evaluation_results: Optional[EvaluationResults] = None

    @classmethod
    def from_config(
        cls,
        dataset: str,
        preprocess_config: PreprocessConfig,
        pipeline: Pipeline,
        retriever_num_documents: int,
    ) -> Evaluator:

        if Config.evaluation.num_of_documents:
            evaluation_num_documents = Config.evaluation.num_of_documents
        else:
            evaluation_num_documents = None

        match dataset:
            case "wiki-table-questions":
                from .wiki_table_questions_evaluator import WikiTableQuestionsEvaluator

                return WikiTableQuestionsEvaluator(
                    pipeline=pipeline,
                    preprocess_config=preprocess_config,
                    retriever_num_documents=retriever_num_documents,
                    evaluate_ir=Config.evaluation.evaluate_ir,
                    evaluate_qa=Config.evaluation.evaluate_qa,
                    run_document_analysis=Config.evaluation.run_document_analysis,
                    evaluation_num_documents=evaluation_num_documents,
                )
            case "sec-filings":
                from .sec_filing_evaluator import SECFilingEvaluator

                return SECFilingEvaluator(
                    pipeline=pipeline,
                    preprocess_config=preprocess_config,
                    retriever_num_documents=retriever_num_documents,
                    evaluate_ir=Config.evaluation.evaluate_ir,
                    evaluate_qa=Config.evaluation.evaluate_qa,
                    run_document_analysis=Config.evaluation.run_document_analysis,
                    evaluation_num_documents=evaluation_num_documents,
                )
            case _:
                raise ValueError(f"Unknown dataset: {dataset}")

    @abstractmethod
    def get_evaluation_docs(self) -> List[EvaluationDocument]:
        pass

    def evaluate(
        self,
        evaluate_qa: Optional[bool] = None,
        evaluate_ir: Optional[bool] = None,
        run_document_analysis: Optional[bool] = None,
        write_to_file=False,
    ) -> EvaluationResults:

        if evaluate_qa:
            self.evaluate_qa = evaluate_qa
        if evaluate_ir:
            self.evaluate_ir = evaluate_ir
        if run_document_analysis:
            self.run_document_analysis = run_document_analysis

        qa_results = None
        ir_results = None
        token_counts = None
        if self.evaluate_qa:
            qa_results = self._evaluate_qa()
        if self.evaluate_ir:
            ir_results = self._evaluate_ir()
        if self.run_document_analysis:
            token_counts = self._get_token_counts(
                preprocess_config=self.preprocess_config
            )

        full_results = EvaluationResults(
            qa_results=qa_results, ir_results=ir_results, token_counts=token_counts
        )
        if write_to_file and (qa_results or ir_results):
            full_results.write_to_json_file(only_qa_and_ir=(not run_document_analysis))

        return full_results

    def _evaluate_ir(self) -> IRResults:
        evaluation_data = self.get_evaluation_docs()
        predictions_doc_id = []
        ground_truths_doc_id = []
        predictions_text = []
        ground_truths_search_string = []
        similarity_scores = []

        for row in evaluation_data:
            question = row.question
            doc_id = row.doc_id
            search_string = row.search_reference

            retriever_response = self.pipeline.retrieve(question=question)
            predictions_text.append([doc.page_content for doc in retriever_response])
            predictions_doc_id.append(
                [doc.metadata.doc_id for doc in retriever_response]
            )
            ground_truths_doc_id.append(doc_id)
            ground_truths_search_string.append(search_string)

            similarity_scores.append(
                [doc.metadata.similarity_score for doc in retriever_response]
            )

        return IRResults.calculate_metrics(
            predictions_doc_id=predictions_doc_id,
            ground_truths_doc_id=ground_truths_doc_id,
            predictions_text=predictions_text,
            ground_truths_search_string=ground_truths_search_string,
            similarity_scores=similarity_scores,
            retriever_num_documents=self.retriever_num_documents,
        )

    def _evaluate_qa(self) -> QAResults:
        evaluation_data = self.get_evaluation_docs()
        predictions = []
        ground_truths = []

        # For each entry of evaluation data apply RAG model
        for row in evaluation_data:
            question = row.question
            answer = row.answer

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

    def _get_token_counts(
        self, preprocess_config: PreprocessConfig, num_of_documents: int = 100
    ) -> TokenCountsResults:
        """
        Analyze the documents which are used for the application.
        Contains the following approach:
            - Count tokens based on tiktoken counts of cl100k_base
        """
        document_loader = DocumentLoader.from_config()
        loaded_documents = document_loader.load_documents(
            num_of_documents=num_of_documents
        )

        document_preprocessor = DocumentPreprocessor.from_config(
            preprocess_config=preprocess_config
        )
        documents = document_preprocessor.preprocess_multiple_documents(
            loaded_documents
        )

        nums_list = []
        for document in documents:
            doc_dict = {}
            doc_dict["doc_id"] = document.metadata.doc_id
            doc_dict["num_tokens"] = Evaluator.num_tokens_from_string(
                document.page_content
            )
            doc_dict["num_characters"] = len(document.page_content)
            nums_list.append(doc_dict)

        nums_list = sorted(nums_list, key=lambda x: x["num_tokens"], reverse=True)
        num_tokens = [x["num_tokens"] for x in nums_list]
        num_characters = [x["num_characters"] for x in nums_list]
        ids = [x["doc_id"] for x in nums_list]

        token_counts = TokenCountsResults(
            preprocess_mode=preprocess_config.name,
            ids=ids,
            num_characters=num_characters,
            num_tokens=num_tokens,
            avg=sum(num_tokens) / len(num_tokens),
            min=min(num_tokens),
            max=max(num_tokens),
            std=np.std(num_tokens),  # type: ignore
        )

        logging.info(
            "Analysed documents for preprocess_mode: " + preprocess_config.name
        )
        return token_counts

    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    @staticmethod
    def generate_analysis_plot(token_counts: List[TokenCountsResults], file_path: str):
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

        avg_token_counts = [entry.avg for entry in token_counts]
        std_token_counts = [entry.std for entry in token_counts]
        preprocess_modes = [entry.preprocess_mode for entry in token_counts]

        # Create bar plot with error bars
        plt.bar(
            preprocess_modes,
            avg_token_counts,
            yerr=std_token_counts,
            align="center",
            alpha=0.5,
            ecolor="black",
            capsize=10,
        )

        # Add dashed lines and text values on top of the bars
        for i, avg in enumerate(avg_token_counts):
            rounded_avg = round(avg)
            plt.text(
                i,
                avg + std_token_counts[i],
                f"{rounded_avg:,}",
                ha="center",
                va="bottom",
                color="black",
                fontstyle="italic",
            )
            plt.axhline(y=avg, color="gray", linestyle="--", linewidth=0.75)

        # Labels and title
        plt.ylabel("Average number of tokens")
        plt.xlabel("Preprocess mode")
        plt.yscale("log")
        plt.title("Average number of tokens per document")

        # Save the plot
        plt.savefig(file_path)

    @staticmethod
    def write_analysis_to_json(token_counts: List[TokenCountsResults], file_path: str):
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

        # Write all results to disk
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(TokenCountsResults.list_to_json(token_counts))

    @staticmethod
    def write_evaluation_to_json(
        evaluation_results: List[EvaluationResults],
        file_path: str,
        names: Optional[List[str]] = None,
        retriever_num_documents: Optional[int] = None,
    ) -> None:

        modified_results = []
        for i, result in enumerate(evaluation_results):
            result_dict = result.model_dump(exclude={"token_counts"})
            if names:
                result_dict["name"] = names[i]
            modified_results.append(result_dict)

        data = {
            "retriever_num_documents": retriever_num_documents,
            "evaluation_results": modified_results,
        }

        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def save_evaluation_results(
        evaluation_results: List[EvaluationResults],
        retriever_num_documents: Optional[int] = None,
        names: Optional[List[str]] = None,
    ) -> None:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

        file_path_json = f"./data/evaluation/evaluation_results-{timestamp}.json"

        file_path_token_counts = (
            f"./data/evaluation/token_counts/analysis-{timestamp}.json"
        )

        file_path_token_counts_plot = (
            f"./data/evaluation/token_counts/plots/{timestamp}.png"
        )

        if any(result.ir_results or result.qa_results for result in evaluation_results):
            Evaluator.write_evaluation_to_json(
                evaluation_results=evaluation_results,
                file_path=file_path_json,
                retriever_num_documents=retriever_num_documents,
                names=names,
            )

        if not any(result.token_counts is None for result in evaluation_results):
            Evaluator.write_analysis_to_json(
                token_counts=[result.token_counts for result in evaluation_results],
                file_path=file_path_token_counts,
            )
            Evaluator.generate_analysis_plot(
                token_counts=[result.token_counts for result in evaluation_results],
                file_path=file_path_token_counts_plot,
            )

    @classmethod
    def run_single_evaluation(
        cls,
        config: GeneralConfig,
        run_setup: RunConfig,
        save_results: bool = True,
    ) -> Evaluator:

        evaluator = cls.from_config(
            dataset=config.dataset,
            preprocess_config=config.preprocess_config,
            pipeline=run_setup.pipeline,
            retriever_num_documents=config.retriever_num_documents,
        )

        if evaluator.evaluate_qa or evaluator.evaluate_ir:
            run_setup.indexing_service.embed_documents(
                preprocess_config=config.preprocess_config,
                overwrite_existing_collection=False,
            )

        evaluator.evaluation_results = evaluator.evaluate()

        if evaluator.evaluation_results and save_results:
            Evaluator.save_evaluation_results(
                evaluation_results=[evaluator.evaluation_results],
                retriever_num_documents=config.retriever_num_documents,
                names=[config.preprocess_config.name],
            )
        return evaluator

    @classmethod
    def run_multi_evaluation(
        cls,
        general_config: GeneralConfig,
        preprocess_configs: List[PreprocessConfig],
        retriever_num_documents: List[int],
    ) -> None:
        for retriever_num_documents_temp in retriever_num_documents:
            evaluation_results = []
            for preprocess_config in preprocess_configs:
                general_config.update_by_preprocess_config(preprocess_config)
                run_setup = general_config.setup_run_config()
                single_evaluation = Evaluator.run_single_evaluation(
                    config=general_config,
                    run_setup=run_setup,
                )
                evaluation_results.append(single_evaluation.evaluation_results)

            Evaluator.save_evaluation_results(
                evaluation_results=evaluation_results,
                retriever_num_documents=retriever_num_documents_temp,
                names=[
                    preprocess_config.name for preprocess_config in preprocess_configs
                ],
            )

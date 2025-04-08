from __future__ import annotations

from enum import Enum
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional


from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from pydantic import BaseModel
import tiktoken


from ..retrieval.document_preprocessors.table_serializer import HTMLSerializer

from ..model.custom_document import CustomDocument

from ..model.tabtree.tabtree_serializer import TabTreeSerializer
from ..model.tabtree.tabtree_service import TabTreeService
from ..retrieval.document_preprocessors.table_parser.custom_html_parser import (
    HTMLTableParser,
)


from ..retrieval.indexing_service import IndexingService


from ..pipeline import (
    QuestionCategoryPipeline,
    QuestionDomainPipeline,
    RAGPipeline,
    TableHeaderRowsPipeline,
    TableQAPipeline,
)
from ..config.config_model import (
    LLMConfig,
    RAGConfig,
    RunSetup,
    RunSetupRAG,
    RunSetupTableQA,
)
from .evaluation_document import (
    EvaluationDocument,
    EvaluationDocumentWithTable,
    HeaderEvaluationDocument,
    HeaderEvaluationDocumentReduced,
)
from ..config.config import Config
from ..retrieval.document_preprocessors.document_preprocessor import (
    DocumentPreprocessor,
)

from ..retrieval.document_loaders.document_loader import DocumentLoader
from ..retrieval.document_preprocessors.preprocess_config import PreprocessConfig
from .evaluation_results import (
    DocumentStatisticsSingle,
    DatasetSummary,
    EvaluationResults,
    HeaderDetectionResults,
    IRResults,
    QAOnlyResults,
    QAResults,
    DatasetSummaryResults,
    Statistics,
    TableSummary,
)
from abc import ABC, abstractmethod


class EvaluationType(str, Enum):
    EVALUATE_QA = "evaluate_qa"
    EVALUATE_IR = "evaluate_ir"
    RUN_DOCUMENT_ANALYSIS = "run_document_analysis"
    EVALUATE_TABLE_QA_ONLY = "evaluate_table_qa_only"
    EVALUATE_GET_HEADERS = "evaluate_get_headers"

    @staticmethod
    def get_evaluation_type_from_config():
        evaluation_types = []
        if Config.evaluation.evaluate_qa:
            evaluation_types.append(EvaluationType.EVALUATE_QA)
        if Config.evaluation.evaluate_ir:
            evaluation_types.append(EvaluationType.EVALUATE_IR)
        if Config.evaluation.run_document_analysis:
            evaluation_types.append(EvaluationType.RUN_DOCUMENT_ANALYSIS)
        if Config.evaluation.evaluate_table_qa_only:
            evaluation_types.append(EvaluationType.EVALUATE_TABLE_QA_ONLY)
        if Config.evaluation.evaluate_get_headers:
            evaluation_types.append(EvaluationType.EVALUATE_GET_HEADERS)
        return evaluation_types


class Evaluator(ABC, BaseModel):
    run_setup: RunSetup
    evaluation_types: List[EvaluationType]
    llm_config: LLMConfig
    retriever_num_documents: Optional[int] = None
    evaluation_num_documents: Optional[int] = None
    evaluation_results: Optional[EvaluationResults] = None
    evaluation_iterations: int = 1

    @classmethod
    def from_config(
        cls,
        evaluation_types: List[EvaluationType],
        run_setup: RunSetup,
        llm_config: LLMConfig,
        retriever_num_documents: Optional[int] = None,
    ) -> Evaluator:

        if Config.evaluation.num_of_documents:
            evaluation_num_documents = Config.evaluation.num_of_documents
        else:
            evaluation_num_documents = 300

        if Config.evaluation.iterations:
            evaluation_iterations = Config.evaluation.iterations
        else:
            evaluation_iterations = 1

        match llm_config.dataset:
            case "wiki-table-questions":
                from .wiki_tables_evaluator.wiki_table_questions_evaluator import (
                    WikiTableQuestionsEvaluator
                )

                return WikiTableQuestionsEvaluator(
                    run_setup=run_setup,
                    retriever_num_documents=retriever_num_documents,
                    evaluation_types=evaluation_types,
                    llm_config=llm_config,
                    evaluation_num_documents=evaluation_num_documents,
                    evaluation_iterations=evaluation_iterations,
                )
            case "sec-filings":
                from .sec_filing_evaluator import SECFilingEvaluator

                return SECFilingEvaluator(
                    run_setup=run_setup,
                    retriever_num_documents=retriever_num_documents,
                    evaluation_types=evaluation_types,
                    llm_config=llm_config,
                    evaluation_num_documents=evaluation_num_documents,
                    evaluation_iterations=evaluation_iterations,
                )
            case _:
                raise ValueError(f"Unknown dataset: {llm_config.dataset}")

    @abstractmethod
    def get_evaluation_docs(
        self,
    ) -> List[EvaluationDocument]:
        pass

    @abstractmethod
    def get_evaluation_docs_list(self) -> List[List[EvaluationDocument]]:
        pass

    def evaluate(
        self,
    ) -> EvaluationResults:
        from .wiki_tables_evaluator.wiki_table_questions_evaluator import (
            WikiTableQuestionsEvaluator,
        )

        qa_only_results = None
        qa_results = None
        ir_results = None
        dataset_summary = None

        if EvaluationType.EVALUATE_GET_HEADERS in self.evaluation_types:
            self.evaluate_table_header_detection_all()
        
        if (
            EvaluationType.RUN_DOCUMENT_ANALYSIS in self.evaluation_types
        ):
            if isinstance(self, WikiTableQuestionsEvaluator):
                qa_docs = self.get_evaluation_docs_list()
                dataset_summary = self._get_dataset_summary(evaluation_data=qa_docs) # type: ignore
            else:
                evaluation_docs = self.get_evaluation_docs()
                dataset_summary = self._get_dataset_summary(
                    evaluation_data=evaluation_docs # type: ignore
                )
        
        if EvaluationType.EVALUATE_TABLE_QA_ONLY in self.evaluation_types:
            if isinstance(self, WikiTableQuestionsEvaluator):
                qa_docs = self.get_evaluation_docs_list()
            else:
                self.evaluation_iterations = 1
                self.evaluation_num_documents = len(self.get_evaluation_docs())
                qa_docs = [self.get_evaluation_docs()]
            qa_only_results = self._evaluate_table_qa_only(evaluation_data=qa_docs)    # type: ignore
        if EvaluationType.EVALUATE_QA in self.evaluation_types:
            evaluation_docs = self.get_evaluation_docs()
            qa_results = self._evaluate_qa(evaluation_data=evaluation_docs) # type: ignore
        if EvaluationType.EVALUATE_IR in self.evaluation_types:
            evaluation_docs = self.get_evaluation_docs()
            ir_results = self._evaluate_ir(evaluation_data=evaluation_docs) # type: ignore
            
        if EvaluationType.EVALUATE_QA in self.evaluation_types and isinstance(self.run_setup.pipeline, RAGPipeline):
            chunk_size_exceed_count = self.run_setup.pipeline.retriever.exceed_chunk_size # type: ignore
        else:
            chunk_size_exceed_count = None

        full_results = EvaluationResults(
            qa_results=qa_results,
            ir_results=ir_results,
            dataset_summary=dataset_summary,
            qa_only_results=qa_only_results,
            error_count=self.run_setup.pipeline.error_count,
            chunk_size_exceed_count=chunk_size_exceed_count,
            
        )
        return full_results

    def _evaluate_ir(self, evaluation_data: List[EvaluationDocument]) -> IRResults:
        predictions_doc_id = []
        ground_truths_doc_id = []
        predictions_text = []
        predictions_table_string = []
        ground_truths_search_string = []
        similarity_scores = []

        if not self.retriever_num_documents:
            raise ValueError("Retriever num documents must be set to run IR evaluation")

        for row in evaluation_data:
            question = row.question
            doc_id = row.doc_id
            search_string = row.search_reference

            if not isinstance(self.run_setup.pipeline, RAGPipeline):
                raise ValueError(
                    "Evaluator must be of type RunSetupRAG to run IR evaluation"
                )
            retriever_response = self.run_setup.pipeline.retrieve(question=question)
            predictions_text.append([doc.page_content for doc in retriever_response])
            predictions_doc_id.append(
                [doc.metadata.doc_id for doc in retriever_response]  # type: ignore
            )
            predictions_table_string.append(
                [doc.metadata.table_string for doc in retriever_response] # type: ignore
            )
            
            ground_truths_doc_id.append(doc_id)
            ground_truths_search_string.append(search_string)

            similarity_scores.append(
                [doc.metadata.similarity_score for doc in retriever_response]  # type: ignore
            )
            
            # Sleep to avoid rate limiting
            time.sleep(1)

        ir_results = IRResults.calculate_metrics(
            predictions_doc_id=predictions_doc_id,
            ground_truths_doc_id=ground_truths_doc_id,
            predictions_text=predictions_text,
            predictions_table_string=predictions_table_string,
            ground_truths_search_string=ground_truths_search_string,
            similarity_scores=similarity_scores,
            retriever_num_documents=self.retriever_num_documents,
        )
        
        self.write_match_list_to_dict(ir_results=ir_results, evaluation_data=evaluation_data)
        
        return ir_results
    
    def write_match_list_to_dict(self, ir_results: IRResults, evaluation_data: List[EvaluationDocumentWithTable]) -> None:
        # write ir_results.match_list to disk with question category and question html_table
        match_list_dict = []
        for i, match in enumerate(ir_results.match_list):
            html_table = HTMLSerializer().serialize_table_to_str(
                table_str=evaluation_data[i].html_table,
            )
            
            match_dict = {
                "question": evaluation_data[i].question,
                "question_category": evaluation_data[i].category,
                "question_html_table_parsed": html_table,
                "question_html_table": evaluation_data[i].html_table,
                "match": match,
            }
            match_list_dict.append(match_dict)
        # timestamp
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        # create directory if it does not exist
        directory = os.path.dirname("../data/evaluation/ir/")
        os.makedirs(directory, exist_ok=True)
        # write to json
        with open(f"./data/evaluation/ir/match_list-{timestamp}.json", "w") as f:
            json.dump(match_list_dict, f, indent=4)

    def _evaluate_table_qa_only(
        self, evaluation_data: List[List[EvaluationDocument]]
    ) -> QAOnlyResults:
        results: List[QAResults] = []
        for docs in evaluation_data:
            results.append(self._evaluate_table_qa_single(docs))
        return QAOnlyResults.combine_list_of_results(results)

    def _evaluate_table_qa_single(
        self, evaluation_data: List[EvaluationDocument]
    ) -> QAResults:
        predictions = []
        predictions_parsed = []
        ground_truths = []

        for row in evaluation_data:
            # only run serialization again if doc_id changes
            if isinstance(row, EvaluationDocumentWithTable) and row.html_table.strip() == "":
                continue
            if not isinstance(row, EvaluationDocumentWithTable):
                raise ValueError(
                    "Data must be of type EvaluationDocumentWithTable to run TableQA evaluation"
                )
            
            if not self.llm_config.table_serializer:
                raise ValueError(
                    "Table serializer must be set to run TableQA evaluation"
                )

            table_serializer = self.llm_config.table_serializer
            table = table_serializer.serialize_table_to_str(
                table_str=row.html_table,
            )
            table = table[0]
            # doc_id = row.doc_id

            question = row.question
            answer = row.answer

            if not isinstance(self.run_setup.pipeline, TableQAPipeline):
                raise ValueError(
                    "Evaluator must be of type RunSetupTableQA to run TableQA evaluation"
                )
            if not table:
                logging.warning(
                    f"Table could not be serialized for question: {question}"
                )
                continue

            llm_response = self.run_setup.pipeline.invoke(
                table=table, question=question, table_title=row.table_title
            )
            print(f"Question: {question}")
            print(f"Answer: {llm_response}")
            logging.info(f"LLM Question: {question}")
            logging.info(f"LLM Response: {llm_response}")
            
            predictions.append(llm_response.strip())
            predictions_parsed.append(self.parse_llm_response(llm_response))
            ground_truths.append(answer)
            
            # Sleep to avoid rate limiting
            time.sleep(1)

        evaluation_results = QAResults.calculate_metrics(
            predictions_parsed, ground_truths
        )

        evaluation_results.data = {
            "predictions": predictions,
            "predictions_parsed:": predictions_parsed,
            "ground_truths": ground_truths,
            "num_documents": len(evaluation_data),
        }

        return evaluation_results

    def parse_llm_response(self, llm_response: str) -> str:
        """ Find pattern in string: Answer: <Answer> and return list of answers split by '|'. If only one answer is found, return it directly. """
        
        # Find Pattern in string: Answer: <Answer>
        answer = re.search(r"Answer: (.*)", llm_response)

        # Split string by semicolon and return first part
        if answer:
            answer = answer.group(1)
            answer = answer.split(";")
            # Remove trailing LLM artifacts like '**', '*' or other symbols and split multiple answers by '|'
            return "|".join([re.sub(r"[\*\s]+$", "", a.strip()) for a in answer])


        return "None"

    def _evaluate_qa(self, evaluation_data: List[EvaluationDocument]) -> QAResults:
        predictions = []
        predictions_parsed = []
        ground_truths = []
        
        

        # For each entry of evaluation data apply RAG model
        for row in evaluation_data:
            question = row.question
            answer = row.answer

            if not isinstance(self.run_setup.pipeline, RAGPipeline):
                raise ValueError(
                    "Evaluator must be of type RunSetupRAG to run QA evaluation"
                )
            llm_response = self.run_setup.pipeline.invoke(question=question)
            
            print(f"Question: {question}")
            print(f"Answer: {llm_response}")
            logging.info(f"LLM Question: {question}")
            logging.info(f"LLM Response: {llm_response}")
            
            predictions.append(llm_response.strip())
            predictions_parsed.append(self.parse_llm_response(llm_response))
            ground_truths.append(answer)

            # Sleep to avoid rate limiting
            time.sleep(1)

        # Calculate accuracy and F1 score
        evaluation_results = QAResults.calculate_metrics(predictions_parsed, ground_truths)

        evaluation_results.data = {
            "predictions": predictions,
            "predictions_parsed:": predictions_parsed,
            "ground_truths": ground_truths,
            "num_documents": len(evaluation_data),
        }

        return evaluation_results

    def _get_dataset_summary(
        self, evaluation_data: List[EvaluationDocument] | List[List[EvaluationDocument]]
    ) -> DatasetSummaryResults:
        if isinstance(evaluation_data[0], EvaluationDocument):
            return self._get_dataset_summary_single(evaluation_data)  # type: ignore
        else:
            # flatten evaluation data
            data: List[EvaluationDocument] = [
                doc for docs in evaluation_data for doc in docs
            ]  # type: ignore
            unique_doc_ids = {doc.doc_id for doc in data}
            unique_docs = []
            for doc in unique_doc_ids:
                unique_docs.append(next(d for d in data if d.doc_id == doc))
            summary_data = self._get_dataset_summary_single(unique_docs)
            return summary_data

    def _get_dataset_summary_single(
        self, evaluation_data: List[EvaluationDocument]
    ) -> DatasetSummaryResults:
        # get unique doc ids
        doc_ids = {doc.doc_id for doc in evaluation_data}

        document_list = IndexingService.load_and_preprocess_documents(
            preprocess_config=self.llm_config.preprocess_config,
            id_list=list(doc_ids),
        )

        return self._get_documents_summary(documents=document_list)

    def _get_documents_summary(
        self, documents: List[CustomDocument]
    ) -> DatasetSummaryResults:
        """
        Get a summary of the documents in the dataset.
        """

        document_summary = DatasetSummary.get_summary(documents)

        logging.info(
            f"Analysed documents for preprocessing mode: {self.llm_config.preprocess_config.name} and table serialization: {self.llm_config.preprocess_config.table_serialization}"
        )
        return DatasetSummaryResults(
            dataset_summary=document_summary,
            preprocess_mode=str(self.llm_config.preprocess_config.name),
            num_documents_per_iteration=self.evaluation_num_documents,
            iterations=self.evaluation_iterations,
            table_serialization_mode=str(
                self.llm_config.preprocess_config.table_serialization.table_serializer
            ),
        )

    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str = "o200k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    @staticmethod
    def generate_analysis_plot(
        token_counts: List[Statistics], preprocess_modes: List[str], file_path: str
    ):
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

        avg_token_counts = [entry.avg for entry in token_counts]
        std_token_counts = [entry.std for entry in token_counts]
        preprocess_modes = [entry for entry in preprocess_modes]

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
    def write_analysis_to_json(
        token_counts: List[DatasetSummaryResults], file_path: str
    ):
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

        # Write all results to disk
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(DatasetSummaryResults.list_to_json(token_counts))

    @staticmethod
    def write_evaluation_to_json(
        evaluation_results: List[EvaluationResults],
        file_path: str,
        names: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:

        modified_results = []
        for i, result in enumerate(evaluation_results):
            result_dict = {}
            if names:
                result_dict["name"] = names[i]
            exclude_list = ["dataset_summary"]
            if not result.qa_only_results:
                exclude_list.append("qa_only_results")
            if not result.qa_results:
                exclude_list.append("qa_results")
            if not result.ir_results:
                exclude_list.append("ir_results")

            result_dict.update(result.model_dump(exclude={"token_counts"}))
            modified_results.append(result_dict)

        data = {
            "evaluation_results": modified_results,
        }
        if metadata:
            data["metadata"] = metadata  # type: ignore

        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def save_evaluation_results(
        evaluation_results: List[EvaluationResults],
        metadata: Optional[dict] = None,
        names: Optional[List[str]] = None,
    ) -> None:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        
        file_path_base = "./data/evaluation/"

        file_path_json = f"./data/evaluation/evaluation_results-{timestamp}.json"

        file_path_dataset_summary = (
            f"./data/evaluation/dataset_analysis/analysis-{timestamp}.json"
        )

        file_path_dataset_summary_plot = (
            f"./data/evaluation/dataset_analysis/plots/{timestamp}.png"
        )
        
        if all(result.ir_results and not result.qa_results and not result.qa_only_results and not result.dataset_summary for result in evaluation_results):
            file_path_json = file_path_base + f"/ir/evaluation_results-{timestamp}.json"
        elif all(result.qa_results and not result.ir_results and not result.qa_only_results and not result.dataset_summary for result in evaluation_results):
            file_path_json = file_path_base + f"/qa/evaluation_results-{timestamp}.json"
        elif all(result.qa_only_results and not result.ir_results and not result.qa_results and not result.dataset_summary for result in evaluation_results):
            file_path_json = file_path_base + f"/qa_only/evaluation_results-{timestamp}.json"

        if any(
            result.ir_results or result.qa_results or result.qa_only_results
            for result in evaluation_results
        ):
            Evaluator.write_evaluation_to_json(
                evaluation_results=evaluation_results,
                file_path=file_path_json,
                metadata=metadata,
                names=names,
            )

        if not any(result.dataset_summary is None for result in evaluation_results):
            Evaluator.write_analysis_to_json(
                token_counts=[
                    result.dataset_summary
                    for result in evaluation_results
                    if result.dataset_summary
                ],  # type: ignore
                file_path=file_path_dataset_summary,
            )
            Evaluator.generate_analysis_plot(
                token_counts=[
                    result.dataset_summary.dataset_summary.token_statistics  # type: ignore
                    for result in evaluation_results
                ],
                preprocess_modes=[
                    f"Preprocess: {result.dataset_summary.preprocess_mode} \n Table Serialization: {result.dataset_summary.table_serialization_mode}"  # type: ignore
                    for result in evaluation_results
                ],
                file_path=file_path_dataset_summary_plot,
            )

    def check_for_indexing(self, evaluation_types: List[EvaluationType]) -> None:
        if (
            EvaluationType.EVALUATE_QA in evaluation_types
            or EvaluationType.EVALUATE_IR in evaluation_types
        ):
            if not isinstance(self.run_setup, RunSetupRAG):
                raise ValueError(
                    "Evaluator must be of type RunSetupRAG to run QA or IR evaluation"
                )
            self.run_setup.indexing_service.embed_documents(
                preprocess_config=self.llm_config.preprocess_config,
                overwrite_existing_collection=False,
            )

    @classmethod
    def get_evaluator(
        cls,
        evaluation_types: List[EvaluationType],
        preprocess_config: Optional[PreprocessConfig] = None,
        retriever_num_documents: Optional[int] = None,
    ) -> Evaluator:
        if (
            EvaluationType.EVALUATE_TABLE_QA_ONLY in evaluation_types
            or EvaluationType.RUN_DOCUMENT_ANALYSIS in evaluation_types
        ):
            config = LLMConfig.from_config(preprocess_config=preprocess_config)
            run_setup = RunSetupTableQA.run_setup()

            evaluator = cls.from_config(
                run_setup=run_setup,
                llm_config=config,
                evaluation_types=evaluation_types,
            )
        else:
            if not retriever_num_documents:
                retriever_num_documents = Config.run.retriever_num_documents[0]
            
            if not isinstance(retriever_num_documents, int):
                raise ValueError("Retriever num documents must be an integer")
            
            config = RAGConfig.from_config(preprocess_config=preprocess_config)
            config.retriever_num_documents = retriever_num_documents
            run_setup = RunSetupRAG.run_setup(config)
                        
            evaluator = cls.from_config(
                run_setup=run_setup,
                llm_config=config,
                retriever_num_documents=retriever_num_documents,
                evaluation_types=evaluation_types,
            )

        return evaluator

    @classmethod
    def run_single_evaluation(
        cls,
        save_results: bool = True,
        preprocess_config: Optional[PreprocessConfig] = None,
        retriever_num_documents: Optional[int] = None,
    ) -> Evaluator:
        evaluation_types = EvaluationType.get_evaluation_type_from_config()
        evaluator = cls.get_evaluator(
            evaluation_types=evaluation_types,
            preprocess_config=preprocess_config,
            retriever_num_documents=retriever_num_documents,
        )
        
        if preprocess_config:
            evaluation_names = cls.generate_evaluation_names(preprocess_configs=[preprocess_config])
        else:
            evaluation_names = [""]

        # Check if indexing is required
        evaluator.check_for_indexing(evaluation_types=evaluation_types)
        evaluator.evaluation_results = evaluator.evaluate()

        if evaluator.evaluation_results and save_results:
            Evaluator.save_evaluation_results(
                evaluation_results=[evaluator.evaluation_results],
                metadata={
                    "num_evaluation_documents": evaluator.evaluation_num_documents
                },
                names=evaluation_names,
            )
        return evaluator
    
    @staticmethod
    def generate_evaluation_names(preprocess_configs: List[PreprocessConfig]) -> List[str]: 
        evaluation_names = []
        for preprocess_config in preprocess_configs:
            if EvaluationType.EVALUATE_QA:
                if preprocess_config and preprocess_config.table_serialization and preprocess_config.table_serialization.include_related_tables:
                    evaluation_name = preprocess_config.name + "-RT-" + preprocess_config.table_serialization_related_tables.name # type: ignore
                else:
                    evaluation_name = preprocess_config.name + "-NO-RT"
            else:
                evaluation_name = preprocess_config.name
            evaluation_names.append(evaluation_name)
        return evaluation_names
            

    @classmethod
    def run_multi_evaluation(
        cls,
        preprocess_configs: List[PreprocessConfig],
    ) -> None:
        if (
            EvaluationType.EVALUATE_TABLE_QA_ONLY
            in EvaluationType.get_evaluation_type_from_config()
            or EvaluationType.RUN_DOCUMENT_ANALYSIS
            in EvaluationType.get_evaluation_type_from_config()
            or EvaluationType.EVALUATE_GET_HEADERS
            in EvaluationType.get_evaluation_type_from_config()
        ):
            cls.run_multi_evaluation_without_rag(preprocess_configs=preprocess_configs)
        else:
            retriever_num_documents = Config.run.retriever_num_documents
            if not retriever_num_documents:
                retriever_num_documents = [3]
            cls.run_multi_evaluation_rag(
                preprocess_configs=preprocess_configs,
                retriever_num_documents=retriever_num_documents,
            )

    @classmethod
    def run_multi_evaluation_without_rag(
        cls,
        preprocess_configs: List[PreprocessConfig],
    ) -> None:
        evaluation_results = []
        for preprocess_config in preprocess_configs:
            single_evaluation = cls.run_single_evaluation(
                preprocess_config=preprocess_config,
                save_results=False,
            )
            evaluation_results.append(single_evaluation.evaluation_results)

        Evaluator.save_evaluation_results(
            evaluation_results=evaluation_results,
            names=cls.generate_evaluation_names(preprocess_configs=preprocess_configs),
        )

    @classmethod
    def run_multi_evaluation_rag(
        cls,
        preprocess_configs: List[PreprocessConfig],
        retriever_num_documents: List[int],
    ) -> None:
        
        
        for retriever_num_documents_temp in retriever_num_documents:
            evaluation_results = []
            evaluation_names = cls.generate_evaluation_names(preprocess_configs=preprocess_configs)
            for preprocess_config in preprocess_configs:
                single_evaluation = cls.run_single_evaluation(
                    preprocess_config=preprocess_config,
                    save_results=False,
                    retriever_num_documents=retriever_num_documents_temp,
                )
                evaluation_results.append(single_evaluation.evaluation_results)

            Evaluator.save_evaluation_results(
                evaluation_results=evaluation_results,
                metadata={"retriever_num_documents": retriever_num_documents_temp},
                names=evaluation_names
            )

    def evaluate_table_header_detection(self) -> None:
        eval_data = self.evaluate_table_header_detection_prepare()
        self.evaluate_table_header_detection_single(eval_data=eval_data)
        
    def evaluate_table_header_detection_all(self) -> None:
        from ..evaluation.sec_filing_evaluator import SECFilingEvaluator
        from ..evaluation.wiki_tables_evaluator.wiki_table_questions_evaluator import (
            WikiTableQuestionsEvaluator,
        )
        
        sec_evaluator = SECFilingEvaluator(
            run_setup=self.run_setup,
            retriever_num_documents=self.retriever_num_documents,
            evaluation_types=self.evaluation_types,
            llm_config=self.llm_config,
            evaluation_num_documents=self.evaluation_num_documents,
            evaluation_iterations=self.evaluation_iterations,
        )
        wiki_evaluator = WikiTableQuestionsEvaluator(
            run_setup=self.run_setup,
            retriever_num_documents=self.retriever_num_documents,
            evaluation_types=self.evaluation_types,
            llm_config=self.llm_config,
            evaluation_num_documents=self.evaluation_num_documents,
            evaluation_iterations=self.evaluation_iterations,
        )
        
        
        sec_eval_data = sec_evaluator.evaluate_table_header_detection_prepare()
        wiki_eval_data = wiki_evaluator.evaluate_table_header_detection_prepare()
        # combine the two lists
        eval_data = sec_eval_data + wiki_eval_data
        self.evaluate_table_header_detection_single(eval_data=eval_data)
    
    @abstractmethod
    def evaluate_table_header_detection_prepare(self) -> List[HeaderEvaluationDocumentReduced]:
        pass
    

    
    
    def evaluate_table_header_detection_single(self, eval_data: List[HeaderEvaluationDocumentReduced]) -> None:        
        results = HeaderDetectionResults()
        for data in eval_data:
            table = data.html_table
            
            # document_loader = DocumentLoader.from_config()
            # max_column_header_row, max_row_label_column = document_loader.get_header_from_table_string(table_str=table)
            max_column_header_row = None
            max_row_label_column = None
            
            
            if max_column_header_row is None or max_row_label_column is None:
                # Find the corresponding document content by position
                html_table = HTMLTableParser.parse_and_clean_table(table)
                

                # Ensure that no header / label is set
                html_table.max_column_header_row = None
                html_table.max_row_label_column = None
                html_table_with_headers = TabTreeService.set_headers(html_table)

                # Ask LLM for Headers
                max_column_header_row, max_row_label_column = (
                    html_table_with_headers.max_column_header_row,
                    html_table_with_headers.max_row_label_column,
                )

            results.predictions_column_header_rows.append(
                list(range(max_column_header_row + 1))
            )
            results.predictions_row_label_columns.append(
                list(range(max_row_label_column + 1))
            )

            results.ground_truth_column_header_rows.append(data.column_header_rows)
            results.ground_truth_row_label_columns.append(data.row_label_columns)

        results.accuracy_rows = EvaluationResults.calculate_accuracy(
            predictions=results.predictions_column_header_rows,
            ground_truths=results.ground_truth_column_header_rows,
        )
        results.accuracy_columns = EvaluationResults.calculate_accuracy(
            predictions=results.predictions_row_label_columns,
            ground_truths=results.ground_truth_row_label_columns,
        )

        results.advanced_analysis = {
            "accuracy_column_header_rows": HeaderDetectionResults.calculate_advanced_metrics(
                results.predictions_column_header_rows,
                results.ground_truth_column_header_rows,
            )[
                0
            ],
            "f1_score_column_header_rows": HeaderDetectionResults.calculate_advanced_metrics(
                results.predictions_column_header_rows,
                results.ground_truth_column_header_rows,
            )[
                1
            ],
            "accuracy_row_label_columns": HeaderDetectionResults.calculate_advanced_metrics(
                results.predictions_row_label_columns,
                results.ground_truth_row_label_columns,
            )[
                0
            ],
            "f1_score_row_label_columns": HeaderDetectionResults.calculate_advanced_metrics(
                results.predictions_row_label_columns,
                results.ground_truth_row_label_columns,
            )[
                1
            ],
            "mean_accuracy_column_header_rows": np.mean(
                HeaderDetectionResults.calculate_advanced_metrics(
                    results.predictions_column_header_rows,
                    results.ground_truth_column_header_rows,
                )[0]
            ),
            "mean_accuracy_row_label_columns": np.mean(
                HeaderDetectionResults.calculate_advanced_metrics(
                    results.predictions_row_label_columns,
                    results.ground_truth_row_label_columns,
                )[0]
            ),
            "mean_f1_score_column_header_rows": np.mean(
                HeaderDetectionResults.calculate_advanced_metrics(
                    results.predictions_column_header_rows,
                    results.ground_truth_column_header_rows,
                )[1]
            ),
            "mean_f1_score_row_label_columns": np.mean(
                HeaderDetectionResults.calculate_advanced_metrics(
                    results.predictions_row_label_columns,
                    results.ground_truth_row_label_columns,
                )[1]
            ),
        }

        # plot and save histogram of row and column f1 scores
        plt.hist(
            results.advanced_analysis["f1_score_column_header_rows"], bins=3, alpha=0.5, label="Rows"
        )
        plt.hist(
            results.advanced_analysis,
            bins=3,
            alpha=0.5,
            label="Columns",
        )
        plt.legend(loc="upper right")
        plt.title("Histogram of row and column F1 scores")
        plt.xlabel("F1 Score")
        plt.xlim(0, 1)
        plt.ylabel("Frequency")
        # make y axis integer
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        # save data
        directory = os.path.dirname("./data/evaluation/header_detection/")
        os.makedirs(directory, exist_ok=True)

        file_path_histogram = os.path.join(
            directory, f"histogram_{time.strftime('%Y-%m-%d-%H-%M-%S')}.png"
        )
        file_path_json = os.path.join(
            directory,
            f"header_detection_results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.json",
        )

        # Save the plot
        plt.savefig(file_path_histogram)
        # Write all results to disk
        with open(file_path_json, "w", encoding="utf-8") as file:
            file.write(results.model_dump_json(indent=2))


    def store_sample_documents(
        self, valid_questions: List[List[EvaluationDocumentWithTable]], mapper_path: str
    ):
        from .wiki_tables_evaluator.wiki_table_questions_evaluator import (
            WikiTableQuestionsEvaluator,
        )
        
        # get question category by LLM
        table_serializer = HTMLSerializer()

        question_category_pipeline = QuestionCategoryPipeline.from_config()
        question_domain_pipeline = QuestionDomainPipeline.from_config()
        # read dict from json file
        try:
            with open(mapper_path, "r", encoding="utf-8") as file:
                id_mapper = json.load(file)
        except FileNotFoundError:
            id_mapper = []

        # flatten the list
        flattened_questions = [item for sublist in valid_questions for item in sublist]

        # id mapper is dict with [{"id": id, "question_id": question_id}, ...]
        # update id mapper
        for question in flattened_questions:
            if isinstance(self, WikiTableQuestionsEvaluator):
                dict_item = next(
                    (item for item in id_mapper if item["id"] == question.doc_id),
                    None,
                )
            else:
                dict_item = next(
                    (
                        item
                        for item in id_mapper
                        if item["question_id"] == question.question_id
                    ),
                    None,
                )
            if dict_item is None:
                if isinstance(self, WikiTableQuestionsEvaluator):
                    serialized_table = table_serializer.serialize_table_to_str(
                        question.html_table
                    )[0]
                else:
                    serialized_table = question.html_table

                if question.category and question.category == "not answerable":
                    question_category_short = "not answerable"
                else:
                    logging.info(
                        f"Domain & Category Detection: Question {question.doc_id} not found in id_mapper, asking LLM"
                    )

                    # get category by LLL
                    question_category = question_category_pipeline.predict_category(
                        question=question.question,
                        table=serialized_table if serialized_table else "",
                    )
                    question_category_short = self.parse_llm_response(
                        question_category
                    ).lower()

                # get domain by LLL
                if isinstance(self, WikiTableQuestionsEvaluator):
                    question_domain = question_domain_pipeline.predict_domain(
                        question=question.question,
                        table=serialized_table if serialized_table else "",
                    )

                    question_domain_short = self.parse_llm_response(
                        question_domain
                    ).lower()
                else:
                    question_domain_short = "business & economy"

                id_mapper.append(
                    {
                        "id": question.doc_id,
                        "question_id": question.question_id,
                        "category": question_category_short,
                        "domain": question_domain_short,
                        "table": serialized_table,
                    }
                )
            else:
                id_mapper.remove(dict_item)
                id_mapper.append(
                    {
                        "id": question.doc_id,
                        "question_id": question.question_id,
                        "category": dict_item["category"],
                        "domain": dict_item["domain"],
                        "table": question.html_table,
                    }
                )

        # write dict to json file
        with open(mapper_path, "w", encoding="utf-8") as file:
            json.dump(id_mapper, file, indent=4)

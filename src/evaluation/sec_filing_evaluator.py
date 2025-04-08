import logging
import os
import re
import time
from typing import List, Literal, override
import numpy as np
from pandas import DataFrame

from ..model.custom_document import SplitContent

from ..evaluation.evaluation_results import EvaluationResults, HeaderDetectionResults
from ..model.tabtree.tabtree_service import TabTreeService
from ..retrieval.document_preprocessors.table_parser.custom_html_parser import HTMLTableParser
from ..retrieval.indexing_service import IndexingService

from ..retrieval.document_loaders.sec_filing_loader import SECFilingLoader
from .evaluation_document import (
    EvaluationDocument,
    EvaluationDocumentWithTable,
    HeaderEvaluationDocument,
    HeaderEvaluationDocumentReduced,
)
from ..config.config import Config
from .evaluator import EvaluationType, Evaluator
import pandas as pd


class SECFilingEvaluator(Evaluator):
    
    @staticmethod
    def get_full_documents_by_evaluation_docs() -> List[str]:
        df = SECFilingEvaluator.get_evaluation_docs_by_path_name(
            Config.sec_filings.evaluation_data_path
        )
        return list(set(df["file"]))
        
        
        
    def get_evaluation_docs(self) -> List[EvaluationDocumentWithTable]:
        df = self.get_evaluation_docs_by_path_name(
            Config.sec_filings.evaluation_data_path
        )
        questions = [
            EvaluationDocument(
                doc_id=row["file"],
                question=row["question"],
                question_id=row["question"],
                answer=row["answer"],
                search_reference=row["search reference"],
                category=row["category"],
            )
            for _, row in df.iterrows()
        ]
        
        # Filter out questions that are not answerable for IR evaluation
        if EvaluationType.EVALUATE_IR in self.evaluation_types:
            questions = [
                question
                for question in questions
                if question.category and question.category.lower() != "not answerable"
            ]

        questions = SECFilingLoader.add_tables_to_questions(questions)
        self.store_sample_documents([questions])

        # if self.evaluation_num_documents:
        #    questions = questions[: self.evaluation_num_documents]

        return questions

    def get_evaluation_docs_list(self) -> List[List[EvaluationDocument]]:
        raise ValueError("For SEC Filings, only single evaluation is supported")

    @staticmethod
    def get_evaluation_docs_by_path_name(
        file_path: str = Config.sec_filings.evaluation_data_path,
    ) -> DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        df = pd.read_csv(file_path, sep=";", dtype=str, keep_default_na=False)
        return df

    @override
    def store_sample_documents(
        self,
        questions: List[List[EvaluationDocumentWithTable]],
        mapper_path: str = "./data/sec_filings/0_id_mapper_questions.json",
    ):
        return super().store_sample_documents(questions, mapper_path)

    def get_tabtree_header_evaluation_data(self) -> List[HeaderEvaluationDocument]:
        file_path: str = Config.sec_filings.evaluation_get_header_data_path
        eval_data = HeaderEvaluationDocument.from_sec_filing_csv(file_path=file_path)
        return eval_data

    @staticmethod
    def search_table_by_regexp(string_to_search: str, chunk: str, chunk_table_string: str, mode: Literal["detailed", "normal"] = "normal") -> bool: 
        
        if re.search(string_to_search, chunk_table_string, flags=re.DOTALL):
            # string to search is always like the following: 'United Kingdom.*?4,?215.*?6,?522' 
            # or 'Provision for \(benefit from\) income taxes.*?213'
            # therefore we split the string_to_search by '.*?' and check if at least the last part is in the chunk 
            last_part = string_to_search.split('.*?')[-1]
            if re.search(last_part, chunk, flags=re.DOTALL):
                return True
            else:
                logging.info(f"Attention: Original was found, but last item of search string not!")
                logging.info(f"Original: {string_to_search}")
                logging.info(f"Chunk: {chunk}")
                logging.info(f"Table: {chunk_table_string}")
                if mode == "detailed":
                    return False
                else:
                    logging.info(f"BUT: Returning True for normal mode")
                    return True
        return False


    def evaluate_table_header_detection_prepare(self) -> List[HeaderEvaluationDocumentReduced]:
        eval_data = self.get_tabtree_header_evaluation_data()

        # Ensure that col and rowspans are not deleted by resetting the table serializer
        self.llm_config.preprocess_config.table_serialization.table_serializer = "none"
        self.llm_config.preprocess_config.consider_colspans_rowspans = True
        
        document = IndexingService.load_and_preprocess_document(
            preprocess_config=self.llm_config.preprocess_config,
            dataset="sec-filings"
        )

        tables = []
        if not document.splitted_content:
            raise ValueError("Document has no splitted content")

        for data in eval_data:
            # Find the corresponding document content by position
            
            def search_str_in_regexp(
                content: str,
                search_str: str,
            ) -> bool:
                if re.search(pattern=search_str.strip(), string=content, flags=re.DOTALL):
                    return True
                return False

            
            # search table by regexp
            html_table = [
                content.content
                for content in document.splitted_content
                if search_str_in_regexp(content=content.content, search_str=data.search_regexp)
                and content.type == "table"
            ]
            if len(html_table) == 0:
                raise ValueError(
                    f"Table with search string {data.search_regexp} not found in document {document.metadata.doc_id}"
                )                
            tables.append(html_table[0])
            
        return [HeaderEvaluationDocumentReduced(
            html_table=table,
            row_label_columns=data.row_label_columns,
            column_header_rows=data.column_header_rows,
        ) for table, data in zip(tables, eval_data)]

        

            
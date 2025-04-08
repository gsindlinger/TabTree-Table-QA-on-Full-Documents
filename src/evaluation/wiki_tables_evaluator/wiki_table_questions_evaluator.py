from __future__ import annotations
import json
import logging
import os


import os
import random
import time
from typing import List, override


from ...retrieval.indexing_service import IndexingService
from ...retrieval.document_preprocessors.table_parser.custom_html_parser import (
    HTMLTableParser,
)
from ...config.config import Config
from ..evaluation_document import EvaluationDocumentWithTable, HeaderEvaluationDocument, HeaderEvaluationDocumentReduced
from ...retrieval.document_loaders.wiki_table_questions_loader import (
    WikiTableQuestionsLoader,
)

from ..evaluator import Evaluator


class WikiTableQuestionsEvaluator(Evaluator):

    def get_evaluation_docs_list(self) -> List[List[EvaluationDocumentWithTable]]:
        valid_questions = self.get_valid_questions_from_list(shuffle=True)
        return valid_questions

    def get_evaluation_docs(self) -> List[EvaluationDocumentWithTable]:
        if Config.wiki_table_questions.is_single_evaluation:
            while True:
                questions = WikiTableQuestionsLoader.load_unique_questions(
                    Config.wiki_table_questions.single_document_id
                )
                questions = WikiTableQuestionsLoader.add_tables_to_questions(questions)
                if all(
                    HTMLTableParser.is_valid_table(question.html_table)
                    for question in questions
                ):
                    return questions
        else:
            valid_question_list = self.get_valid_questions_from_list()
            if len(valid_question_list) > 1:
                raise ValueError(
                    "Evaluation iterations are set to more than 1, but the method for a single iteration was called. Please call get_evaluation_docs_list instead."
                )
            return valid_question_list[0]

    def get_valid_questions_from_list(
        self, shuffle: bool = False, seed: int = 42
    ) -> List[List[EvaluationDocumentWithTable]]:
        if self.evaluation_num_documents is None:
            raise ValueError("Evaluation num documents must be set")
        questions = WikiTableQuestionsLoader.load_questions()

        if shuffle:
            random.seed(seed)
            random.shuffle(questions)

        valid_questions: List[List[EvaluationDocumentWithTable]] = []
        valid_questions_iterations: List[EvaluationDocumentWithTable] = []
        doc_ids_all = set()
        i = 0
        for question in questions:
            # make sure that the questions from different documents are retrieved
            if question.doc_id in doc_ids_all:
                continue
            question = WikiTableQuestionsLoader.add_table_to_question(question)
            if HTMLTableParser.is_valid_table(question.html_table):
                valid_questions_iterations.append(question)
                i += 1
                doc_ids_all.add(question.doc_id)
            if i > 0 and i % self.evaluation_num_documents == 0:
                valid_questions.append(valid_questions_iterations)
                valid_questions_iterations = []
            if i == self.evaluation_num_documents * self.evaluation_iterations:
                break
        self.store_sample_documents(valid_questions)
        
        # write flattened questions to file
        flattened_questions = [
            {
            "doc_id": question.doc_id, 
            "question_id": question.question_id,
            "question": question.question
            } for questions in valid_questions for question in questions]
        
        with open(f"./data/wiki_table_questions/evaluation_ids_{len(flattened_questions)}.json", "w") as f:
            json.dump(flattened_questions, f, indent=4)
            
            
        return valid_questions

    @override
    def store_sample_documents(
        self,
        questions: List[List[EvaluationDocumentWithTable]],
        mapper_path: str = "./data/wiki_table_questions/id_mapper_questions.json",
    ):
        return super().store_sample_documents(questions, mapper_path)

    def get_tabtree_header_evaluation_data(self) -> List[HeaderEvaluationDocument]:
        file_path: str = Config.wiki_table_questions.evaluation_get_header_data_path
        eval_data = HeaderEvaluationDocument.from_wiki_table_questions_csv(file_path=file_path)
        return eval_data
    
    
    def evaluate_table_header_detection_prepare(self) -> List[HeaderEvaluationDocumentReduced]:

        
        eval_data = self.get_tabtree_header_evaluation_data()

        # Ensure that col and rowspans are not deleted by resetting the table serializer
        self.llm_config.preprocess_config.table_serialization.table_serializer = "none"
        self.llm_config.preprocess_config.consider_colspans_rowspans = True
        
        tables = IndexingService.load_and_preprocess_documents(
            preprocess_config=self.llm_config.preprocess_config, 
            id_list=[doc.doc_id for doc in eval_data],
            dataset="wiki-table-questions"
        )
        
        results = [HeaderEvaluationDocumentReduced(
            html_table=table.page_content,
            row_label_columns=data.row_label_columns,
            column_header_rows=data.column_header_rows,
        ) for data, table in zip(eval_data, tables)]

        return results

    
    
import json
import logging
import os
from typing import ClassVar, List, Optional, Tuple

import pandas as pd
import requests

from ...evaluation.evaluation_document import (
    EvaluationDocument,
    EvaluationDocumentWithTable,
)
from ...config.config import Config
from .document_loader import DocumentLoader
from ...model.custom_document import CustomDocument, FullMetadata


class WikiTableQuestionsLoader(DocumentLoader):
    page_base_url: ClassVar[str] = (
        "https://raw.githubusercontent.com/ppasupat/WikiTableQuestions/master/page/"
    )
    csv_base_url: ClassVar[str] = (
        "https://raw.githubusercontent.com/ppasupat/WikiTableQuestions/master/csv/"
    )
    training_data_url: ClassVar[str] = (
        "https://raw.githubusercontent.com/ppasupat/WikiTableQuestions/master/data/training.tsv"
    )

    def load_single_document(self, id: Optional[str] = None) -> CustomDocument:
        """Loads a single document (in this case from local storage) by its id and calls preprocessing method."""

        if id is None:
            id = Config.wiki_table_questions.single_document_id
        if not isinstance(id, str):
            raise ValueError(f"Invalid ID format. Must be string")
        return self.load_single_document_with_id(id=id)[0]

    def load_single_document_with_id(
        self, id: str, load_table_only: bool = True
    ) -> Tuple[CustomDocument, str, Tuple[int | None, int | None], str | None]:
        """Loads a single document (in this case from local storage) by its id and calls preprocessing method.
        Args:
            id: The ID of the document to load
            load_table_only: If True, only the table will be loaded to html of custom document
        Returns:
            Full custom document, HTML of the specific table, max_column_header_row, max_row_label_column and the tables title
        """
        if id and "-" in id:
            splitted_id = id.split("-")
        else:
            raise ValueError(f"Invalid ID format. Expected format: xxx-yyy, got {id}")

        folder_path = self.get_folder_path_by_id(id)
        os.makedirs(folder_path, exist_ok=True)

        # Create original download links
        page_url = f"{self.page_base_url}{splitted_id[0]}-page/{splitted_id[1]}.html"
        table_url = f"{self.csv_base_url}{splitted_id[0]}-csv/{splitted_id[1]}.html"
        metadata_url = (
            f"{self.page_base_url}{splitted_id[0]}-page/{splitted_id[1]}.json"
        )

        # Download files
        self.load_unique_questions(id=id)
        self.check_download_write_file(
            url=page_url, file_path=f"{folder_path}/{id}.html"
        )
        self.check_download_write_file(
            url=table_url, file_path=f"{folder_path}/{id}-table.html"
        )
        self.check_download_write_file(
            url=metadata_url, file_path=f"{folder_path}/{id}.json"
        )

        # Load data from disk
        with open(f"{folder_path}/{id}.html", "r", encoding="utf-8") as f:
            html = f.read()
        with open(f"{folder_path}/{id}.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        with open(f"{folder_path}/{id}-table.html", "r", encoding="utf-8") as f:
            html_table = f.read()

        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid metadata format for {id}")
        max_row_label_column = metadata.get("max_row_label_column")
        max_column_header_row = metadata.get("max_column_header_row")
        table_title = metadata.get("title")
        if not max_row_label_column is None:
            max_row_label_column = int(max_row_label_column)
        if not max_column_header_row is None:
            max_column_header_row = int(max_column_header_row)

        if load_table_only:
            html = html_table
        return (
            CustomDocument(
                page_content=html,
                metadata=FullMetadata(
                    doc_id=id,
                    additional_metadata=metadata,
                ),
            ),
            html_table,
            (max_column_header_row, max_row_label_column),
            table_title,
        )

    # @staticmethod
    # def write_header_back_to_file(
    #     max_column_header_row: int,
    #     max_row_label_column: int,
    #     doc_id: str,
    # ) -> None:
    #     folder_path = WikiTableQuestionsLoader.get_folder_path_by_id(doc_id)
    #     with open(f"{folder_path}/{doc_id}.json", "r", encoding="utf-8") as f:
    #         metadata = json.load(f)

    #     metadata["max_column_header_row"] = max_column_header_row
    #     metadata["max_row_label_column"] = max_row_label_column

    #     with open(f"{folder_path}/{doc_id}.json", "w", encoding="utf-8") as f:
    #         json.dump(metadata, f, indent=4)

    @staticmethod
    def add_tables_to_questions(
        questions: List[EvaluationDocument],
    ) -> List[EvaluationDocumentWithTable]:
        """Load tables for each question by appending the html table to the question object"""
        questions_with_tables = []
        for question in questions:
            questions_with_tables.append(
                WikiTableQuestionsLoader.add_table_to_question(question)
            )
        return questions_with_tables

    @staticmethod
    def add_table_to_question(
        question: EvaluationDocument,
    ) -> EvaluationDocumentWithTable:
        """Load table for a single question by appending the html table to the question object"""
        table = WikiTableQuestionsLoader().load_single_document_with_id(question.doc_id)
        logging.info(f"Loaded table for {question.doc_id}")
        question_with_table = EvaluationDocumentWithTable(
            doc_id=question.doc_id,
            question_id=question.question_id,
            question=question.question,
            answer=question.answer,
            search_reference=question.search_reference,
            html_table=table[1],
            max_column_header_row=table[2][0],
            max_row_label_column=table[2][1],
            table_title=table[3],
        )
        return question_with_table

    def load_documents(
        self,
        num_of_documents: Optional[int] = None,
    ) -> List[CustomDocument]:
        training_data = self.load_questions()

        unique_ids = list({doc.doc_id for doc in training_data})

        if num_of_documents:
            unique_ids = unique_ids[:num_of_documents]

        return [self.load_single_document_with_id(id)[0] for id in unique_ids]

    @staticmethod
    def get_folder_path_by_id(id: str) -> str:
        return f"{Config.wiki_table_questions.data_path}{id}"

    @staticmethod
    def get_all_evaluation_docs_path() -> str:
        return f"{Config.wiki_table_questions.data_path}all_training_questions.csv"

    @staticmethod
    def get_single_evaluation_doc_path(id: str) -> str:
        return f"{Config.wiki_table_questions.data_path}{id}/questions.csv"

    @staticmethod
    def get_id(context: str) -> str:
        # csv/204-csv/590.csv -> 204-590
        splitted_context = context.split("/")
        return f"{splitted_context[1].split('-csv')[0]}-{splitted_context[-1].replace('.csv', '')}"

    @staticmethod
    def load_questions() -> List[EvaluationDocument]:
        questions_path = WikiTableQuestionsLoader.get_all_evaluation_docs_path()
        if not os.path.exists(questions_path):
            os.makedirs(os.path.dirname(questions_path), exist_ok=True)
            questions = pd.read_csv(
                WikiTableQuestionsLoader.training_data_url,
                sep="\t",
                quotechar='"',
            )
            questions.to_csv(path_or_buf=questions_path, sep=";", index=False)

        questions = pd.read_csv(filepath_or_buffer=questions_path, sep=";")
        questions.fillna("", inplace=True)

        return [
            EvaluationDocument(
                doc_id=WikiTableQuestionsLoader.get_id(context=row["context"]),
                question_id=row["id"],
                question=row["utterance"],
                answer=row["targetValue"],
                search_reference=row["targetValue"],
            )
            for _, row in questions.iterrows()
        ]

    @staticmethod
    def load_unique_questions(id: str) -> List[EvaluationDocument]:
        file_path = (
            f"{WikiTableQuestionsLoader.get_folder_path_by_id(id)}/questions.csv"
        )
        if (not os.path.exists(file_path)) or os.path.getsize(file_path) < 20:
            all_questions = WikiTableQuestionsLoader.load_questions()
            filtered_questions = EvaluationDocument.filter_documents_by_id(
                documents=all_questions,
                id=id,
            )
            EvaluationDocument.to_csv(filtered_questions, file_path)

        return EvaluationDocument.from_csv(file_path)

    @staticmethod
    def id_to_initial_context(id: str) -> str:
        # 204-590 -> csv/204-csv/590.csv
        splitted_id = id.split("-")
        return f"csv/{splitted_id[0]}-csv/{splitted_id[1]}.csv"

    @staticmethod
    def check_download_write_file(url, file_path):
        if not os.path.exists(file_path):
            WikiTableQuestionsLoader.download_and_write_file(
                url=url, file_path=file_path
            )

    @staticmethod
    def download_and_write_file(url: str, file_path: str) -> None:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                WikiTableQuestionsLoader.write_file_to_disk(file_path, response)
            else:
                print(
                    f"Failed to download from {url}, status code: {response.status_code}"
                )
        except requests.exceptions.Timeout:
            print(f"Request timed out for {url}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

    @staticmethod
    def write_file_to_disk(file_path: str, response: requests.Response) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)

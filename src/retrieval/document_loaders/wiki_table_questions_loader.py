import json
import logging
import os
from typing import ClassVar, List, Optional

import pandas as pd
import requests

from ...evaluation.evaluation_document import EvaluationDocument
from ...config.config import Config
from .document_loader import DocumentLoader
from ...model.custom_document import CustomDocument, FullMetadata


class WikiTableQuestionsLoader(DocumentLoader):
    page_base_url: ClassVar[str] = (
        "https://raw.githubusercontent.com/ppasupat/WikiTableQuestions/master/page/"
    )
    training_data_url: ClassVar[str] = (
        "https://raw.githubusercontent.com/ppasupat/WikiTableQuestions/master/data/training.tsv"
    )

    def load_single_document(self) -> CustomDocument:
        id = Config.wiki_table_questions.single_document_id
        return self.load_single_document_with_id(id=id)

    def load_single_document_with_id(
        self,
        id: str,
    ) -> CustomDocument:
        if id and "-" in id:
            splitted_id = id.split("-")
        else:
            raise ValueError(f"Invalid ID format. Expected format: xxx-yyy, got {id}")

        folder_path = self.get_folder_path_by_id(id)
        os.makedirs(folder_path, exist_ok=True)

        # Create original download links
        page_url = f"{self.page_base_url}{splitted_id[0]}-page/{splitted_id[1]}.html"
        metadata_url = (
            f"{self.page_base_url}{splitted_id[0]}-page/{splitted_id[1]}.json"
        )

        # Download files
        self.load_unique_questions(id=id)
        self.check_download_write_file(
            url=page_url, file_path=f"{folder_path}/{id}.html"
        )
        self.check_download_write_file(
            url=metadata_url, file_path=f"{folder_path}/{id}.json"
        )

        # Load data from disk
        with open(f"{folder_path}/{id}.html", "r", encoding="utf-8") as f:
            html = f.read()
        with open(f"{folder_path}/{id}.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return CustomDocument(
            page_content=html,
            metadata=FullMetadata(
                doc_id=id,
                additional_metadata=metadata,
            ),
        )

    def load_documents(
        self,
        num_of_documents: Optional[int] = None,
    ) -> List[CustomDocument]:
        training_data = self.load_questions()

        unique_ids = list({doc.doc_id for doc in training_data})

        if num_of_documents:
            unique_ids = unique_ids[:num_of_documents]

        return [self.load_single_document_with_id(id) for id in unique_ids]

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
                WikiTableQuestionsLoader.training_data_url, sep="\t"
            )
            questions.to_csv(path_or_buf=questions_path, sep=";", index=False)

        questions = pd.read_csv(filepath_or_buffer=questions_path, sep=";")
        questions.fillna("", inplace=True)

        return [
            EvaluationDocument(
                doc_id=WikiTableQuestionsLoader.get_id(context=row["context"]),
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
        else:
            logging.info(f"File {file_path} already exists. Skipping download.")

    @staticmethod
    def download_and_write_file(url: str, file_path: str) -> None:
        response = requests.get(url)
        if response.status_code == 200:
            WikiTableQuestionsLoader.write_file_to_disk(file_path, response)
        else:
            print(f"Failed to download from {url}")

    @staticmethod
    def write_file_to_disk(file_path: str, response: requests.Response) -> None:
        with open(file_path, "wb", encoding="utf-8") as f:
            f.write(response.content)

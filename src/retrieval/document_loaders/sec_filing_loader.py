import glob
import logging
import os
import re
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup
from pydantic import Field

from ...retrieval.document_preprocessors.preprocess_config import PreprocessConfig
from ...retrieval.indexing_service import IndexingService

from ...evaluation.evaluation_document import (
    EvaluationDocument,
    EvaluationDocumentWithTable,
)

from .document_loader import DocumentLoader
from ...config.config import Config
from ...model.custom_document import CustomDocument, FullMetadata


class SECFilingLoader(DocumentLoader):
    folder_path: str = Field(default_factory=lambda: Config.sec_filings.data_path)
    file_encoding: str = Field(default="ascii")

    def load_single_document(self, id: Optional[str] = None) -> CustomDocument:
        if not id:
            return self.load_single_document_with_file_path(
                file_path=Config.sec_filings.data_path_single
            )
        else:
            return self.load_single_document_with_id(id=id)

    def load_single_document_with_id(
        self,
        id: str,
    ) -> CustomDocument:
        """Loads a single document (in this case from local storage) by its id and calls preprocessing method."""
        file_path = self.folder_path + id
        return self.load_single_document_with_file_path(file_path=file_path)

    def load_single_document_with_file_path(
        self,
        file_path: Optional[str] = None,
    ) -> CustomDocument:
        """Loads a single document (in this case from local storage) and calls preprocessing method."""

        if not file_path:
            file_path = glob.glob(self.folder_path + "*")[0]

        with open(file_path, "r", encoding=self.file_encoding) as file:
            html_content = file.read()
            filing = CustomDocument(
                page_content=html_content,
                metadata=FullMetadata(doc_id=os.path.basename(file_path)),
            )
            if filing.metadata is None:
                raise ValueError("Metadata is None.")

            logging.info(f"Loaded document {filing.metadata.doc_id}")
            return filing

    @staticmethod
    def add_tables_to_questions(
        questions: List[EvaluationDocument],
    ) -> List[EvaluationDocumentWithTable]:
        ids = {question.doc_id for question in questions}
        preprocess_config = PreprocessConfig.from_config()
        documents = IndexingService.load_and_preprocess_documents(
            preprocess_config=preprocess_config,
            id_list=list(ids),
        )

        questions_with_tables = [
            SECFilingLoader.add_table_to_question(question, documents)
            for question in questions
        ]

        return questions_with_tables

    @staticmethod
    def add_table_to_question(
        question: EvaluationDocument,
        documents: List[CustomDocument],
    ) -> EvaluationDocumentWithTable:
        """Extracts the table from the HTML content of a document and adds it to the question."""
        document = next(doc for doc in documents if doc.metadata.doc_id == question.doc_id)  # type: ignore

        if not document.splitted_content:
            raise ValueError(
                f"No splitted content in document {document.metadata.doc_id}"
            )

        tables = [
            split_content.original_content
            for split_content in document.splitted_content
            if split_content.type == "table"
        ]

        def find_table_by_regex_in_list_of_tables(tables: List[str], regex: str) -> str:
            for table in tables:
                raw_string = r"{}".format(regex)
                if re.search(raw_string, table, re.DOTALL):
                    return table
            return ""

        if any(table is None for table in tables):
            raise ValueError(
                "Trying to add tables to questions, but some tables are None."
            )

        table = find_table_by_regex_in_list_of_tables(tables, question.search_reference)  # type: ignore

        if table == "":
            logging.warning(
                f"Table not found for question {question.doc_id} and search reference {question.search_reference}"
            )
        return EvaluationDocumentWithTable(
            doc_id=question.doc_id,
            question_id=question.question_id,
            question=question.question,
            answer=question.answer,
            search_reference=question.search_reference,
            html_table=table.replace("\n", ""),
            category=question.category,
        )

    def load_documents(
        self,
        num_of_documents: Optional[int] = None,
    ) -> List[CustomDocument]:

        documents = []
        count = 0
        for file_path in glob.glob(self.folder_path + "*"):
            if num_of_documents and count == num_of_documents:
                break
            with open(file_path, "r", encoding="utf-8") as file:
                html_content = file.read()
                filing = CustomDocument(
                    page_content=html_content,
                    metadata=FullMetadata(doc_id=os.path.basename(file_path)),
                )

            # Append to documents
            documents.append(filing)
            count += 1
        return documents

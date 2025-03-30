from __future__ import annotations
from abc import ABC, abstractmethod
import hashlib
import json
import logging
from typing import List, Literal, Optional

from pydantic import BaseModel

from ...retrieval.document_preprocessors.table_serializer import HTMLSerializer
from ...config.config import Config
from ...model.custom_document import CustomDocument


class DocumentLoader(ABC, BaseModel):

    @classmethod
    def from_config(cls, dataset: Optional[Literal["wiki-table-questions", "sec-filings"]] = None) -> DocumentLoader:
        mode = dataset if dataset else Config.run.dataset
        match mode:
            case "wiki-table-questions":
                from .wiki_table_questions_loader import WikiTableQuestionsLoader

                return WikiTableQuestionsLoader()
            case "sec-filings":
                from .sec_filing_loader import SECFilingLoader

                return SECFilingLoader()
            case _:
                raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def string_to_hash(table_string: str) -> str:
        
        # ensure that always same hash is generated for same table, therefore preprocess it
        serialized_table_str = HTMLSerializer().serialize_table_to_str(table_string)[0]
        if not serialized_table_str:
            serialized_table_str = table_string
        
        return hashlib.sha256(table_string.encode()).hexdigest()
    
    def get_table_summary(self, table: str, preceding_sentence: str) -> str:
        from ...pipeline import TableSummaryPipeline
        
        table_str = self.string_to_hash(table)
        table_header_dict_path = Config.indexing.table_header_path

        # check if file exists and load it, if not create file
        try:
            with open(table_header_dict_path, "r") as file:
                table_header_dict = json.load(file)
        except FileNotFoundError:
            logging.info("Table header file not found. Creating new file.")
            table_header_dict = {}

        if table_str in table_header_dict and "summary" in table_header_dict[table_str]:
            logging.info(f"Loading headers from file for table hash: {table_str}")
            return table_header_dict[table_str]["summary"]
        else:
            table_summary_pipeline = TableSummaryPipeline.from_config()
            table_summary = table_summary_pipeline.predict_table_summary(table=table, preceding_sentence=preceding_sentence)
            
            # parse by retrieving everything after "Table Summary: "
            table_summary = table_summary.split("Table Summary: ")[-1]
            print(f"Table Summary: {table_summary}")
            logging.info(f"Table Summary: {table_summary}")

            
            # update dict with table summary
            table_header_dict[table_str]['summary'] = table_summary
            
            # write back to file
            with open(table_header_dict_path, "w") as file:
                json.dump(table_header_dict, file, indent=4)
                
            return table_summary

        
    def get_header_from_table_string(
        self, table_str: str
    ) -> tuple[int | None, int | None]:
        
        
        table_str = self.string_to_hash(table_str)
        table_header_dict_path = Config.indexing.table_header_path

        # check if file exists and load it, if not create file
        try:
            with open(table_header_dict_path, "r") as file:
                table_header_dict = json.load(file)
        except FileNotFoundError:
            logging.info("Table header file not found. Creating new file.")
            table_header_dict = {}
            

        if table_str in table_header_dict:
            logging.info(f"Loading headers from file for table hash: {table_str}")

            return (
                table_header_dict[table_str].get("max_column_header_row_new", None),
                table_header_dict[table_str].get("max_row_label_column_new", None),
            )
        else:
            return None, None

    def write_header_back_to_file(
        self,
        table_str: str,
        max_column_header_row: int | None,
        max_row_label_column: int | None,
    ) -> None:
        # read json
        table_header_dict_path = Config.indexing.table_header_path

        # check if file exists and load it, if not create file
        try:
            with open(table_header_dict_path, "r") as file:
                table_header_dict = json.load(file)
        except FileNotFoundError:
            logging.info("Table header file not found. Creating new file.")
            table_header_dict = {}

        table_hash = self.string_to_hash(table_str)

        # update json
        if table_header_dict.get(table_hash) is None:
            table_header_dict[table_hash] = {
                "max_column_header_row": max_column_header_row,
                "max_row_label_column": max_row_label_column,
                "table_str_start": f"{table_str[:300]}...",
            }
        else:
            table_header_dict[table_hash]["max_column_header_row"] = max_column_header_row
            table_header_dict[table_hash]["max_row_label_column"] = max_row_label_column

        

        # write json pretty
        with open(table_header_dict_path, "w") as file:
            json.dump(table_header_dict, file, indent=4)

    @abstractmethod
    def load_documents(
        self, num_of_documents: Optional[int] = None
    ) -> List[CustomDocument]:
        pass

    @abstractmethod
    def load_single_document(self, id: Optional[str] = None) -> CustomDocument:
        pass

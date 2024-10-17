from __future__ import annotations
from abc import ABC, abstractmethod
from io import StringIO
from typing import List
from pydantic import BaseModel
import pandas as pd

from ...model.custom_document import CustomDocument
from .preprocess_config import PreprocessConfig


class TableSerializer(ABC, BaseModel):

    @classmethod
    def from_preprocess_config(cls, config: PreprocessConfig) -> TableSerializer | None:
        if config.table_serialization == "html":
            return HTMLSerializer()
        elif config.table_serialization == "csv":
            return CSVSerializer()
        elif config.table_serialization == "tsv":
            return TSVSerializer()
        elif config.table_serialization == "df-loader":
            return DFLoaderSerializer()
        elif config.table_serialization == "json-records":
            return JSONSerializerRecords()
        elif config.table_serialization == "json-split":
            return JSONSerializerSplit()
        elif config.table_serialization == "json-table-index":
            return JSONSerializerTableIndex()
        elif config.table_serialization == "markdown":
            return MarkdownSerializer()
        elif config.table_serialization == "text":
            return TextSerializer()
        elif config.table_serialization == "text-bullet-points":
            return TextSerializerBulletPoints()
        elif config.table_serialization == "list-item":
            return ListItemSerializer()
        elif config.table_serialization == "none":
            return None
        else:
            raise ValueError(
                f"Table serialization type {config.table_serialization} is not supported"
            )

    def serialize_table(self, table: str) -> str:
        df_table = self.load_table_to_df(table)
        return self.df_to_serialized_string(df_table)

    @abstractmethod
    def df_to_serialized_string(self, df: pd.DataFrame) -> str:
        pass

    def load_table_to_df(self, table: str) -> pd.DataFrame:
        tables = pd.read_html(StringIO(table))
        if not tables:
            raise ValueError("Could not load table to DataFrame")
        elif len(tables) > 1:
            raise ValueError("More than one table found in the table string")
        return tables[0]

    def serialize_tables_in_document(self, document: CustomDocument) -> str:
        serialized_docs = []
        if not document.splitted_content:
            raise ValueError("Document does not have any splitted content")

        for content in document.splitted_content:
            if content.type == "table":
                new_content = self.serialize_table(content.content)
            else:
                new_content = content.content
            serialized_docs.append(new_content)

        return " ".join(serialized_docs)


class DFLoaderSerializer(TableSerializer):
    def df_to_serialized_string(self, df: pd.DataFrame) -> str:
        return df.to_string(index=False)


class HTMLSerializer(TableSerializer):
    def df_to_serialized_string(self, df: pd.DataFrame) -> str:
        return df.to_html(header=True, index=False)


class CSVSerializer(TableSerializer):
    def df_to_serialized_string(self, df: pd.DataFrame) -> str:
        return df.to_csv(header=True, index=False, sep=";")


class TSVSerializer(TableSerializer):
    def df_to_serialized_string(self, df: pd.DataFrame) -> str:
        return df.to_csv(header=True, index=False, sep="\t")


class JSONSerializerRecords(TableSerializer):
    def df_to_serialized_string(self, df: pd.DataFrame) -> str:
        return df.to_json(orient="records")


class JSONSerializerSplit(TableSerializer):
    def df_to_serialized_string(self, df: pd.DataFrame) -> str:
        return df.to_json(orient="split")


class JSONSerializerTableIndex(TableSerializer):
    def df_to_serialized_string(self, df: pd.DataFrame) -> str:
        return df.to_json(orient="index")


class MarkdownSerializer(TableSerializer):
    def df_to_serialized_string(self, df: pd.DataFrame) -> str:
        return df.to_markdown(index=False)


class TextSerializer(TableSerializer):
    def df_to_serialized_string(self, df: pd.DataFrame) -> str:
        text_template = []
        for _, row in df.iterrows():
            row_text = [f"The {col} is {row[col]}." for col in df.columns]
            text_template.append(" ".join(row_text))
        return "\n".join(text_template)


class TextSerializerBulletPoints(TableSerializer):
    def df_to_serialized_string(self, df: pd.DataFrame) -> str:
        bullet_point_text = []
        for _, row in df.iterrows():
            row_text = [f"{col} is {row[col]}" for col in df.columns]
            bullet_point_text.append("- " + ". ".join(row_text) + ".")
        return "\n".join(bullet_point_text)


class ListItemSerializer(TableSerializer):
    def df_to_serialized_string(self, df: pd.DataFrame) -> str:
        text_template = []
        for _, row in df.iterrows():
            row_text = [f"The {col} is {row[col]}." for col in df.columns]
            text_template.append(" ".join(row_text))
        return "\n".join(text_template)

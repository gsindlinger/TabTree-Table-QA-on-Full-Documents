from __future__ import annotations
from abc import ABC, abstractmethod
from io import StringIO
import json
import logging
from typing import List, Optional, Tuple, override
from bs4 import BeautifulSoup
from pydantic import BaseModel
import pandas as pd

from ...retrieval.document_preprocessors.table_parser.custom_html_parser import HTMLTableParser

from ...model.tables import ExtendedTable, SerializedTable
from ...model.custom_document import CustomDocument, SplitContent
from .preprocess_config import PreprocessConfig, TableSerializerPreprocessConfig

END_OF_TABLE: str = r"<EOT>"
BEGINNING_OF_TABLE: str = r"<BOT>"
# since EOF Tags are always showing up behind a table,
# the split will be empty if there are still html table tags,
# which will be deleted anyway (see HTMLPreprocessor for details)
END_OF_TABLE_REGEX: str = rf"{END_OF_TABLE}|(?<=</table>)|(?<=</table\s>)"


class TableSerializer(ABC, BaseModel):
    table_splitter_backup: str = r"</\s*[^>]+>\s*?"
    
    @classmethod
    def from_table_serializer_preprocess_config(cls, table_serialization: TableSerializerPreprocessConfig):
        from ...model.tabtree.tabtree_serializer import TabTreeSerializer

        if table_serialization.table_serializer == "html":
            return HTMLSerializer()
        elif table_serialization.table_serializer == "csv":
            return CSVSerializer()
        elif table_serialization.table_serializer == "tsv":
            return TSVSerializer()
        elif table_serialization.table_serializer == "df-loader":
            return DFLoaderSerializer()
        elif table_serialization.table_serializer == "json-records":
            return JSONSerializerRecords()
        elif table_serialization.table_serializer == "json-split":
            return JSONSerializerSplit()
        elif table_serialization.table_serializer == "json-index":
            return JSONSerializerIndex()
        elif table_serialization.table_serializer == "markdown":
            return MarkdownSerializer()
        elif table_serialization.table_serializer == "text":
            return TextSerializer()
        elif table_serialization.table_serializer == "text-bullet-points":
            return TextSerializerBulletPoints()
        elif table_serialization.table_serializer == "list-item":
            return ListItemSerializer()
        elif table_serialization.table_serializer == "matrix":
            return MatrixSerializer()
        elif table_serialization.table_serializer == "none":
            return None
        elif table_serialization.table_serializer == "plain_text":
            return PlainTextSerializer()
        elif table_serialization.table_serializer == "tabtree":
            return TabTreeSerializer(node_approach=table_serialization.tabtree_approach)
        else:
            raise ValueError(
                f"Table serialization type {table_serialization.table_serializer} is not supported"
            )

    @classmethod
    def from_preprocess_config(cls, config: PreprocessConfig) -> TableSerializer | None:
        return cls.from_table_serializer_preprocess_config(table_serialization=config.table_serialization)

    def serialize_table_to_str(
        self,
        table_str: str,
    ) -> SerializedTable:
        logging.info(
            f"Start serializing table with table serializer class: {self.__class__}"
        )
        df_table = self.serialize_table_to_extended_table(table_str)
        if not df_table or not df_table.serialized_table:
            return "", None
        else:
            return df_table.serialized_table, None

    def serialize_table_to_extended_table(self, table_str: str) -> ExtendedTable | None:
        df_table = self.table_str_to_df(table_str)
        if not df_table:
            return None
        else:
            serialized_string = self.df_to_serialized_string(df_table)
            df_table.serialized_table = serialized_string
        return df_table

    def table_str_to_df(self, table: str) -> ExtendedTable | None:
        df_table = self.load_table_to_df(table)
        df_table = self.delete_nan_columns_and_rows(df_table)
        if not df_table:
            return None

        df_table = self.delete_duplicate_columns_and_rows(df_table)
        return self.replace_nan_values(df_table, "")

    def delete_duplicate_columns_and_rows(
        self, custom_table: ExtendedTable
    ) -> ExtendedTable:
        df = custom_table.df
        df = df.drop_duplicates()
        df = df.T.drop_duplicates().T
        return ExtendedTable(
            df=df,
            has_header=custom_table.has_header,
            raw_table=custom_table.raw_table,
        )

    def replace_nan_values(
        self, custom_table: ExtendedTable, value: str
    ) -> ExtendedTable:
        df = custom_table.df
        df.fillna(value, inplace=True)
        df = df.infer_objects()
        return ExtendedTable(
            df=df,
            has_header=custom_table.has_header,
            raw_table=custom_table.raw_table,
        )

    def delete_nan_columns_and_rows(
        self, custom_table: ExtendedTable
    ) -> ExtendedTable | None:
        df = custom_table.df
        logging.info(f"Original shape of the Table: {df.shape}")
        df = df.dropna(axis=0, how="all")
        df = df.dropna(axis=1, how="all")
        logging.info(
            f"Shape of the Table after dropping NaN columns and rows: {df.shape}"
        )
        if df.empty:
            return None
        else:
            return ExtendedTable(
                df=df,
                has_header=custom_table.has_header,
                raw_table=custom_table.raw_table,
            )

    @abstractmethod
    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        pass

    def load_table_to_df(self, table: str) -> ExtendedTable:
        soup = BeautifulSoup(table, "html.parser")
        has_thead = soup.find(["thead", "th"]) is not None

        tables = pd.read_html(StringIO(table), header=0 if has_thead else None)
        if not tables:
            raise ValueError("Could not load table to DataFrame")
        elif len(tables) > 1:
            raise ValueError("More than one table found in the table string")
        return ExtendedTable(df=tables[0], has_header=has_thead, raw_table=table)

    def serialize_tables_in_document(
        self, document: CustomDocument
    ) -> Tuple[str, List[SplitContent]]:
        serialized_docs: List[SplitContent] = []
        if not document.splitted_content:
            raise ValueError("Document does not have any splitted content")

        for content in document.splitted_content:
            if content.type == "table":
                content.original_content = content.content
                new_content = self.serialize_table_to_str(content.content)[0]
                if new_content is None:
                    raise ValueError("Could not serialize table")
                content.content = new_content
                new_content = (
                    f"{new_content}{END_OF_TABLE}" if new_content.strip() != "" else ""
                )
            else:
                new_content = (
                    content.content + " " if content.content.strip() != "" else ""
                )
                content.content = new_content

            serialized_docs.append(content)

        # only consider sentences / content items which are longer than 2 characters (e.g. filter out page numbering)
        serialized_docs_filtered = [
            doc for doc in serialized_docs if len(doc.content.strip()) > 3
        ]
        return (
            "".join([doc.content for doc in serialized_docs_filtered]),
            serialized_docs_filtered,
        )


class DFLoaderSerializer(TableSerializer):
    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        df = df_with_header.df
        return df.to_string(index=False)


class HTMLSerializer(TableSerializer):
    table_splitter_backup: str = r"</\s*[^>]+>\s*?"
    
    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        raise NotImplementedError(
            "HTMLSerializer should not be used for serialization. Use HTMLTableParser instead."
        )
    
    @override
    def serialize_table_to_extended_table(self, table_str: str) -> ExtendedTable | None:
        html_table = HTMLTableParser.parse_and_clean_table(table_str)
        if not html_table:
            return None
        return html_table.to_extended_table(serialized_table=html_table.to_html())


class CSVSerializer(TableSerializer):
    table_splitter_backup: str = r"\n"
    index: bool = True

    def df_to_serialized_string(
        self,
        df_with_header: ExtendedTable,
    ) -> str:
        df = df_with_header.df
        return df.to_csv(header=df_with_header.has_header, index=self.index, sep=";")


class TSVSerializer(TableSerializer):
    table_splitter_backup: str = r"\n"

    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        df = df_with_header.df
        return df.to_csv(header=df_with_header.has_header, index=True, sep="\t")


class JSONSerializerRecords(TableSerializer):
    table_splitter_backup: str = r"\},\s"

    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        df = df_with_header.df
        records_dict = df.to_dict(orient="records")
        # get rid of null values for each item in records
        records = []
        for record in records_dict:
            record = {k: v for k, v in record.items() if pd.notnull(v) and v != ""}
            if len(record) > 0:
                records.append(record)

        if len(records) == 0:
            return ""
        else:
            return json.dumps(records, indent=None)


class JSONSerializerSplit(TableSerializer):
    table_splitter_backup: str = r"\},\s"

    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        df = df_with_header.df
        return df.to_json(orient="split")


class JSONSerializerIndex(TableSerializer):
    table_splitter_backup: str = r"\},\s"

    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        df = df_with_header.df
        return df.to_json(orient="index")


class MatrixSerializer(TableSerializer):
    table_splitter_backup: str = r"],"

    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        df = df_with_header.df
        return df.to_json(orient="values")


class MarkdownSerializer(TableSerializer):
    table_splitter_backup: str = r"\n"

    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        df = df_with_header.df
        return df.to_markdown(index=True, missingval="")


class TextSerializer(TableSerializer):
    table_splitter_backup: str = r"\n"

    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        df = df_with_header.df
        text_template = []
        for _, row in df.iterrows():
            row_text = [f"The {col} is {row[col]}." for col in df.columns]
            text_template.append(" ".join(row_text))
        return "\n".join(text_template)


class TextSerializerBulletPoints(TableSerializer):
    table_splitter_backup: str = r"\n"

    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        df = df_with_header.df
        bullet_point_text = []
        for _, row in df.iterrows():
            row_text = [f"{col} is {row[col]}" for col in df.columns]
            bullet_point_text.append("- " + ". ".join(row_text) + ".")
        return "\n".join(bullet_point_text)


class ListItemSerializer(TableSerializer):
    table_splitter_backup: str = r"\n"

    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        df = df_with_header.df
        text_template = []
        for _, row in df.iterrows():
            row_text = [f"The {col} is {row[col]}." for col in df.columns]
            text_template.append(" ".join(row_text))
        return "\n".join(text_template)


class PlainTextSerializer(TableSerializer):
    table_splitter_backup: str = r"\n"

    @override
    def serialize_table_to_str(
        self,
        table_str: str,
    ) -> SerializedTable:
        """Returns the plain characters of the table"""
        table_str = BeautifulSoup(table_str, "html.parser").get_text(strip=True)
        return table_str, None

    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        return df_with_header.df.to_string(index=False)

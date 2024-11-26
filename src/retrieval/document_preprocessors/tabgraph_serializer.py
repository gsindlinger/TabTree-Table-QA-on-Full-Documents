from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple
import pandas as pd
from pydantic import BaseModel
from anytree import Node, RenderTree

from ...pipeline import TableHeaderRowsPipeline
from .table_serializer import CustomTable, CustomTable


class TabGraphSerializer(BaseModel):
    table_splitter_backup: str = r"\n"

    def df_to_serialized_string(self, custom_table: CustomTable) -> str:
        tab_graph = TabGraph.from_df(custom_table)
        return tab_graph[0].generate_serialized_string()


class TabGraph(ABC, BaseModel):
    """Assuming empty rows and columns are already deleted."""

    class Config:
        arbitrary_types_allowed = True

    tree: Node
    custom_table: CustomTable

    @classmethod
    def from_df(cls, custom_table: CustomTable) -> Tuple[TabGraph, TabGraph]:
        row_header_indices, col_header_indices = TabGraph._get_headers(
            custom_table.raw_table
        )
        custom_table.set_headers(row_header_indices, col_header_indices)

        tabgraph_row = TabGraphRowBased.generate_tree(custom_table)
        tabgraph_column = TabGraphColBased.generate_tree(custom_table)

        return tabgraph_row, tabgraph_column

    @classmethod
    @abstractmethod
    def generate_tree(cls, custom_table: CustomTable) -> TabGraph:
        pass

    @staticmethod
    def _get_headers(raw_table: str) -> tuple[list[int], list[int]]:
        table_header_rows_columns = TableHeaderRowsPipeline.from_config()
        return table_header_rows_columns.get_table_header_rows_columns(raw_table)

    def generate_serialized_string(self) -> str:
        return ""


class TabGraphRowBased(TabGraph):
    @classmethod
    def generate_tree(cls, custom_table: CustomTable) -> TabGraph:
        tree = Node("root")
        return cls(tree=tree, custom_table=custom_table)


class TabGraphColBased(TabGraph):
    @classmethod
    def generate_tree(cls, custom_table: CustomTable) -> TabGraph:
        tree = Node("root")
        return cls(tree=tree, custom_table=custom_table)

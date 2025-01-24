from __future__ import annotations
from abc import ABC
import logging
from typing import ClassVar, Dict, Tuple
from pydantic import (
    BaseModel,
    model_validator,
)
from enum import Enum
import networkx as nx

from ...retrieval.document_preprocessors.table_parser.custom_cell import CustomCell


class TabTree(nx.DiGraph):
    COLUMN_HEADER_TREE_ROOT: str = "column_header_tree_root"
    ROW_LABEL_TREE_ROOT: str = "row_label_tree_root"

    def add_node(self, node: ColouredNode) -> None:
        super().add_node(node.id, **node.to_dict())

    def add_edge(self, source: ColouredNode, target: ColouredNode) -> None:
        self.add_node(source)
        self.add_node(target)
        super().add_edge(source.id, target.id)

    def has_edge(self, source: ColouredNode, target: ColouredNode) -> bool:
        return super().has_edge(source.id, target.id)

    def get_node_by_cell(self, cell: CustomCell) -> ColouredNode | None:
        row_index = cell.row_index
        col_index = cell.col_index
        return self.get_node_by_index(row_index, col_index)

    def get_node_by_index(self, row_index: int, col_index: int) -> ColouredNode | None:
        node_id = CellNode.generate_id(row_index, col_index)
        node = self.nodes.get(node_id)

        if node:
            return ColouredNode.from_dict(node)
        else:
            return None

    def get_column_nodes_by_row_index(
        self, row_index: int, start_col_index: int = 0
    ) -> list[ColouredNode]:
        lst = []
        j = start_col_index
        while node := self.get_node_by_index(row_index, j):
            if isinstance(node, CellNode) and node.colspan[0] == 0:
                lst.append(node)
                j += node.colspan[1] + 1
        return lst

    def get_row_nodes_by_column_index(
        self, column_index: int, start_row_index: int = 0
    ) -> list[ColouredNode]:
        lst = []
        i = start_row_index
        while node := self.get_node_by_index(i, column_index):
            if isinstance(node, CellNode) and node.rowspan[0] == 0:
                lst.append(node)
                i += node.rowspan[1] + 1
        return lst


class NodeColor(str, Enum):
    YELLOW = "yellow"
    BLUE = "blue"
    ORANGE = "orange"
    GRAY = "gray"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> NodeColor:
        return NodeColor(value)


class ColouredNode(BaseModel):
    id: str
    colour: NodeColor

    @property
    def node_type(self) -> str:
        return self.__class__.__name__

    def to_dict(self) -> Dict:
        return self.model_dump()

    @staticmethod
    def from_dict(node: Dict) -> ColouredNode:
        match NodeColor(node["colour"]):
            case NodeColor.YELLOW:
                if node["id"] == TabTree.COLUMN_HEADER_TREE_ROOT:
                    return ColumnHeaderTreeRoot()
                else:
                    return ColumnHeaderNode(**node)
            case NodeColor.BLUE:
                if node["id"] == TabTree.ROW_LABEL_TREE_ROOT:
                    return RowLabelTreeRoot()
                else:
                    return RowLabelNode(**node)
            case NodeColor.GRAY:
                return ValueNode(**node)
            case NodeColor.ORANGE:
                return ContextIntersectionNode(**node)
            case _:
                raise ValueError("Invalid colour")


class CellNode(ColouredNode):
    row_index: int
    col_index: int
    value: str
    rowspan: Tuple[int, int]
    colspan: Tuple[int, int]

    @staticmethod
    def generate_id(row_index: int, col_index: int) -> str:
        return f"row-{row_index}_col-{col_index}"

    @classmethod
    def from_custom_cell(cls, cell: CustomCell) -> CellNode:
        return cls(
            id=cls.generate_id(cell.row_index, cell.col_index),
            row_index=cell.row_index,
            col_index=cell.col_index,
            value=cell.value,
            rowspan=cell.rowspan,
            colspan=cell.colspan,
        )  # type: ignore


class ColumnHeaderNode(CellNode):
    colour: NodeColor = NodeColor.YELLOW


class RowLabelNode(CellNode):
    colour: NodeColor = NodeColor.BLUE


class ValueNode(CellNode):
    colour: NodeColor = NodeColor.GRAY


class ContextIntersectionNode(CellNode):
    colour: NodeColor = NodeColor.ORANGE


class ColumnHeaderTreeRoot(ColouredNode):
    colour: NodeColor = NodeColor.YELLOW
    id: str = TabTree.COLUMN_HEADER_TREE_ROOT


class RowLabelTreeRoot(ColouredNode):
    colour: NodeColor = NodeColor.BLUE
    id: str = TabTree.ROW_LABEL_TREE_ROOT

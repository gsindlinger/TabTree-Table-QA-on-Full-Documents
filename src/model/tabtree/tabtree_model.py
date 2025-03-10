from __future__ import annotations
from abc import ABC
from typing import Callable, Dict, Iterator, List, Optional, Self, Tuple
from pydantic import (
    BaseModel,
    model_validator,
)
from enum import Enum
import networkx as nx

from ...retrieval.document_preprocessors.table_parser.custom_cell import CustomCell
from .string_generation.approaches import ContextNodeApproach, NodeApproach


class TabTree(nx.DiGraph):

    COLUMN_HEADER_TREE_ROOT: str = "column_header_tree_root"
    ROW_LABEL_TREE_ROOT: str = "row_label_tree_root"

    def __init__(self, context_colour: NodeColor):
        super().__init__()
        self.context_colour = context_colour

    @property
    def secondary_colour(self) -> NodeColor:
        match self.context_colour:
            case NodeColor.YELLOW:
                return NodeColor.BLUE
            case NodeColor.BLUE:
                return NodeColor.YELLOW
            case _:
                raise ValueError("Invalid primary colour")

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

    def get_node_by_network_id(self, node_id: str) -> ColouredNode | None:
        node = self.nodes.get(node_id)
        if node:
            return ColouredNode.from_dict(node)
        else:
            return None

    def successors(self, node_id: str) -> Iterator[CellNode]:
        for successor in super().successors(node_id):
            successor_obj = self.get_node_by_network_id(successor)

            if not isinstance(successor_obj, CellNode):
                raise ValueError(
                    "Found successor which isn't a cell node. This should not happen."
                )
            else:
                yield successor_obj

    def predecessors(self, node_id: str) -> Iterator[ColouredNode]:
        for predecessor in super().predecessors(node_id):
            predecessor_obj = self.get_node_by_network_id(predecessor)
            if predecessor_obj:
                yield predecessor_obj
            else:
                raise ValueError("Predecessor not found")

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

    def get_children_filtered(self, node: ColouredNode) -> List[CellNode]:
        return [
            child for child in self.successors(node.id) if child.colour == node.colour
        ]

    def get_parent(
        self, node: ColouredNode, filter_colour: Optional[NodeColor] = None
    ) -> ColouredNode | None:
        for pred in self.predecessors(node.id):
            if not filter_colour or pred.colour == filter_colour:
                return pred
        return None

    def get_siblings_filtered(self, node: ColouredNode) -> List[ColouredNode]:
        parent = self.get_parent(node)
        if not parent:
            return []
        return [child for child in self.get_children_filtered(parent) if child != node]

    def get_root_node(self) -> ColouredNode:
        match self.context_colour:
            case NodeColor.YELLOW:
                root = self.nodes.get(self.COLUMN_HEADER_TREE_ROOT)
            case NodeColor.BLUE:
                root = self.nodes.get(self.ROW_LABEL_TREE_ROOT)

        if not root:
            raise ValueError("Root node not found")

        return ColouredNode.from_dict(root)

    def get_context_intersection_sequence(self, node: ContextNode) -> List[CellNode]:
        sequence: List[CellNode] = [node]
        current_node = node
        # find unique node with colour orange in the path starting from node
        while current_node := self.get_parent(current_node, NodeColor.ORANGE):
            if isinstance(current_node, ContextIntersectionNode):
                # only append if the node is not empty and is different from the last node
                if current_node.value.strip() != "" and not (
                    current_node.value.strip() == sequence[-1].value.strip()
                ):
                    sequence.append(current_node)
        sequence.reverse()
        return sequence

    def get_value_sequence(self, node: ValueNode) -> List[CellNode]:
        sequence: List[CellNode] = [node]
        current_node = node
        # find unique node with colour equals primary tree colour
        while current_node := self.get_parent(current_node, self.context_colour):
            # only append context nodes / exclude root node
            if isinstance(current_node, ContextNode):
                sequence.append(current_node)
        sequence.reverse()
        return sequence

    def get_value_sequence_with_context_intersection(
        self, node: ValueNode
    ) -> List[List[CellNode]]:
        sequence: List[List[CellNode]] = [[node]]
        current_node = node
        # find unique node with colour orange in the path starting from node
        while current_node := self.get_parent(current_node, self.context_colour):
            # only append context nodes / exclude root node
            if isinstance(current_node, ContextNode):
                context_intersection_sequence = self.get_context_intersection_sequence(
                    current_node
                )
                sequence.append(context_intersection_sequence)
        sequence.reverse()
        return sequence

    def dfs_serialization(
        self,
        secondary_tree: TabTree,
        approaches: Tuple[Optional[NodeApproach], Optional[NodeApproach]],
    ) -> str:
        from .string_generation.context_string import ContextStringGeneration
        from .string_generation.value_string import ValueStringGeneration

        # Define generation approaches / methods which will be executed in the dfs_serialization
        context_string_generation: Callable[
            [ContextNode | ColumnHeaderTreeRoot | RowLabelTreeRoot], str
        ] = lambda node: ContextStringGeneration.generate_string(
            node=node, primary_tree=self, approach=approaches[0]
        )
        value_string_generation: Callable[[ValueNode], str] = (
            lambda node: ValueStringGeneration.generate_string(
                node,
                primary_tree=self,
                secondary_tree=secondary_tree,
                approach=approaches[1],
            )
        )

        start_node = self.get_root_node()
        serialized_str = ""
        visited = []
        stack = [(start_node, 0)]

        while stack:
            node, level = stack.pop()
            if node not in visited:
                if (
                    isinstance(
                        node, ContextNode | ColumnHeaderTreeRoot | RowLabelTreeRoot
                    )
                    and approaches[0] != ContextNodeApproach.EMPTY
                ):
                    context_string = context_string_generation(node=node)
                    serialized_str += level * "  " + context_string + "\n"
                elif isinstance(node, ValueNode):
                    # if context node approach is empty, then add no tab spaces
                    value_string = value_string_generation(node=node)
                    # since there are sometimes duplicate value node combinations,
                    # we only add the value node if the same value node combination is not already in the serialized string
                    if value_string not in serialized_str:
                        if approaches[0] == ContextNodeApproach.EMPTY:
                            serialized_str += value_string + "\n"
                        else:
                            serialized_str += level * "  " + value_string + "\n"
                visited.append(node)

                # sort the successors in order to keep 'initial' table ordering
                successor_list = list(self.successors(node.id))
                if successor_list:
                    if self.context_colour == NodeColor.YELLOW:
                        if all(
                            isinstance(successor, ValueNode)
                            for successor in successor_list
                        ):
                            successor_list.sort(key=lambda x: x.row_index, reverse=True)
                        else:
                            successor_list.sort(key=lambda x: x.col_index, reverse=True)
                    elif self.context_colour == NodeColor.BLUE:
                        if all(
                            isinstance(successor, ValueNode)
                            for successor in successor_list
                        ):
                            successor_list.sort(key=lambda x: x.col_index, reverse=True)
                        else:
                            successor_list.sort(key=lambda x: x.row_index, reverse=True)
                stack.extend([(successor, level + 1) for successor in successor_list])
        return serialized_str


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
        try:
            node_colour = NodeColor.from_str(node["colour"])
        except TypeError:
            raise ValueError("Invalid colour")
        match node_colour:
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

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        """If the value is empty and the node is not a context-intersection node, set the value to "null" """
        if self.value == "" and self.colour != NodeColor.ORANGE:
            self.value = "null"
        return self

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


class ContextNode(ABC, CellNode):
    pass


class ColumnHeaderNode(ContextNode):
    colour: NodeColor = NodeColor.YELLOW


class RowLabelNode(ContextNode):
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

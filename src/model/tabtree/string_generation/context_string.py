from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel

from .general import StringGenerationService
from .approaches import (
    ContextNodeApproach,
    NodeApproach,
)
from .separator_approach import SeparatorApproach
from ....config.config import Config
from ..tabtree_model import CellNode, ContextNode, NodeColor, TabTree
from .general import StringGenerationService


class ContextStringGeneration(StringGenerationService):
    @classmethod
    def generate_string(
        cls,
        node: ContextNode,
        primary_tree: TabTree,
        approach: Optional[NodeApproach] = None,
    ) -> str:
        service = cls.from_config(approach)
        if not isinstance(service, ContextStringGeneration):
            raise ValueError(
                f"Invalid approach type for context string generation: {approach}"
            )
        if not isinstance(node, ContextNode):
            raise ValueError(f"Invalid node type for context string generation: {node}")
        if node.colour != primary_tree.context_colour:
            raise ValueError(
                f"Node colour does not match primary tree colour: {node.colour} != {primary_tree.context_colour}"
            )
        return service.generate_context_string(node=node, primary_tree=primary_tree)

    @abstractmethod
    def generate_context_string(
        self,
        node: ContextNode,
        primary_tree: TabTree,
    ) -> str:
        pass

    def retrieve_siblings_and_children(
        self, node: CellNode, primary_tree: TabTree
    ) -> Tuple[str, List, Optional[str], List, Optional[str]]:
        primary_tree_str = (
            "column header"
            if primary_tree.context_colour == NodeColor.YELLOW
            else "row label"
        )

        siblings = primary_tree.get_siblings_filtered(node)
        siblings = [sibling for sibling in siblings if isinstance(sibling, CellNode)]
        if len(siblings) == 0:
            siblings_str = None
        else:
            siblings_str = StringGenerationService.node_sequence_to_string(siblings)

        children = primary_tree.get_children_filtered(node)
        children = [child for child in children if isinstance(child, CellNode)]
        if len(children) == 0:
            children_str = None
        else:
            children_str = StringGenerationService.node_sequence_to_string(children)
        return primary_tree_str, siblings, siblings_str, children, children_str


class ContextStringGenerationBase(ContextStringGeneration):
    approach: Optional[NodeApproach] = NodeApproach(
        approach=ContextNodeApproach.BASE, include_context_intersection=False
    )

    def generate_context_string(
        self,
        node: ContextNode,
        primary_tree: TabTree,
    ) -> str:
        return f"{node.value}:"


class ContextStringGenerationText(ContextStringGeneration):
    approach: Optional[NodeApproach] = NodeApproach(
        approach=ContextNodeApproach.TEXT, include_context_intersection=False
    )

    def generate_context_string(
        self,
        node: ContextNode,
        primary_tree: TabTree,
    ) -> str:
        primary_tree_str, siblings, siblings_str, children, children_str = (
            self.retrieve_siblings_and_children(node, primary_tree)
        )

        if len(siblings) > 0 and len(children) > 0:
            return f"The {primary_tree_str} {node.value} has siblings {siblings_str}. The children of {node.value} are {children_str}."
        elif len(siblings) > 0 and len(children) == 0:
            return f"The {primary_tree_str} {node.value} has siblings {siblings_str}. The values of {node.value} are:"
        elif len(siblings) == 0 and len(children) > 0:
            return f"The {primary_tree_str} {node.value} has no siblings. The children of {node.value} are {children_str}."
        else:
            return f"The values of the {primary_tree_str} {node.value} are:"


class ContextStringGenerationTextWithIntersection(ContextStringGeneration):
    approach: Optional[NodeApproach] = NodeApproach(
        approach=ContextNodeApproach.TEXT, include_context_intersection=True
    )

    def generate_context_string(
        self,
        node: ContextNode,
        primary_tree: TabTree,
    ) -> str:

        context_intersection_sequence = primary_tree.get_context_intersection_sequence(
            node
        )
        context_intersection_str = StringGenerationService.node_sequence_to_string(
            context_intersection_sequence,
            separator_approach=SeparatorApproach.AND_SYMBOL_COLON,
        )
        return f"{context_intersection_str}:"


class ContextStringGenerationBaseWithIntersection(ContextStringGeneration):
    approach: Optional[NodeApproach] = NodeApproach(
        approach=ContextNodeApproach.BASE, include_context_intersection=True
    )

    def generate_context_string(
        self,
        node: ContextNode,
        primary_tree: TabTree,
    ) -> str:
        primary_tree_str, siblings, siblings_str, children, children_str = (
            self.retrieve_siblings_and_children(node, primary_tree)
        )
        context_intersection_sequence = primary_tree.get_context_intersection_sequence(
            node
        )[:-1]

        if len(context_intersection_sequence) == 0:
            return ContextStringGenerationText().generate_context_string(
                node=node, primary_tree=primary_tree
            )

        context_intersection_str = StringGenerationService.node_sequence_to_string(
            context_intersection_sequence,
            separator_approach=SeparatorApproach.COMMA_AND,
        )
        context_intersection_str = (
            f"The {primary_tree_str} represents {context_intersection_str}."
        )

        if len(siblings) > 0 and len(children) > 0:
            return f"{node.value} has siblings {siblings_str}. The children of {node.value} are {children_str}."
        elif len(siblings) > 0 and len(children) == 0:
            return f"{node.value} has siblings {siblings_str}."
        elif len(siblings) == 0 and len(children) > 0:
            return f"{node.value} has children {children_str}."
        else:
            return f"The values of {node.value} are:"

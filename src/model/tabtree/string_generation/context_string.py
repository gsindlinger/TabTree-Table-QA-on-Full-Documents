from __future__ import annotations
from abc import abstractmethod
from typing import List, Optional, Tuple

from .general import StringGenerationService
from .approaches import (
    ContextNodeApproach,
    NodeApproach,
)
from .separator_approach import SeparatorApproach
from ..tabtree_model import (
    CellNode,
    ColumnHeaderTreeRoot,
    ContextNode,
    NodeColor,
    RowLabelTreeRoot,
    TabTree,
)
from .general import StringGenerationService


class ContextStringGeneration(StringGenerationService):
    @classmethod
    def generate_string(
        cls,
        node: ContextNode | ColumnHeaderTreeRoot | RowLabelTreeRoot,
        primary_tree: TabTree,
        approach: Optional[NodeApproach] = None,
    ) -> str:
        service = cls.from_config(approach)
        if not isinstance(service, ContextStringGeneration):
            raise ValueError(
                f"Invalid approach type for context string generation: {approach}"
            )
        if not isinstance(node, ContextNode | ColumnHeaderTreeRoot | RowLabelTreeRoot):
            raise ValueError(f"Invalid node type for context string generation: {node}")
        if node.colour != primary_tree.context_colour:
            raise ValueError(
                f"Node colour does not match primary tree colour: {node.colour} != {primary_tree.context_colour}"
            )

        # If we want to add information at the root level
        if isinstance(node, ColumnHeaderTreeRoot | RowLabelTreeRoot):
            return cls.get_root_string(
                node=node, primary_tree=primary_tree, approach=approach
            )

        return service.generate_context_string(node=node, primary_tree=primary_tree)

    @abstractmethod
    def generate_context_string(
        self,
        node: ContextNode,
        primary_tree: TabTree,
    ) -> str:
        pass

    @classmethod
    def get_root_string(
        cls,
        node: ColumnHeaderTreeRoot | RowLabelTreeRoot,
        primary_tree: TabTree,
        approach: Optional[NodeApproach] = None,
    ) -> str:
        # Only apply for context node approach text
        if not approach or not approach.approach == ContextNodeApproach.TEXT:
            return ""

        root_children = primary_tree.get_children_filtered(node)
        if isinstance(node, ColumnHeaderTreeRoot):
            primary_tree_str = "column header"
        else:
            primary_tree_str = "row label"

        if len(root_children) == 0:
            return ""
        elif len(root_children) == 1:
            return f"The table captures {root_children[0].value} as its main {primary_tree_str}."
        else:
            return f"The table captures {cls.node_sequence_to_string(root_children)} as its main {primary_tree_str}s."

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


class ContextStringGenerationEmpty(ContextStringGeneration):
    approach: Optional[NodeApproach] = NodeApproach(
        approach=ContextNodeApproach.EMPTY, include_context_intersection=False
    )

    def generate_context_string(
        self,
        node: ContextNode,
        primary_tree: TabTree,
    ) -> str:
        return ""


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


class ContextStringGenerationBaseWithIntersection(ContextStringGeneration):
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
            separator_approach=SeparatorApproach.AND,
        )
        return f"{context_intersection_str}:"


class ContextStringGenerationTextWithIntersection(ContextStringGeneration):
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
        context_intersection_str = f"The {primary_tree_str} {node.value} represents {context_intersection_str}."

        if len(siblings) > 0 and len(children) > 0:
            return f"{context_intersection_str} The {primary_tree_str} {node.value} has siblings {siblings_str}. The children of {node.value} are {children_str}."
        elif len(siblings) > 0 and len(children) == 0:
            return f"{context_intersection_str} The {primary_tree_str} {node.value} has siblings {siblings_str}. The values of the {primary_tree_str} {node.value} are:"
        elif len(siblings) == 0 and len(children) > 0:
            return f"{context_intersection_str} The {primary_tree_str} {node.value} has children {children_str}."
        else:
            return f"{context_intersection_str} The values of the {primary_tree_str} {node.value} are:"

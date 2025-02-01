from __future__ import annotations
from abc import abstractmethod
from typing import List, Optional

from .approaches import NodeApproach, ValueNodeApproach
from .general import StringGenerationService
from .separator_approach import SeparatorApproach
from ..tabtree_model import (
    ContextNode,
    ValueNode,
    NodeColor,
    TabTree,
)


class ValueStringGeneration(StringGenerationService):
    @classmethod
    def generate_string(
        cls,
        node: ValueNode,
        primary_tree: TabTree,
        secondary_tree: TabTree,
        approach: Optional[NodeApproach] = None,
    ) -> str:
        generation_service = StringGenerationService.from_config(approach)
        if not isinstance(generation_service, ValueStringGeneration):
            raise ValueError(
                f"Invalid approach type for value string generation: {generation_service}"
            )
        if not isinstance(node, ValueNode):
            raise ValueError(f"Invalid node type for context string generation: {node}")
        if node.colour != NodeColor.GRAY:
            raise ValueError(
                f"Node colour does not match the value node colour: {node.colour} != {NodeColor.GRAY}"
            )
        return generation_service.generate_value_string(
            node=node, primary_tree=primary_tree, secondary_tree=secondary_tree
        )

    @abstractmethod
    def generate_value_string(
        self, node: ValueNode, primary_tree: TabTree, secondary_tree: TabTree
    ) -> str:
        pass


class ValueStringGenerationBase(ValueStringGeneration):
    approach: Optional[NodeApproach] = NodeApproach(
        approach=ValueNodeApproach.BASE, include_context_intersection=False
    )

    def generate_value_string(
        self,
        node: ValueNode,
        primary_tree: TabTree,
        secondary_tree: TabTree,
    ) -> str:
        value_sequence = secondary_tree.get_value_sequence(node)
        value_str = StringGenerationService.node_sequence_to_string(
            value_sequence, separator_approach=SeparatorApproach.HIERARCHY_COLON
        )
        return value_str


class ValueStringGenerationText(ValueStringGeneration):
    approach: Optional[NodeApproach] = NodeApproach(
        approach=ValueNodeApproach.TEXT, include_context_intersection=False
    )

    def generate_value_string(
        self,
        node: ValueNode,
        primary_tree: TabTree,
        secondary_tree: TabTree,
    ) -> str:

        value_sequence = secondary_tree.get_value_sequence(node)[:-1]
        value_str = StringGenerationService.node_sequence_to_string(
            value_sequence, separator_approach=SeparatorApproach.COMMA
        )

        return self.text_approach_generation(
            node=node,
            value_sequence=value_sequence,
            value_str=value_str,
            primary_tree=primary_tree,
            secondary_tree=secondary_tree,
        )

    @staticmethod
    def text_approach_generation(
        node: ValueNode,
        value_sequence: List,
        value_str: str,
        primary_tree: TabTree,
        secondary_tree: TabTree,
    ):

        primary_tree_str = (
            "column header"
            if primary_tree.context_colour == NodeColor.YELLOW
            else "row label"
        )
        secondary_tree_str = (
            "column header"
            if secondary_tree.context_colour == NodeColor.YELLOW
            else "row label"
        )

        parent_node = primary_tree.get_parent(
            node, filter_colour=primary_tree.context_colour
        )

        if not isinstance(parent_node, ContextNode):  # so directly connected to root
            if len(value_sequence) < 1:
                raise ValueError(
                    f"There must be at least one connected context node the value node: {node.id}"
                )
            elif len(value_sequence) == 1:
                return f"The value of the {secondary_tree_str} {value_str} is {node.value}."
            else:
                return f"The value of the {secondary_tree_str} combination {value_str} is {node.value}."
        else:
            if len(value_sequence) < 1:
                return f"The value of the {primary_tree_str} {parent_node.value} is {node.value}."
            elif len(value_sequence) == 1:
                return f"The value of the {primary_tree_str} {parent_node.value} and the {secondary_tree_str} {value_str} is {node.value}."
            else:
                return f"The value of the {primary_tree_str} {parent_node.value} and the {secondary_tree_str} combination {value_str} is {node.value}."


class ValueStringGenerationBaseWithIntersection(ValueStringGeneration):
    approach: Optional[NodeApproach] = NodeApproach(
        approach=ValueNodeApproach.BASE, include_context_intersection=True
    )

    def generate_value_string(
        self,
        node: ValueNode,
        primary_tree: TabTree,
        secondary_tree: TabTree,
    ) -> str:
        value_sequence = secondary_tree.get_value_sequence_with_context_intersection(
            node
        )[:-1]
        value_str = StringGenerationService.sequence_of_node_sequence_to_string(
            value_sequence,
            first_level_separator_approach=SeparatorApproach.HIERARCHY,
            second_level_separator_approach=SeparatorApproach.AND,
        )
        return f"{value_str}: {node.value}"


class ValueStringGenerationTextWithIntersection(ValueStringGeneration):
    approach: Optional[NodeApproach] = NodeApproach(
        approach=ValueNodeApproach.TEXT, include_context_intersection=True
    )

    def generate_value_string(
        self,
        node: ValueNode,
        primary_tree: TabTree,
        secondary_tree: TabTree,
    ) -> str:
        value_sequence = secondary_tree.get_value_sequence_with_context_intersection(
            node
        )[:-1]
        value_str = StringGenerationService.sequence_of_node_sequence_to_string(
            value_sequence,
            first_level_separator_approach=SeparatorApproach.COMMA,
            second_level_separator_approach=SeparatorApproach.AND,
        )
        return ValueStringGenerationText.text_approach_generation(
            node=node,
            value_sequence=value_sequence,
            value_str=value_str,
            primary_tree=primary_tree,
            secondary_tree=secondary_tree,
        )


class ValueStringGenerationTextAugmented(ValueStringGeneration):
    approach: Optional[NodeApproach] = NodeApproach(
        approach=ValueNodeApproach.TEXT_AUGMENTED, include_context_intersection=False
    )

    def generate_value_string(
        self,
        node: ValueNode,
        primary_tree: TabTree,
        secondary_tree: TabTree,
    ) -> str:

        value_sequence_primary = primary_tree.get_value_sequence(node)[:-1]
        value_str_primary = StringGenerationService.node_sequence_to_string(
            value_sequence_primary, separator_approach=SeparatorApproach.COMMA
        )
        value_sequence_secondary = secondary_tree.get_value_sequence(node)[:-1]
        value_str_secondary = StringGenerationService.node_sequence_to_string(
            value_sequence_secondary, separator_approach=SeparatorApproach.COMMA
        )

        return self.text_augmented_approach_generation(
            node=node,
            value_sequence_primary=value_sequence_primary,
            value_str_primary=value_str_primary,
            value_sequence_secondary=value_sequence_secondary,
            value_str_secondary=value_str_secondary,
            primary_tree=primary_tree,
            secondary_tree=secondary_tree,
        )

    @staticmethod
    def text_augmented_approach_generation(
        node: ValueNode,
        value_sequence_primary: List,
        value_str_primary: str,
        value_sequence_secondary: List,
        value_str_secondary: str,
        primary_tree: TabTree,
        secondary_tree: TabTree,
    ):

        primary_tree_str = (
            "column header"
            if primary_tree.context_colour == NodeColor.YELLOW
            else "row label"
        )
        secondary_tree_str = (
            "column header"
            if secondary_tree.context_colour == NodeColor.YELLOW
            else "row label"
        )

        if len(value_sequence_primary) < 1:
            if len(value_sequence_secondary) < 1:
                raise ValueError(
                    f"There must be at least one connected context node to the value node: {node.id}"
                )
            elif len(value_sequence_secondary) == 1:
                return f"The value of the {secondary_tree_str} {value_str_secondary} is {node.value}."
            elif len(value_sequence_secondary) > 1:
                return f"The value of the {secondary_tree_str} combination {value_str_secondary} is {node.value}."
        elif len(value_sequence_primary) == 1:
            if len(value_sequence_secondary) < 1:
                return f"The value of the {primary_tree_str} {value_str_primary} is {node.value}."
            elif len(value_sequence_secondary) == 1:
                return f"The value of the {primary_tree_str} {value_str_primary} and the {secondary_tree_str} {value_str_secondary} is {node.value}."
            elif len(value_sequence_secondary) > 1:
                return f"The value of the {primary_tree_str} {value_str_primary} and the {secondary_tree_str} combination {value_str_secondary} is {node.value}."
        elif len(value_sequence_primary) > 1:
            if len(value_sequence_secondary) < 1:
                return f"The value of the {primary_tree_str} combination {value_str_primary} is {node.value}."
            elif len(value_sequence_secondary) == 1:
                return f"The value of the {primary_tree_str} combination {value_str_primary} and the {secondary_tree_str} {value_str_secondary} is {node.value}."
            elif len(value_sequence_secondary) > 1:
                return f"The value of the {primary_tree_str} combination {value_str_primary} and the {secondary_tree_str} combination {value_str_secondary} is {node.value}."

        raise ValueError(
            f"Invalid value sequence lengths: {len(value_sequence_primary)} and {len(value_sequence_secondary)}"
        )


class ValueStringGenerationTextAugmentedWithIntersection(ValueStringGeneration):
    approach: Optional[NodeApproach] = NodeApproach(
        approach=ValueNodeApproach.TEXT_AUGMENTED, include_context_intersection=True
    )

    def generate_value_string(
        self,
        node: ValueNode,
        primary_tree: TabTree,
        secondary_tree: TabTree,
    ) -> str:
        value_sequence_primary = (
            primary_tree.get_value_sequence_with_context_intersection(node)[:-1]
        )
        value_str_primary = StringGenerationService.sequence_of_node_sequence_to_string(
            value_sequence_primary,
            first_level_separator_approach=SeparatorApproach.COMMA,
            second_level_separator_approach=SeparatorApproach.AND,
        )
        value_sequence_secondary = (
            secondary_tree.get_value_sequence_with_context_intersection(node)[:-1]
        )
        value_str_secondary = (
            StringGenerationService.sequence_of_node_sequence_to_string(
                value_sequence_secondary,
                first_level_separator_approach=SeparatorApproach.COMMA,
                second_level_separator_approach=SeparatorApproach.AND,
            )
        )

        return ValueStringGenerationTextAugmented.text_augmented_approach_generation(
            node=node,
            value_sequence_primary=value_sequence_primary,
            value_str_primary=value_str_primary,
            value_sequence_secondary=value_sequence_secondary,
            value_str_secondary=value_str_secondary,
            primary_tree=primary_tree,
            secondary_tree=secondary_tree,
        )

from __future__ import annotations
from enum import Enum
from typing import Tuple, Type

from pydantic import BaseModel

from ....config.config import Config

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ....model.tabtree.primary_subtree_approach import PrimarySubtreeApproach


class NodeApproach(BaseModel):
    approach: ValueNodeApproach | ContextNodeApproach
    include_context_intersection: bool

    @classmethod
    def from_dict(
        cls: Type[NodeApproach], data: dict
    ) -> Tuple[NodeApproach, NodeApproach, PrimarySubtreeApproach]:
        from ....model.tabtree.primary_subtree_approach import PrimarySubtreeApproach

        """Return a tuple of two NodeApproach objects, first for the value node and second for the context node."""
        value_node_approach = ValueNodeApproach.from_str(data["value_string_approach"])
        context_node_approach = ContextNodeApproach.from_str(
            data["context_string_approach"]
        )
        value_node_approach_include_context_intersection = data[
            "value_string_with_context_intersection"
        ]
        context_node_approach_include_context_intersection = data[
            "context_string_with_context_intersection"
        ]
        
        primary_subtree_approach = PrimarySubtreeApproach.from_str(data.get("primary_subtree_approach"))


        # first context node, second value node
        return (
            cls(
                approach=context_node_approach,
                include_context_intersection=context_node_approach_include_context_intersection,
            ),
            cls(
                approach=value_node_approach,
                include_context_intersection=value_node_approach_include_context_intersection,
            ),
            primary_subtree_approach
        )

    @classmethod
    def from_config(cls: Type[NodeApproach], generation_cls) -> NodeApproach:
        from .context_string import ContextStringGeneration
        from .value_string import ValueStringGeneration

        if issubclass(generation_cls, ContextStringGeneration):
            approach = ContextNodeApproach.from_str(
                Config.evaluation.table_serialization_config.context_string_approach
            )
            include_context_intersection = (
                Config.evaluation.table_serialization_config.context_string_with_context_intersection
            )

        elif issubclass(generation_cls, ValueStringGeneration):
            approach = ValueNodeApproach.from_str(
                Config.evaluation.table_serialization_config.value_string_approach
            )
            include_context_intersection = (
                Config.evaluation.table_serialization_config.value_string_with_context_intersection
            )
        else:
            raise ValueError(f"Invalid approach: {generation_cls}")

        return cls(
            approach=approach, include_context_intersection=include_context_intersection
        )


class ContextNodeApproach(str, Enum):
    EMPTY = "context_empty"
    BASE = "context_base"
    TEXT = "context_text"

    @classmethod
    def from_str(cls, value: str) -> ContextNodeApproach:
        return ContextNodeApproach(value)


class ValueNodeApproach(str, Enum):
    BASE = "value_base"
    TEXT = "value_text"
    TEXT_AUGMENTED = "value_text_augmented"

    @classmethod
    def from_str(cls, value: str) -> ValueNodeApproach:
        return ValueNodeApproach(value)

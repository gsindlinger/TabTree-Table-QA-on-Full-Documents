from __future__ import annotations
import random
from typing import List, Optional
from pydantic import BaseModel


from .separator_approach import SeparatorApproach
from ..tabtree_model import CellNode

from .approaches import ContextNodeApproach, NodeApproach, ValueNodeApproach


class StringGenerationService(BaseModel):
    approach: Optional[NodeApproach] = None

    @classmethod
    def from_config(
        cls,
        approach: Optional[NodeApproach] = None,
    ) -> StringGenerationService:
        from .context_string import (
            ContextStringGenerationBase,
            ContextStringGenerationBaseWithIntersection,
            ContextStringGenerationText,
            ContextStringGenerationTextWithIntersection,
            ContextStringGenerationEmpty,
        )
        from .value_string import (
            ValueStringGenerationBase,
            ValueStringGenerationBaseWithIntersection,
            ValueStringGenerationText,
            ValueStringGenerationTextAugmented,
            ValueStringGenerationTextAugmentedWithIntersection,
            ValueStringGenerationTextWithIntersection,
        )

        if not approach:
            approach = NodeApproach.from_config(cls)

        match approach.approach:
            case ContextNodeApproach.BASE:
                if approach.include_context_intersection:
                    return ContextStringGenerationBaseWithIntersection()
                else:
                    return ContextStringGenerationBase()
            case ContextNodeApproach.TEXT:
                if approach.include_context_intersection:
                    return ContextStringGenerationTextWithIntersection()
                else:
                    return ContextStringGenerationText()
            case ContextNodeApproach.EMPTY:
                return ContextStringGenerationEmpty()
            case ValueNodeApproach.BASE:
                if approach.include_context_intersection:
                    return ValueStringGenerationBaseWithIntersection()
                else:
                    return ValueStringGenerationBase()
            case ValueNodeApproach.TEXT:
                if approach.include_context_intersection:
                    return ValueStringGenerationTextWithIntersection()
                else:
                    return ValueStringGenerationText()
            case ValueNodeApproach.TEXT_AUGMENTED:
                if approach.include_context_intersection:
                    return ValueStringGenerationTextAugmentedWithIntersection()
                else:
                    return ValueStringGenerationTextAugmented()
        raise ValueError(f"Invalid approach: {approach.approach}")

    @staticmethod
    def node_sequence_to_string(
        node_sequence: List[CellNode],
        separator_approach: SeparatorApproach = SeparatorApproach.COMMA,
    ) -> str:
        return separator_approach.sequence_to_string(node_sequence)

    @staticmethod
    def node_set_to_string(
        node_set: List[CellNode],
        separator_approach: SeparatorApproach = SeparatorApproach.COMMA,
    ) -> str:
        random.shuffle(node_set)
        return StringGenerationService.node_sequence_to_string(
            node_set, separator_approach
        )

    @staticmethod
    def sequence_of_node_sequence_to_string(
        sequence_of_node_sequence: List[List[CellNode]],
        first_level_separator_approach: SeparatorApproach = SeparatorApproach.COMMA,
        second_level_separator_approach: SeparatorApproach = SeparatorApproach.AND,
    ) -> str:
        sequence_str_list = [
            StringGenerationService.node_sequence_to_string(
                node_sequence, second_level_separator_approach
            )
            for node_sequence in sequence_of_node_sequence
        ]
        return first_level_separator_approach.string_sequence_to_single_string(
            sequence_str_list
        )

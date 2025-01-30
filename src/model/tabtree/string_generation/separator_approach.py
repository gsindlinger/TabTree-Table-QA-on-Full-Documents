from __future__ import annotations
from enum import Enum
from typing import List


from ..tabtree_model import CellNode


class SeparatorApproach(Enum):
    COMMA = "comma"
    COMMA_COLON = "comma_colon"
    COMMA_IS = "comma_is"
    HIERARCHY_COLON = "hierarchy_colon"
    AND_SYMBOL_COLON = "and_symbol_colon"
    COMMA_AND = "comma_and"
    AND = "and"
    HIERARCHY = "hierarchy"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

    def string_sequence_to_single_string(self, sequence: List[str]) -> str:
        if not sequence:
            raise ValueError("Sequence must contain at least one string.")
        if len(sequence) == 0:
            return ""
        if len(sequence) == 1:
            return sequence[0]

        match self:
            case SeparatorApproach.COMMA:
                return ", ".join(sequence)
            case SeparatorApproach.COMMA_COLON:
                return ", ".join(sequence[:-1]) + f": {sequence[-1]}"
            case SeparatorApproach.COMMA_IS:
                return ", ".join(sequence[:-1]) + f" is {sequence[-1]}"
            case SeparatorApproach.HIERARCHY_COLON:
                return " > ".join(sequence[:-1]) + f": {sequence[-1]}"
            case SeparatorApproach.AND_SYMBOL_COLON:
                return " & ".join(sequence[:-1]) + f": {sequence[-1]}"
            case SeparatorApproach.COMMA_AND:
                return ", ".join(sequence[:-1]) + f" and {sequence[-1]}"
            case SeparatorApproach.AND:
                return " & ".join(sequence)
            case SeparatorApproach.HIERARCHY:
                return " > ".join(sequence)
            case _:
                raise ValueError(f"Invalid separator approach: {self}")

    def sequence_to_string(self, sequence: List[CellNode]) -> str:
        return self.string_sequence_to_single_string([node.value for node in sequence])

    @classmethod
    def from_str(cls, value: str) -> SeparatorApproach:
        return SeparatorApproach(value)

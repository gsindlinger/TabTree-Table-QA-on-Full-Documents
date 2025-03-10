from __future__ import annotations
from enum import Enum
import random
from typing import Literal, Optional
from typing import TYPE_CHECKING

from ...config.config import Config
from .tabtree_model import NodeColor
from ...retrieval.document_preprocessors.table_parser.custom_table import CustomTableWithHeader

class PrimarySubtreeApproach(Enum):
    COLUMN_HEADER_TREE = "column_header_tree"
    ROW_LABEL_TREE = "row_label_tree"
    CONCATENATE = "concatenate"
    HEURISTIC = "heuristic"

    @classmethod
    def from_str(cls, approach: str | None) -> PrimarySubtreeApproach:
        match approach:
            case "column_header_tree":
                return cls.COLUMN_HEADER_TREE
            case "row_label_tree":
                return cls.ROW_LABEL_TREE
            case "concatenate":
                return cls.CONCATENATE
            case "heuristic":
                return cls.HEURISTIC
            case _:
                return cls.HEURISTIC # default

    @classmethod
    def from_table(
        cls, table: CustomTableWithHeader, approach: Optional[PrimarySubtreeApproach] = None
    ) -> NodeColor | Literal[PrimarySubtreeApproach.CONCATENATE]:
        if not approach:
            config_approach = cls.from_str(Config.tabtree.primary_subtree_approach)
        else:
            config_approach = approach
            
        match config_approach:
            case cls.COLUMN_HEADER_TREE:
                return NodeColor.YELLOW
            case cls.ROW_LABEL_TREE:
                return NodeColor.BLUE
            case cls.CONCATENATE:
                return cls.CONCATENATE
            case cls.HEURISTIC:
                if table.max_column_header_row > table.max_row_label_column:
                    return NodeColor.YELLOW
                elif table.max_column_header_row < table.max_row_label_column:
                    return NodeColor.BLUE
                else:
                    return random.choice([NodeColor.YELLOW, NodeColor.BLUE])
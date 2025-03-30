from __future__ import annotations
from pydantic import BaseModel
from typing import Dict, List, Literal, Optional, Tuple

from ...model.tabtree.primary_subtree_approach import PrimarySubtreeApproach
from ...model.tabtree.string_generation.approaches import NodeApproach
from ...config.config import Config


class PreprocessConfig(BaseModel):
    name: str
    preprocess_mode: List[
        Literal[
            "none",
            "basic",
            "remove-invisible",
            "remove-images",
            "remove-xbrl",
            "remove-attributes",
            "unwrap-irrelevant",
            "unwrap-divs",
            "replace-br",
            "normalize-whitespace",
            "only-text-except-tables",
            "delete-hr",
        ]
    ]
    reduced_sections: Optional[bool] = False
    ignore_tables_for_embeddings: Optional[bool] = False
    table_serialization: Optional[
        Literal[
            "none",
            "html",
            "csv",
            "tsv",
            "df-loader",
            "json-records",
            "json-split",
            "json-index",
            "markdown",
            "text",
            "text-bullet-points",
            "list-item",
            "matrix",
            "tabtree",
            "plain_text",
        ]
    ] = "none"
    consider_colspans_rowspans: Optional[bool] = False
    merge_sentence_infront_of_table: Optional[bool] = False
    tabtree_approach: Tuple[Optional[NodeApproach], Optional[NodeApproach], Optional[PrimarySubtreeApproach]] = (
        None,
        None,  # first context node and second value node
        None, # primary subtree approach
    )

    @classmethod
    def from_config(cls, config_data: Optional[Dict] = None) -> PreprocessConfig:
        if not config_data:
            preprocess_data = Config.evaluation.preprocess_config.copy()
            table_serializer_data = Config.evaluation.table_serialization_config
        else:
            preprocess_data = config_data["evaluation"]["preprocess_config"].copy()
            table_serializer_data = config_data["evaluation"][
                "table_serialization_config"
            ]

        table_serializer_data = table_serializer_data[0]
        return cls.from_config_single(preprocess_data, table_serializer_data)

    @classmethod
    def from_config_single(
        cls, preprocess_data: Dict, table_serializer_data: Dict
    ) -> PreprocessConfig:
        if not table_serializer_data or ("name" not in table_serializer_data):
            raise ValueError(
                "The config_data for table serialization must contain the field 'method' at least"
            )

        if (
            table_serializer_data["method"] == "tabtree"
            and "context_string_approach" in table_serializer_data
            and "context_string_with_context_intersection" in table_serializer_data
            and "value_string_approach" in table_serializer_data
            and "value_string_with_context_intersection" in table_serializer_data
        ):
            tabtree_approach = NodeApproach.from_dict(table_serializer_data)
            table_serialization = table_serializer_data.get("method")
        else:
            tabtree_approach = (None, None, None)
            table_serialization = table_serializer_data.get("method")

        # Ensure that the config_data contains the required fields
        if (
            not preprocess_data
            or "name" not in preprocess_data
            or "preprocess_mode" not in preprocess_data
        ):
            raise ValueError(
                "The config_data must contain the fields 'method' and 'preprocess_mode'"
            )

        obj = cls(
            table_serialization=table_serialization,
            tabtree_approach=tabtree_approach,
            **preprocess_data,
        )

        if table_serializer_data.get("name"):
            if obj.name.strip() != "":
                obj.name = f"{obj.name}-{table_serializer_data.get('name')}"
            else:
                obj.name = table_serializer_data.get("name", "Missing_Table_Serializer_Name")
        return obj

    @classmethod
    def from_config_multi(cls) -> List[PreprocessConfig]:
        preprocess_data = Config.evaluation.preprocess_config_multi
        table_serializer_data = Config.evaluation.table_serialization_config

        lst = []
        for preprocess_config in preprocess_data:
            for table_serializer_data_single in table_serializer_data:
                lst.append(
                    cls.from_config_single(preprocess_config, table_serializer_data_single)
                )
        return lst

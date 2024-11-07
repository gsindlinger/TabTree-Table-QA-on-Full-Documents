from __future__ import annotations
from pydantic import BaseModel
from typing import Dict, List, Literal, Optional

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
        ]
    ] = "none"
    consider_colspans_rowspans: Optional[bool] = False
    merge_sentence_infront_of_table: Optional[bool] = False

    @classmethod
    def from_config(cls, config_data: Optional[Dict] = None) -> PreprocessConfig:
        if not config_data:
            config_data = Config.evaluation.preprocess_config.copy()

        # Ensure that the config_data contains the required fields
        if (
            not config_data
            or "name" not in config_data
            or "preprocess_mode" not in config_data
        ):
            raise ValueError(
                "The config_data must contain the fields 'name' and 'preprocess_mode'"
            )

        return cls(**config_data)

    @classmethod
    def from_config_multi(cls) -> List[PreprocessConfig]:
        preprocess_configs = [
            cls.from_config(item) for item in Config.evaluation.preprocess_configs_multi
        ]
        return preprocess_configs

from abc import ABC
import glob
import os
from typing import Callable, List, Optional

from ..retrieval.document_preprocessors.table_parser.custom_table import (
    CustomTableWithHeaderOptional,
)
from ..config.config import Config
from ..retrieval.document_preprocessors.table_parser.custom_html_parser import (
    HTMLTableParser,
)


class AbstractTableTests(ABC):
    html_str_list: List[str] = []
    parsed_df: List[CustomTableWithHeaderOptional] = []
    custom_parser: HTMLTableParser = HTMLTableParser()

    def _common_setup(
        self, parse_method: Callable[[str], Optional["CustomTableWithHeaderOptional"]]
    ):
        self.html_str_list = []
        self.parsed_df = []

        html_file_path = Config.test.sample_table_html_path
        html_files = glob.glob(os.path.join(html_file_path, "*.html"))

        # sort by file name ascending
        html_files.sort()
        for file_path in html_files:
            with open(file_path, "r", encoding="utf-8") as file:
                self.html_str_list.append(file.read())

        for html_str in self.html_str_list:
            parsed_table = parse_method(html_str)
            if parsed_table is None:
                raise ValueError("The table couldn't be parsed.")
            self.parsed_df.append(parsed_table)

    def setup_only_parse(self):
        self._common_setup(parse_method=self.custom_parser.parse_table)

    def setup_parse_and_clean(self):
        self._common_setup(parse_method=self.custom_parser.parse_and_clean_table)

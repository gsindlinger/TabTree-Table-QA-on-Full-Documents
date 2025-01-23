from __future__ import annotations
import logging
from typing import List, Literal, Optional
from pandas import DataFrame
from pydantic import BaseModel

from ..table_serializer import ExtendedTable
from .custom_cell import CustomCell

CustomTable = List[List[CustomCell]]


class CustomTableWithHeader(BaseModel):
    table: CustomTable
    max_row_label_col: int
    max_column_header_row: int
    raw_table: str

    def __getitem__(self, key: int) -> List[CustomCell]:
        return self.table[key]

    def __setitem__(self, key: int, value: List[CustomCell]):
        self.table[key] = value

    def __len__(self):
        return len(self.table)

    @property
    def rows(self):
        return len(self.table)

    @property
    def columns(self):
        first_col_len = len(self.table[0])
        if any(len(row) != first_col_len for row in self.table):
            logging.warning("Table has inconsistent column lengths")
        return first_col_len

    def get_cell(self, row_index: int, column_index: int) -> CustomCell:
        return self.table[row_index][column_index]

    def get_column(self, column_index: int) -> List[CustomCell]:
        return [row[column_index] for row in self.table]

    def get_row(self, row_index: int) -> List[CustomCell]:
        return self.table[row_index]

    def to_extended_table(self, serialized_table: str) -> ExtendedTable:
        return ExtendedTable(
            df=DataFrame(self.table),
            raw_table=self.raw_table,
            has_header=True,
            header_columns=self.max_row_label_col,
            header_rows=self.max_column_header_row,
            serialized_table=serialized_table,
        )

    def set_headers(self, row_labels: int, column_headers: int):
        if self.max_column_header_row is not None:
            logging.info(
                "Column headers will be overriden: %s -> %s",
                self.max_column_header_row,
                column_headers,
            )
        if self.max_row_label_col is not None:
            logging.info(
                "Row labels will be overriden: %s -> %s",
                self.max_row_label_col,
                row_labels,
            )

        self.max_row_label_col = row_labels
        self.max_column_header_row = column_headers

    @staticmethod
    def print_line_with_span(
        items: List[CustomCell], orientation: Literal["row", "column"]
    ) -> str:
        line = []
        counter = 0
        while counter < len(items):
            if orientation == "row":
                span_high = items[counter].colspan[1]
                span_str = f" (colspan: {span_high+1})" if span_high > 0 else ""
            else:
                span_high = items[counter].rowspan[1]
                span_str = f" (rowspan: {span_high+1})" if span_high > 0 else ""

            line.append(f"{items[counter].value}{span_str}")
            counter += span_high + 1

        return str(line)

    def split_cells_on_headers(self) -> CustomTableWithHeader:
        j_star = self.max_row_label_col
        i_star = self.max_column_header_row
        for i, row in enumerate(self.table):
            for j, cell in enumerate(row):
                if j > j_star and i > i_star:  # Case: Both i > i* and j > j*
                    new_colspan = (0, 0)
                    new_rowspan = (0, 0)
                elif j > j_star and i <= i_star:  # Case: i <= i* and j > j*
                    new_colspan = (
                        min(j - j_star - 1, cell.colspan[0]),
                        cell.colspan[1],
                    )
                    new_rowspan = cell.rowspan
                elif j <= j_star and i <= i_star:  # Case: i <= i* and j <= j*
                    new_colspan = (cell.colspan[0], min(j_star - j, cell.colspan[1]))
                    new_rowspan = (cell.rowspan[0], min(i_star - i, cell.rowspan[1]))
                else:  # Case: i > i* and j <= j*
                    new_colspan = cell.colspan
                    new_rowspan = (
                        min(i - i_star - 1, cell.rowspan[0]),
                        cell.rowspan[1],
                    )

                # Create a new CustomCell with updated values
                cell.colspan = new_colspan
                cell.rowspan = new_rowspan
        return self


class CustomTableWithHeaderOptional(CustomTableWithHeader):
    max_row_label_col: Optional[int] = None
    max_column_header_row: Optional[int] = None

    def has_context(self) -> bool:
        if self.max_row_label_col is None or self.max_column_header_row is None:
            return False
        return True

    def to_extended_table(self, serialized_table: str) -> ExtendedTable:
        return ExtendedTable(
            df=DataFrame(self.table),
            raw_table=self.raw_table,
            has_header=self.has_context(),
            header_columns=self.max_row_label_col,
            header_rows=self.max_column_header_row,
            serialized_table=serialized_table,
        )

    def to_custom_table_with_header(self) -> CustomTableWithHeader:
        if not self.max_column_header_row or not self.max_row_label_col:
            raise ValueError("Table has no context.")
        return CustomTableWithHeader(
            table=self.table,
            max_row_label_col=self.max_row_label_col,
            max_column_header_row=self.max_column_header_row,
            raw_table=self.raw_table,
        )

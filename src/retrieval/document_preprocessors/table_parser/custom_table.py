from __future__ import annotations
import logging
from typing import List, Literal, Optional
from pandas import DataFrame
from pydantic import BaseModel

from ....model.tables import ExtendedTable
from .custom_cell import CustomCell

CustomTable = List[List[CustomCell]]


class CustomTableWithHeader(BaseModel):
    table: CustomTable
    max_row_label_column: int
    max_column_header_row: int
    raw_table: str

    def __getitem__(self, key: int) -> List[CustomCell]:
        return self.table[key]

    def __setitem__(self, key: int, value: List[CustomCell]):
        self.table[key] = value

    def __len__(self):
        return len(self.table)

    def has_no_headers(self) -> bool:
        return self.max_row_label_column == -1 and self.max_column_header_row == -1
    
    def to_html(self) -> str:
        html = "<table>\n"
        
        max_column_header_row = -1
                
        # Add table header (thead)
        if self.max_column_header_row is not None:
            max_column_header_row = self.max_column_header_row
        
        if max_column_header_row >= 0:
            html += "  <thead>\n"
            for i in range(self.max_column_header_row + 1):
                html += "    <tr>\n"
                for cell in self.table[i]:
                    if cell.colspan[0] == 0 and cell.rowspan[0] == 0:
                        colspan = ""
                        rowspan = ""
                        if cell.colspan[1] > 0:
                            colspan = f' colspan="{cell.colspan[1] + 1}"' if cell.colspan[1] > 0 else ""
                        if cell.rowspan[1] > 0:
                            rowspan = f' rowspan="{cell.rowspan[1] + 1}"' if cell.rowspan[1] > 0 else ""
                        html += f'      <th{colspan}{rowspan}>{cell.value}</th>\n'
                html += "    </tr>\n"
            html += "  </thead>\n"
        
        # Add table body (tbody)
        html += "  <tbody>\n"
        for i in range(max_column_header_row + 1, len(self.table)):
            html += "    <tr>\n"
            for cell in self.table[i]:
                if cell.colspan[0] == 0 and cell.rowspan[0] == 0:
                    colspan = ""
                    rowspan = ""
                    if cell.colspan[1] > 0:
                        colspan = f' colspan="{cell.colspan[1] + 1}"' if cell.colspan[1] > 0 else ""
                    if cell.rowspan[1] > 0:
                        rowspan = f' rowspan="{cell.rowspan[1] + 1}"' if cell.rowspan[1] > 0 else ""
                    html += f'      <td{colspan}{rowspan}>{cell.value}</td>\n'
            html += "    </tr>\n"
        html += "  </tbody>\n"
        
        html += "</table>"
        return html

    def to_csv(self, file_path: str, include_span: bool = False):
        with open(file_path, "w") as file:
            for row in self.table:
                if include_span:
                    file.write(
                        ";".join(
                            f"value: '{cell.value}', colspan: {cell.colspan}, rowspan: {cell.rowspan}"
                            for cell in row
                        )
                        + "\n"
                    )
                else:
                    file.write(";".join(cell.value for cell in row) + "\n")

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

    def get_cell_considering_span(
        self, row_index: int, column_index: int
    ) -> CustomCell:
        cell = self.get_cell(row_index, column_index)
        cell_span_considered = self.get_cell(
            row_index - cell.rowspan[0], column_index - cell.colspan[0]
        )
        return cell_span_considered

    def get_column(self, column_index: int) -> List[CustomCell]:
        return [row[column_index] for row in self.table]

    def get_row(self, row_index: int) -> List[CustomCell]:
        return self.table[row_index]

    def to_extended_table(self, serialized_table: str) -> ExtendedTable:
        return ExtendedTable(
            df=DataFrame(self.table),
            raw_table=self.raw_table,
            has_header=True,
            max_row_label_column=self.max_row_label_column,
            max_column_header_row=self.max_column_header_row,
            serialized_table=serialized_table,
        )

    def set_headers(
        self,
        max_column_header_row: int,
        max_row_label_column: int,
        override: bool = False,
    ) -> None:
        """Set maximum row label column and maximum column header row.

        Args:
            max_column_header_row (int): Maximum column header row.
            max_row_label_column (int): Maximum row label column.
        """

        if self.max_column_header_row is None or override:
            self.max_column_header_row = max_column_header_row
        else:
            logging.info(
                "Column headers will NOT be overridden. Keeping old value: %s, Potential new value: %s",
                self.max_column_header_row,
                max_column_header_row,
            )

        if self.max_row_label_column is None or override:
            self.max_row_label_column = max_row_label_column
        else:
            logging.info(
                "Row labels will NOT be overridden. Keeping old value: %s, Potential new value: %s",
                self.max_row_label_column,
                max_row_label_column,
            )

    @staticmethod
    def print_line_with_span(
        items: List[CustomCell], orientation: Literal["row", "column"]
    ) -> str:
        line = []
        counter = 0
        while counter < len(items):
            if orientation == "row":
                span_high = items[counter].colspan[1]
                span_str = (
                    f" (colspan: {span_high+1})"
                    if span_high > 0 and items[counter].value != ""
                    else ""
                )
            else:
                span_high = items[counter].rowspan[1]
                span_str = (
                    f" (rowspan: {span_high+1})"
                    if span_high > 0 and items[counter].value != ""
                    else ""
                )

            line.append(f"{items[counter].value}{span_str}")
            counter += span_high + 1

        return str(line)

    def split_cells_on_headers(self) -> CustomTableWithHeader:
        j_star = self.max_row_label_column
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

    def to_custom_table_with_header_optional(self) -> CustomTableWithHeaderOptional:
        return CustomTableWithHeaderOptional(
            table=self.table,
            max_row_label_column=self.max_row_label_column,
            max_column_header_row=self.max_column_header_row,
            raw_table=self.raw_table,
        )


class CustomTableWithHeaderOptional(CustomTableWithHeader):
    max_row_label_column: Optional[int] = None
    max_column_header_row: Optional[int] = None

    def has_context(self) -> bool:
        if self.max_row_label_column is None or self.max_column_header_row is None:
            return False
        return True

    def to_extended_table(self, serialized_table: str) -> ExtendedTable:
        return ExtendedTable(
            df=DataFrame(self.table),
            raw_table=self.raw_table,
            has_header=self.has_context(),
            max_row_label_column=self.max_row_label_column,
            max_column_header_row=self.max_column_header_row,
            serialized_table=serialized_table,
        )

    def to_custom_table_with_header(self) -> CustomTableWithHeader:
        if self.max_column_header_row is None or self.max_row_label_column is None:
            raise ValueError("Table has no context.")
        return CustomTableWithHeader(
            table=self.table,
            max_row_label_column=self.max_row_label_column,
            max_column_header_row=self.max_column_header_row,
            raw_table=self.raw_table,
        )

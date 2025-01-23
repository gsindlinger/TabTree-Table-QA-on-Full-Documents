from typing import Optional, Tuple
from pydantic import BaseModel


class CustomCell(BaseModel):
    """
    Represents a custom cell of a table for the usage within the TabTree model.

    Attributes:
        value (str): The value stored in the cell.
        row_index (int): The row index of the cell.
        col_index (int): The column index of the cell.
        colspan (Tuple[int, int]): The number of columns the cell spans, represented as a tuple.
                                    The first item represents the preceding columns which are also covered,
                                    the second item represents the following columns that are spanned by it.
                                    (start_column, end_column).
        rowspan (Tuple[int, int]): The number of rows the cell spans, represented as a tuple.
                                    The first item represents the preceding rows which are also covered,
                                    the second item represents the following rows that are spanned by it.
                                    (start_row, end_row).

    Methods:
        __repr__: Custom string representation of the `CustomCell` instance.
    """

    value: str
    row_index: int
    col_index: int
    colspan: Tuple[int, int]
    rowspan: Tuple[int, int]
    tag_name: Optional[str] = None

    def __repr__(self):
        return f"CustomCell(value={self.value}, colspan={self.colspan}, rowspan={self.rowspan})"

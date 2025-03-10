from typing import Optional, Tuple

from pandas import DataFrame
from pydantic import BaseModel


SerializedTable = Tuple[Optional[str], Optional[Tuple[int, int]]]


class DataFrameWithHeader(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    df: DataFrame
    has_header: bool
    max_row_label_column: Optional[int] = None
    max_column_header_row: Optional[int] = None

    def set_headers(self, max_column_header_row: int, max_row_label_column: int):
        """Set the headers of the table

        Args:
            max_column_header_row (int): Maximum row index of the column header
            max_row_label_column (int): Maximum column index of the row label
        """
        self.has_header = True
        self.max_column_header_row = max_column_header_row
        self.max_row_label_column = max_row_label_column
        
        
class ExtendedTable(DataFrameWithHeader):
    raw_table: str
    serialized_table: Optional[str] = None

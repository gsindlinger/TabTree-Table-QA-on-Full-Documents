from typing import Any, List, Literal, Optional, Tuple
from bs4 import BeautifulSoup, NavigableString, Tag
from pandas import DataFrame
from pydantic import BaseModel


class CustomCell(BaseModel):
    value: str
    colspan: Optional[Tuple[int, int]] = None
    rowspan: Optional[Tuple[int, int]] = None
    tag_name: Optional[str] = None

    def __repr__(self):
        if self.colspan and self.rowspan:
            return f"CustomCell(value={str(self.value)}, colspan={str(self.colspan)}, rowspan={str(self.rowspan)})"
        elif self.colspan:
            return f"CustomCell(value={str(self.value)}, colspan={str(self.colspan)})"
        elif self.rowspan:
            return f"CustomCell(value={str(self.value)}, rowspan={str(self.rowspan)})"
        else:
            return f"CustomCell(value={str(self.value)})"


type CustomTable = List[List[CustomCell]]


class HTMLTableParser(BaseModel):
    @staticmethod
    def parse_table(html: str) -> DataFrame:
        """Parse an HTML table into a pandas DataFrame. Assuming a single table in the provided HTML string."""
        p = HTMLTableParser()
        return p.custom_parse_html(html)

    def _parse_thead_tr(self, table: Tag) -> CustomTable:
        header = []
        thead = table.find("thead")
        if thead:
            for row in thead.find_all("tr"):  # type: ignore
                header.append(self._parse_row(row))
        return header

    def _parse_tbody_tr(self, table: Tag) -> CustomTable:
        body = []
        tbody = table.find("tbody")
        if tbody:
            for row in tbody.find_all("tr"):  # type: ignore
                body.append(self._parse_row(row))
        return body

    def _parse_tfoot_tr(self, table: Tag) -> CustomTable:
        footer = []
        tfoot = table.find("tfoot")
        if tfoot:
            for row in tfoot.find_all("tr"):  # type: ignore
                footer.append(self._parse_row(row))
        return footer

    def custom_parse_html(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        # Find table
        table = soup.find("table")
        if not table:
            raise ValueError("No table found in the provided HTML.")
        elif isinstance(table, NavigableString):
            raise ValueError("Table is not a tag.")

        thead = self._parse_thead_tr(table)
        tbody = self._parse_tbody_tr(table)
        tfoot = self._parse_tfoot_tr(table)

        def _row_is_all_th(row: List[CustomCell]):
            """Check if a row contains only <th> elements."""
            return all(cell.tag_name == "th" for cell in row)

        if not thead:
            # The table has no <thead>. Move the top all-<th> rows from
            # body_rows to header_rows. (This is a common case because many
            # tables in the wild have no <thead> or <tfoot>
            while tbody and _row_is_all_th(tbody[0]):
                thead.append(tbody.pop(0))

        table = thead + tbody + tfoot
        table = self._expand_colspan(table)
        table = self._expand_rowspan(table)

        return DataFrame(table)

    def delete_nan_columns_and_rows(self, table: CustomTable) -> CustomTable:
        for row_index, row in enumerate(table):
            if all(cell.value == "" for cell in row):
                table = self.delete_row(table, row_index)

        table_copy = table.copy()
        for column_index, column in enumerate(zip(*table_copy)):
            if all(cell.value == "" for cell in column):
                table = self.delete_column(table, column_index)

        return table

    def delete_row(self, table: CustomTable, row_index: int) -> CustomTable:
        table = self._update_rowspan(table, row_index, type="rowspan")
        del table[row_index]
        return table

    def delete_column(self, table: CustomTable, column_index: int) -> CustomTable:
        table = list(zip(*table))
        table = self._update_rowspan(table, column_index, type="colspan")
        del table[column_index]
        table = list(zip(*table))
        return table

    def delete_duplicate_columns_and_rows(self, table: CustomTable) -> CustomTable:
        # find duplicate rows

        ### TODO: Implement this function
        for row_index, row in enumerate(table):
            if all(cell.value == "" for cell in row):
                table = self.delete_row(table, row_index)

        table_copy = table.copy()
        for column_index, column in enumerate(zip(*table_copy)):
            if all(cell.value == "" for cell in column):
                table = self.delete_column(table, column_index)

    def _update_rowspan(
        self, table: CustomTable, row_index: int, type: Literal["rowspan", "colspan"]
    ) -> CustomTable:
        for column_index, cell in enumerate(table[row_index]):
            span_item = cell.rowspan if type == "rowspan" else cell.colspan
            if span_item:
                for i in range(1, span_item[0]):
                    previous_span_item = table[row_index - i][column_index]
                    if type == "rowspan":
                        previous_span = previous_span_item.rowspan
                        if previous_span:
                            previous_span = (
                                previous_span[0],
                                max(0, previous_span[1] - 1),
                            )

                        if previous_span == (0, 0):
                            previous_span_item.rowspan = None
                    elif type == "colspan":
                        previous_span = previous_span_item.colspan
                        if previous_span:
                            previous_span = (
                                previous_span[0],
                                max(0, previous_span[1] - 1),
                            )
                        if previous_span == (0, 0):
                            previous_span_item.colspan = None

                for i in range(1, span_item[1] + 1):
                    previous_span_item = table[row_index + i][column_index]
                    if type == "rowspan":
                        previous_span = previous_span_item.rowspan
                        if previous_span:
                            previous_span = (
                                max(0, previous_span[0] - 1),
                                previous_span[1],
                            )
                        if previous_span == (0, 0):
                            previous_span_item.rowspan = None

                    if type == "colspan":
                        previous_span = previous_span_item.colspan
                        if previous_span:
                            previous_span = (
                                max(0, previous_span[0] - 1),
                                previous_span[1],
                            )
                        if previous_span == (0, 0):
                            previous_span_item.colspan = None
        return table

    def _parse_row(self, row):
        parsed_row = []
        for cell in row.find_all(["td", "th"]):
            value = cell.get_text(strip=True)

            # col and rowspans refer to the number of covered cells (before, after) the current cell
            # e.g. if a cell has a colspan of 2, the colspan tuple will be (0, 1)
            colspan = cell.attrs.get("colspan")
            colspan = (0, int(colspan) - 1) if colspan else None
            rowspan = cell.attrs.get("rowspan")
            rowspan = (0, int(rowspan) - 1) if rowspan else None

            parsed_row.append(
                CustomCell(
                    value=value, colspan=colspan, rowspan=rowspan, tag_name=cell.name
                )
            )
        return parsed_row

    def _expand_colspan_rowspan(
        self, table: List[List[CustomCell]]
    ) -> List[List[CustomCell]]:
        for row in table:
            index = 0
            while index < len(row):
                cell = row[index]
                if cell.colspan:
                    _, end = cell.colspan
                    # col and rowspans refer to the number of covered cells (before, after) the current cell
                    # e.g. if a cell has a colspan of 2, the colspan tuple will be (0, 1)
                    for i in range(1, end + 1):
                        row[index + i].colspan = (i, end - i)
                index += end
        return table

    def _expand_colspan(self, table: List[List[CustomCell]]) -> List[List[CustomCell]]:
        return self._expand_colspan_rowspan(table)

    def _expand_rowspan(self, table: List[List[CustomCell]]) -> List[List[CustomCell]]:
        if not all(len(row) == len(table[0]) for row in table):
            raise ValueError("All rows must have the same number of columns.")
        transposed_table = list(zip(*table))
        return list(zip(*self._expand_colspan_rowspan(transposed_table)))

from __future__ import annotations
from typing import Optional, Tuple, override

from ...model.tabtree.primary_subtree_approach import PrimarySubtreeApproach
from ...retrieval.document_loaders.document_loader import DocumentLoader

from .string_generation.approaches import NodeApproach
from ...retrieval.document_preprocessors.table_parser.custom_table import (
    CustomTableWithHeader,
)

from .tabtree_model import NodeColor

from ...retrieval.document_preprocessors.table_parser.custom_html_parser import (
    HTMLTableParser,
)
from .tabtree_service import FullTabTree, TabTreeService
from ...retrieval.document_preprocessors.table_serializer import (
    ExtendedTable,
    ExtendedTable,
    SerializedTable,
    TableSerializer,
)


class TabTreeSerializer(TableSerializer):
    table_splitter_backup: str = r"\n"
    node_approach: Tuple[Optional[NodeApproach], Optional[NodeApproach], Optional[PrimarySubtreeApproach]] = (
        None,
        None,  # first for the value node and second for the context node
        None
    )

    @override
    def serialize_table_to_str(
        self,
        table_str: str,
    ) -> SerializedTable:
        """Returns the serialized table and the headers (max_column_header_row, max_row_label_column) if they were detected"""
        document_loader = DocumentLoader.from_config()
        (max_column_header_row, max_row_label_column) = (
            document_loader.get_header_from_table_string(table_str)
        )

        if max_column_header_row is not None and max_row_label_column is not None:
            html_table = HTMLTableParser.parse_and_clean_table(table_str)
            if not html_table:
                return "", None
            html_table.set_headers(
                max_column_header_row=max_column_header_row,
                max_row_label_column=max_row_label_column,
            )
            return self.serialize_with_headers(html_table), (
                max_column_header_row,
                max_row_label_column,
            )

        else:
            table = self.serialize_table_to_extended_table(table_str)
            if not table:
                return "", None
            if (
                table.max_column_header_row is not None
                and table.max_row_label_column is not None
            ):
                document_loader.write_header_back_to_file(
                    table_str, table.max_column_header_row, table.max_row_label_column
                )
                return (
                    table.serialized_table,
                    (table.max_column_header_row, table.max_row_label_column),
                )
            else:
                return (table.serialized_table, None)

    @override
    def serialize_table_to_extended_table(self, table_str: str) -> ExtendedTable | None:
        html_table = HTMLTableParser.parse_and_clean_table(table_str)
        if not html_table:
            return None

        # if we couldn't detect any headers we just provide the raw table as the serialized table
        html_table_with_headers = TabTreeService.set_headers(html_table)
        serialized_table = self.serialize_with_headers(html_table_with_headers)
        return html_table.to_extended_table(serialized_table=serialized_table)

    def serialize_with_headers(
        self, html_table_with_headers: CustomTableWithHeader
    ) -> str:

        if html_table_with_headers.has_no_headers():
            # provide the data in matrix mode if no headers were specified
            serialized_table = self.table_serialization_backup(html_table_with_headers)
        else:
            # preprocess again using headers
            parser = HTMLTableParser()
            html_table_with_headers = (
                html_table_with_headers.to_custom_table_with_header_optional()
            )
            html_table_with_headers = parser.delete_and_reset_columns_and_rows(
                html_table_with_headers, consider_headers=True
            )
            html_table_with_headers = (
                html_table_with_headers.to_custom_table_with_header()
            )

            # generate tabtree
            full_tab_tree = TabTreeService.generate_full_tabtree(
                html_table_with_headers
            )
            serialized_table = self.serialize_by_primary_subtree_approach(
                html_table_with_headers, full_tab_tree
            )
        return serialized_table

    def serialize_by_primary_subtree_approach(
        self, html_table_with_headers: CustomTableWithHeader, full_tab_tree: FullTabTree
    ) -> str:
        primary_subtree_approach = PrimarySubtreeApproach.from_table(html_table_with_headers, self.node_approach[2])
        if isinstance(primary_subtree_approach, NodeColor):
            primary_colour = primary_subtree_approach
            
        if primary_subtree_approach == PrimarySubtreeApproach.CONCATENATE:
            serialized_table_column_header = TabTreeService.generate_serialized_string(
                full_tab_tree,
                primary_colour=NodeColor.YELLOW,
                approaches=self.node_approach[:2],
            )
            serialized_table_row_label = TabTreeService.generate_serialized_string(
                full_tab_tree,
                primary_colour=NodeColor.BLUE,
                approaches=self.node_approach[:2],
            )

            return f"""
                The table is now presented in two different ways.
                The first one is built upon column headers and the second one is built upon row labels. 
                Please consider that the presented table data values are same in both representations. 
                \n\n
                {serialized_table_column_header}
                \n\n
                {serialized_table_row_label}
                """
        else:
            serialized_table = TabTreeService.generate_serialized_string(
                full_tab_tree,
                primary_colour=primary_colour,
                approaches=self.node_approach[:2],
            )
        return serialized_table

    def df_to_serialized_string(self, df_with_header: ExtendedTable) -> str:
        extended_table = self.serialize_table_to_extended_table(
            df_with_header.raw_table
        )
        if not extended_table or not extended_table.serialized_table:
            return ""
        else:
            return extended_table.serialized_table

    def table_serialization_backup(self, table: CustomTableWithHeader) -> str:
        """Returns the table in matrix mode if no headers were detected"""
        rows = [
            "[" + ", ".join(cell.value for cell in row) + "]" for row in table.table
        ]
        return "[\n" + ",\n".join(rows) + "\n]"

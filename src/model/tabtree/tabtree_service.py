from __future__ import annotations
from typing import Literal

from pydantic import BaseModel
from ...pipeline import TableHeaderRowsPipeline
from ...retrieval.document_preprocessors.table_parser.custom_table import (
    CustomTableWithHeader,
    CustomTableWithHeaderOptional,
)
from .tabtree_model import (
    ColouredNode,
    ColumnHeaderNode,
    ColumnHeaderTreeRoot,
    ContextIntersectionNode,
    TabTree,
    ValueNode,
    RowLabelTreeRoot,
    RowLabelNode,
)


class FullTabTree(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    column_header_tree: TabTree
    row_label_tree: TabTree
    custom_table: CustomTableWithHeader


class TabTreeService(BaseModel):
    """Assuming empty rows and columns are already deleted."""

    @staticmethod
    def generate_full_tabtree(
        custom_table_header_optional: CustomTableWithHeaderOptional,
    ) -> FullTabTree:
        service = TabTreeService()
        if not custom_table_header_optional.has_context():
            row_header_max_index, col_header_max_index = service.get_headers(
                custom_table_header_optional
            )
            custom_table_header_optional.set_headers(
                row_header_max_index, col_header_max_index
            )

        custom_table = custom_table_header_optional.to_custom_table_with_header()

        # preprocess as in Algoithm 3.1
        custom_table = service.preprocess_table(custom_table)

        # generate column header tree as in Algorithm 3.2
        column_header_tree = service.generate_tree(custom_table, orientation="column")

        # generate row label tree as in Algorithm 3.3
        row_label_tree = service.generate_tree(custom_table, orientation="row")

        return FullTabTree(
            custom_table=custom_table,
            column_header_tree=column_header_tree,
            row_label_tree=row_label_tree,
        )

    def preprocess_table(
        self, custom_table: CustomTableWithHeader
    ) -> CustomTableWithHeader:
        return custom_table.split_cells_on_headers()

    def get_headers(
        self, custom_table: CustomTableWithHeaderOptional
    ) -> tuple[int, int]:
        """Get the row and column headers of the table.
        First item of return tuple refers to column header rows and second item refers to row label columns.
        """
        table_header_detection = TableHeaderRowsPipeline.from_config()
        return table_header_detection.predict_headers(custom_table)

    def generate_tree(
        self, custom_table: CustomTableWithHeader, orientation: Literal["column", "row"]
    ) -> TabTree:
        tree = TabTree()
        if orientation == "column":
            # Step 1: Add root node
            parent_node = ColumnHeaderTreeRoot()
        elif orientation == "row":
            parent_node = RowLabelTreeRoot()

        tree.add_node(parent_node)

        range_max = (
            custom_table.max_column_header_row
            if orientation == "column"
            else custom_table.max_row_label_col
        )
        # Step 2: Iterate over all column header cells / row label cells for distinct column headers / row labels
        for current_index in range(range_max + 1):
            self.add_context_nodes(
                current_index=current_index,
                parent_node=parent_node,
                custom_table=custom_table,
                tree=tree,
                orientation=orientation,
            )
            self.add_context_intersection_nodes(
                current_index=current_index,
                custom_table=custom_table,
                tree=tree,
                orientation=orientation,
            )

            # Step 3: Add value cells
            self.add_value_cells(custom_table, tree, orientation)

        return tree

    def add_value_cells(
        self,
        custom_table: CustomTableWithHeader,
        tree: TabTree,
        orientation: Literal["column", "row"],
    ) -> None:
        for row_index in range(
            custom_table.max_column_header_row + 1, len(custom_table.table)
        ):
            for col_index in range(
                custom_table.max_row_label_col + 1, len(custom_table.table[0])
            ):
                new_cell = custom_table.get_cell(row_index, col_index)

                if orientation == "column":
                    parent_cell = custom_table.get_cell_considering_span(
                        custom_table.max_column_header_row, col_index
                    )
                    tree.add_edge(
                        ColumnHeaderNode.from_custom_cell(parent_cell),
                        ValueNode.from_custom_cell(new_cell),
                    )
                else:
                    parent_cell = custom_table.get_cell_considering_span(
                        row_index, custom_table.max_row_label_col
                    )
                    tree.add_edge(
                        RowLabelNode.from_custom_cell(parent_cell),
                        ValueNode.from_custom_cell(new_cell),
                    )

    def add_context_intersection_nodes(
        self,
        current_index: int,
        custom_table: CustomTableWithHeader,
        tree: TabTree,
        orientation: Literal["column", "row"],
    ) -> None:

        if orientation == "column":
            # if no context-intersection cells are present, return
            if custom_table.max_row_label_col == 0:
                return

            col_index = 0
            while True:
                cell_left = custom_table.get_cell_considering_span(
                    current_index, col_index
                )
                col_index += cell_left.colspan[1] + 1
                # Break if the next cell is out of bounds
                if col_index > custom_table.max_row_label_col:
                    break

                # Else add edge to next right cell
                cell_right = custom_table.get_cell_considering_span(
                    current_index, col_index
                )
                tree.add_edge(
                    ContextIntersectionNode.from_custom_cell(cell_left),
                    ContextIntersectionNode.from_custom_cell(cell_right),
                )

            # Retrieve all column header nodes for the current row and connect them to the most left
            column_header_nodes = tree.get_column_nodes_by_row_index(
                row_index=current_index,
                start_col_index=custom_table.max_row_label_col + 1,
            )
            for node in column_header_nodes:
                tree.add_edge(ContextIntersectionNode.from_custom_cell(cell_left), node)

        if orientation == "row":
            # if no context-intersection cells are present, return
            if custom_table.max_column_header_row == 0:
                return

            row_index = 0
            while True:
                cell_top = custom_table.get_cell_considering_span(
                    row_index, current_index
                )
                row_index += cell_top.rowspan[1] + 1
                # Break if the next cell is out of bounds
                if row_index > custom_table.max_column_header_row:
                    break

                cell_bottom = custom_table.get_cell_considering_span(
                    row_index, current_index
                )
                tree.add_edge(
                    ContextIntersectionNode.from_custom_cell(cell_top),
                    ContextIntersectionNode.from_custom_cell(cell_bottom),
                )

            # Retrieve all row label nodes for the current column and connect them to the most top
            row_label_nodes = tree.get_row_nodes_by_column_index(
                column_index=current_index,
                start_row_index=custom_table.max_column_header_row + 1,
            )
            for node in row_label_nodes:
                tree.add_edge(ContextIntersectionNode.from_custom_cell(cell_top), node)

    def add_context_nodes(
        self,
        current_index: int,
        parent_node: ColouredNode | None,
        custom_table: CustomTableWithHeader,
        tree: TabTree,
        orientation: Literal["column", "row"],
    ) -> None:
        # Range starts at max_row_label_col + 1 because context-intersection is excluded
        if orientation == "column":
            row_index = current_index
            col_index = custom_table.max_row_label_col + 1
            while col_index < len(custom_table.table[row_index]):
                new_cell = custom_table.get_cell(row_index, col_index)
                if new_cell.colspan[0] == 0:
                    # for first row parent node is root, for all others connect to previous row
                    if row_index != 0:
                        # parent node is given as the cell in previous row minus its colspan to the left
                        previous_cell = custom_table.get_cell_considering_span(
                            row_index - 1, col_index
                        )
                        parent_node = tree.get_node_by_cell(previous_cell)
                    if not parent_node:
                        raise ValueError("Parent node not found.")
                    # add new cell as node
                    tree.add_edge(
                        parent_node, ColumnHeaderNode.from_custom_cell(new_cell)
                    )
                col_index += new_cell.colspan[1] + 1

        else:
            col_index = current_index
            row_index = custom_table.max_column_header_row + 1
            while row_index < len(custom_table.table):
                new_cell = custom_table.get_cell(row_index, col_index)
                if new_cell.rowspan[0] == 0:
                    # for first column parent node is root, for all others connect to previous column
                    if col_index != 0:
                        # parent node is given as the cell in previous column minus its rowspan to the top
                        previous_cell = custom_table.get_cell_considering_span(
                            row_index, col_index - 1
                        )
                        parent_node = tree.get_node_by_cell(previous_cell)
                    if not parent_node:
                        raise ValueError("Parent node not found.")
                    # add new cell as node
                    tree.add_edge(parent_node, RowLabelNode.from_custom_cell(new_cell))
                row_index += new_cell.rowspan[1] + 1

    def generate_serialized_string(self) -> str:
        return ""

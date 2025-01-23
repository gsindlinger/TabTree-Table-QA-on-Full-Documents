import copy
import glob
import os
from typing import List
import unittest

from .testutils import AbstractTableTests
from ..retrieval.document_preprocessors.table_parser.custom_table import (
    CustomTableWithHeaderOptional,
)
from ..retrieval.document_preprocessors.table_parser.custom_cell import CustomCell


class TestHTMLTableParsing(unittest.TestCase, AbstractTableTests):

    def setUp(self):
        self.setup_only_parse()

    def test_parse_valid_html(self):
        """Test parsing a valid HTML table."""
        for df in self.parsed_df:
            self.assertIsInstance(df, CustomTableWithHeaderOptional)
            self.assertGreater(len(df), 0, "The Table should not be empty.")
            self.assertIsInstance(
                df.get_cell(0, 0),
                CustomCell,
                "The cell should be an instance of CustomCell.",
            )

    def test_value_correctly_parsed(self):
        """Test if the values are correctly parsed"""
        df = self.parsed_df[0]
        cell_1 = df.get_cell(0, 6)
        cell_2 = df.get_cell(1, 3)
        cell_3 = df.get_cell(4, 0)
        cell_4 = df.get_cell(0, 7)

        self.assertEqual(cell_1.value, "grades")  # type: ignore
        self.assertEqual(cell_2.value, "2023")  # type: ignore
        self.assertEqual(cell_3.value, "A")  # type: ignore
        self.assertEqual(cell_4.value, "")  # type: ignore

    def test_parse_rowspan_correctly(self):
        """Test if the colspan and rowspan are correctly parsed"""
        df = self.parsed_df[0]
        cell_1 = df.get_cell(0, 0)
        cell_2 = df.get_cell(1, 3)
        cell_3 = df.get_cell(4, 0)

        if not all(isinstance(cell, CustomCell) for cell in [cell_1, cell_2, cell_3]):
            self.fail("The cell should be an instance of CustomCell.")

        self.assertEqual(cell_1.colspan, (0, 6))  # type: ignore
        self.assertEqual(cell_1.rowspan, (0, 0))  # type: ignore

        self.assertEqual(cell_2.colspan, (0, 1))  # type: ignore
        self.assertEqual(cell_2.rowspan, (0, 0))  # type: ignore

        self.assertEqual(cell_3.colspan, (0, 0))  # type: ignore
        self.assertEqual(cell_3.rowspan, (1, 1))  # type: ignore

    def test_delete_rows(self):
        """Test if the rows are correctly deleted"""
        df = self.parsed_df[0]
        modified_table = self.custom_parser.delete_row(df.table, 3)

        self.assertEqual(len(modified_table), 6)

        cell_1 = modified_table[0][0]
        cell_2 = modified_table[4][0]

        self.assertEqual(cell_1.value, "grades")
        self.assertEqual(cell_2.value, "A")

        self.assertEqual(cell_1.rowspan, (0, 0))
        self.assertEqual(cell_2.rowspan, (1, 0))

    def test_delete_column(self):
        """Test if the columns are correctly deleted"""
        df = self.parsed_df[0]
        modified_table = self.custom_parser.delete_column(df.table, 3)

        self.assertTrue(all(len(row) == 7 for row in modified_table))

        cell_1 = modified_table[0][0]
        cell_2 = modified_table[1][3]
        cell_3 = modified_table[1][5]

        self.assertEqual(cell_1.value, "grades")
        self.assertEqual(cell_2.value, "2023")
        self.assertEqual(cell_3.value, "2024")

        self.assertEqual(cell_1.colspan, (0, 5))
        self.assertEqual(cell_2.colspan, (0, 0))
        self.assertEqual(cell_3.colspan, (1, 0))

    def test_delete_empty_column_row(self):
        """Test if the empty columns are correctly deleted"""
        table_before = self.parsed_df[1]
        table_before_copy = copy.deepcopy(table_before)
        table_after = self.custom_parser.delete_empty_columns_and_rows(
            table_before.table
        )

        self.assertEqual(len(table_after), len(table_before_copy) - 1)
        self.assertEqual(len(table_after[0]), len(table_before_copy[0]) - 1)

    def test_delete_duplicate_column_row(self):
        """Test if the duplicate columns are correctly deleted"""
        table_before = self.parsed_df[1]
        table_before_copy = copy.deepcopy(table_before)
        table_after = self.custom_parser.delete_duplicate_columns_and_rows(table_before)

        self.assertEqual(len(table_after), len(table_before_copy) - 1)
        self.assertEqual(len(table_after[0]), len(table_before_copy[0]) - 1)

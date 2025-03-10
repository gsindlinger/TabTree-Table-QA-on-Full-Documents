import copy
import glob
import os
from typing import List
import unittest

from ..model.tabtree.tabtree_service import TabTreeService

from ..retrieval.document_preprocessors.table_parser.custom_table import (
    CustomTableWithHeaderOptional,
)
from ..retrieval.document_preprocessors.table_parser.custom_html_parser import (
    HTMLTableParser,
)
from ..config.config import Config
from .testutils import AbstractTableTests


class TestHeaderDetection(unittest.TestCase, AbstractTableTests):
    """Test the header detection of the table.
    Be aware that running this test uses a pipeline call, i.e., potentially calling API endpoints.
    """

    def setUp(self):
        self.setup_parse_and_clean()

    def test_header_detection(self):
        header_detection = TabTreeService().get_headers(self.parsed_df[0])
        self.assertTrue(header_detection[0] != -1)
        self.assertTrue(header_detection[1] != -1)

        # ideally we obtain the max column header 2 and max row header 0
        self.assertEqual(header_detection[0], 2)
        self.assertEqual(header_detection[1], 0)

    def test_header_detection_awk_bug(self):
        """Test the header detection with the awk bug."""
        header_detection = TabTreeService().get_headers(self.parsed_df[10])
        self.assertTrue(header_detection[0] != -1)
        self.assertTrue(header_detection[1] != -1)

        # ideally we obtain the max column header 1 and max row header 0
        self.assertEqual(header_detection[0], 0)
        self.assertEqual(header_detection[1], 0)

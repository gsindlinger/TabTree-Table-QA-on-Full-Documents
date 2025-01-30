import unittest
from matplotlib import pyplot as plt

from ..model.tabtree.string_generation.approaches import (
    ContextNodeApproach,
    NodeApproach,
)
from ..model.tabtree.string_generation.context_string import ContextStringGeneration
from ..model.tabtree.string_generation.general import StringGenerationService
from ..model.tabtree.string_generation.separator_approach import SeparatorApproach
from ..model.tabtree.tabtree_service import TabTreeService

from .testutils import AbstractTableTests


class TestTabTreeModel(unittest.TestCase, AbstractTableTests):

    def setUp(self):
        self.setup_parse_and_clean()
        self.tabtree_service = TabTreeService()
        df = self.parsed_df[3]
        df.set_headers(1, 2)

        # generate model / find tests for this in another file
        self.full_tabtree = self.tabtree_service.generate_full_tabtree(
            self.parsed_df[3]
        )

    def test_sequence_string_representation(self):

        # Arrange
        sequence = [
            self.full_tabtree.row_label_tree.get_node_by_index(0, 0),
            self.full_tabtree.row_label_tree.get_node_by_index(2, 1),
            self.full_tabtree.row_label_tree.get_node_by_index(3, 1),
        ]

        # Act
        sequence_str = StringGenerationService.node_sequence_to_string(
            node_sequence=sequence  # type: ignore
        )

        # Assert
        self.assertEqual(sequence_str, "grades, name, John")

    def test_sequence_string_representation_with_separator(self):
        # Arrange
        sequence = [
            self.full_tabtree.row_label_tree.get_node_by_index(0, 0),
            self.full_tabtree.row_label_tree.get_node_by_index(2, 1),
            self.full_tabtree.row_label_tree.get_node_by_index(3, 1),
        ]

        # Act
        sequence_str = StringGenerationService.node_sequence_to_string(
            node_sequence=sequence, separator_approach=SeparatorApproach.COMMA_IS  # type: ignore
        )

        # Assert
        self.assertEqual(sequence_str, "grades, name is John")

    def test_context_string_base_correct(self):
        # Arrange
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(0, 2)  # grades
        node_2 = self.full_tabtree.column_header_tree.get_node_by_index(
            2, 5
        )  # 2024 / math

        approach = NodeApproach(
            approach=ContextNodeApproach.BASE, include_context_intersection=False
        )

        # Act
        context_str_1 = ContextStringGeneration.generate_string(
            node_1, self.full_tabtree.column_header_tree, approach  # type: ignore
        )
        context_str_2 = ContextStringGeneration.generate_string(
            node_2, self.full_tabtree.column_header_tree, approach  # type: ignore
        )

        # Assert
        self.assertEqual(context_str_1, "grades:")
        self.assertEqual(context_str_2, "math:")

    def test_context_string_non_existant_node(self):
        # Arrange
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(
            0, 0
        )  # context-intersection node
        node_2 = self.full_tabtree.column_header_tree.get_node_by_index(
            0, 1
        )  # non-existant node
        node_3 = self.full_tabtree.column_header_tree.get_node_by_index(
            3, 0
        )  # row label node

        approach = NodeApproach(
            approach=ContextNodeApproach.BASE, include_context_intersection=False
        )

        # Act & Assert
        with self.assertRaises(ValueError):
            ContextStringGeneration.generate_string(
                node_1, self.full_tabtree.column_header_tree, approach  # type: ignore
            )
        with self.assertRaises(ValueError):
            ContextStringGeneration.generate_string(
                node_2, self.full_tabtree.column_header_tree, approach  # type: ignore
            )
        with self.assertRaises(ValueError):
            ContextStringGeneration.generate_string(
                node_3, self.full_tabtree.column_header_tree, approach  # type: ignore
            )

    def test_context_string_text(self):
        # Arrange

        # Test case for each combination of siblings and children (see line 12, algorithm 3.7)
        # siblings > 0 & children > 0
        node_1 = self.full_tabtree.row_label_tree.get_node_by_index(
            3, 0
        )  # A, siblings: B, children: John, Tiffany
        expected_string_1 = (
            "The row label A has siblings B. The children of A are John, Tiffany."
        )

        # siblings = 0 & children > 0
        node_2 = self.full_tabtree.column_header_tree.get_node_by_index(
            0, 2
        )  # grades, children: null, 2023, 2024
        expected_string_2 = "The column header grades has no siblings. The children of grades are null, 2023, 2024."

        # siblings > 0 & children = 0
        node_3 = self.full_tabtree.row_label_tree.get_node_by_index(
            3, 1
        )  # John, siblings: Tiffany
        expected_string_3 = (
            "The row label John has siblings Tiffany. The values of John are:"
        )

        # siblings = 0 & children = 0
        node_4 = self.full_tabtree.row_label_tree.get_node_by_index(5, 1)  # Michael
        expected_string_4 = "The values of the row label Michael are:"

        # Text Base Approach
        approach = NodeApproach(
            approach=ContextNodeApproach.TEXT, include_context_intersection=False
        )

        # Act
        context_str_1 = ContextStringGeneration.generate_string(
            node_1, self.full_tabtree.row_label_tree, approach  # type: ignore
        )
        context_str_2 = ContextStringGeneration.generate_string(
            node_2, self.full_tabtree.column_header_tree, approach  # type: ignore
        )
        context_str_3 = ContextStringGeneration.generate_string(
            node_3, self.full_tabtree.row_label_tree, approach  # type: ignore
        )
        context_str_4 = ContextStringGeneration.generate_string(
            node_4, self.full_tabtree.row_label_tree, approach  # type: ignore
        )

        # Assert
        self.assertEqual(context_str_1, expected_string_1)
        self.assertEqual(context_str_2, expected_string_2)
        self.assertEqual(context_str_3, expected_string_3)
        self.assertEqual(context_str_4, expected_string_4)

import re
import unittest
from matplotlib import pyplot as plt

from ..model.tabtree.tabtree_model import NodeColor, ValueNode
from ..model.tabtree.string_generation.value_string import (
    ValueStringGeneration,
    ValueStringGenerationBase,
    ValueStringGenerationText,
)
from ..model.tabtree.string_generation.approaches import (
    ContextNodeApproach,
    NodeApproach,
    ValueNodeApproach,
)
from ..model.tabtree.string_generation.context_string import (
    ContextStringGeneration,
    ContextStringGenerationBase,
    ContextStringGenerationText,
)
from ..model.tabtree.string_generation.general import StringGenerationService
from ..model.tabtree.string_generation.separator_approach import SeparatorApproach
from ..model.tabtree.tabtree_service import TabTreeService

from .testutils import AbstractTableTests


class TestTabTreeModel(unittest.TestCase, AbstractTableTests):

    def setUp(self):
        self.setup_parse_and_clean()
        self.tabtree_service = TabTreeService()
        df = self.parsed_df[3]
        df.set_headers(2, 1, override=True)

        # generate model / find tests for this in another file
        self.full_tabtree = self.tabtree_service.generate_full_tabtree(df)

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

    def test_context_intersection_sequence(self):
        # Arrange
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(
            2, 4
        )  # english / 2023
        expected_intersection_seq_1 = ["class", "name", "english"]

        node_2 = self.full_tabtree.column_header_tree.get_node_by_index(0, 2)  # grades
        expected_intersection_seq_2 = ["grades"]

        node_3 = self.full_tabtree.row_label_tree.get_node_by_index(3, 1)  # John
        expected_intersection_seq_3 = ["grades", "name", "John"]

        # Act
        context_intersection_1 = self.full_tabtree.column_header_tree.get_context_intersection_sequence(
            node_1  # type: ignore
        )
        context_intersection_2 = self.full_tabtree.column_header_tree.get_context_intersection_sequence(
            node_2  # type: ignore
        )

        context_intersection_3 = self.full_tabtree.row_label_tree.get_context_intersection_sequence(
            node_3  # type: ignore
        )

        # Assert
        self.assertEqual(
            [node.value for node in context_intersection_1], expected_intersection_seq_1
        )
        self.assertEqual(
            [node.value for node in context_intersection_2], expected_intersection_seq_2
        )
        self.assertEqual(
            [node.value for node in context_intersection_3], expected_intersection_seq_3
        )

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

    def test_context_string_base_with_intersection(self):
        # Arrange
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(0, 2)  # grades
        expected_str_1 = "grades:"

        node_2 = self.full_tabtree.row_label_tree.get_node_by_index(3, 1)  # John
        expected_str_2 = "grades & name & John:"

        node_3 = self.full_tabtree.column_header_tree.get_node_by_index(1, 3)  # A
        expected_str_3 = "2023:"

        approach = NodeApproach(
            approach=ContextNodeApproach.BASE, include_context_intersection=True
        )

        # Act
        context_str_1 = ContextStringGeneration.generate_string(
            node_1, self.full_tabtree.column_header_tree, approach  # type: ignore
        )
        context_str_2 = ContextStringGeneration.generate_string(
            node_2, self.full_tabtree.row_label_tree, approach  # type: ignore
        )
        context_str_3 = ContextStringGeneration.generate_string(
            node_3, self.full_tabtree.column_header_tree, approach  # type: ignore
        )

        # Assert
        self.assertEqual(context_str_1, expected_str_1)
        self.assertEqual(context_str_2, expected_str_2)
        self.assertEqual(context_str_3, expected_str_3)

    def test_context_string_text_with_intersection(self):
        # Arrange

        # Test case for each combination of siblings and children (see line 12, algorithm 3.7)
        # siblings > 0 & children > 0
        node_1 = self.full_tabtree.row_label_tree.get_node_by_index(
            3, 0
        )  # A (grades & class), siblings: B, children: John, Tiffany
        expected_string_1 = "The row label A represents grades and class. The row label A has siblings B. The children of A are John, Tiffany."

        # siblings = 0 & children > 0
        node_2 = self.full_tabtree.column_header_tree.get_node_by_index(
            0, 2
        )  # grades (none), children: null, 2023, 2024
        expected_string_2 = "The column header grades has no siblings. The children of grades are null, 2023, 2024."

        # siblings > 0 & children = 0 -> no context intersection
        node_3 = self.full_tabtree.row_label_tree.get_node_by_index(
            3, 1
        )  # John (grades & name), siblings: Tiffany
        expected_string_3 = "The row label John represents grades and name. The row label John has siblings Tiffany. The values of the row label John are:"

        # siblings > 0 & children = 0 -> with context intersection
        node_3_1 = self.full_tabtree.column_header_tree.get_node_by_index(
            2, 4
        )  # english / 2023, siblings: math
        expected_string_3_1 = "The column header english represents class and name. The column header english has siblings math. The values of the column header english are:"

        # siblings = 0 & children = 0
        node_4 = self.full_tabtree.row_label_tree.get_node_by_index(5, 1)  # Michael
        expected_string_4 = "The row label Michael represents grades and name. The values of the row label Michael are:"

        # Text Base Approach
        approach = NodeApproach(
            approach=ContextNodeApproach.TEXT, include_context_intersection=True
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
        context_str_3_1 = ContextStringGeneration.generate_string(
            node_3_1, self.full_tabtree.column_header_tree, approach  # type: ignore
        )
        context_str_4 = ContextStringGeneration.generate_string(
            node_4, self.full_tabtree.row_label_tree, approach  # type: ignore
        )

        # Assert
        self.assertEqual(context_str_1, expected_string_1)
        self.assertEqual(context_str_2, expected_string_2)
        self.assertEqual(context_str_3, expected_string_3)
        self.assertEqual(context_str_3_1, expected_string_3_1)
        self.assertEqual(context_str_4, expected_string_4)

    def test_value_sequence(self):
        # Arrange
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(
            3, 4
        )  # grades / 2023 / english / C
        expected_sequence_1 = ["grades", "2023", "english", "C"]
        node_2 = self.full_tabtree.row_label_tree.get_node_by_index(
            3, 2
        )  # A / John / 17
        expected_sequence_2 = ["A", "John", "17"]

        # Act
        sequence_1 = self.full_tabtree.column_header_tree.get_value_sequence(node_1)  # type: ignore
        sequence_2 = self.full_tabtree.row_label_tree.get_value_sequence(node_2)  # type: ignore

        # Assert
        self.assertEqual([node.value for node in sequence_1], expected_sequence_1)
        self.assertEqual([node.value for node in sequence_2], expected_sequence_2)

    def test_value_string_base(self):
        # Arrange
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(
            3, 4
        )  # grades / 2023 / english / C
        expected_string_1 = "grades > 2023 > english: C"
        node_2 = self.full_tabtree.row_label_tree.get_node_by_index(
            3, 2
        )  # A / John / 17
        expected_string_2 = "A > John: 17"

        approach = NodeApproach(
            approach=ValueNodeApproach.BASE, include_context_intersection=False
        )

        # Act
        value_str_1 = ValueStringGeneration.generate_string(
            node_1,  # type: ignore
            primary_tree=self.full_tabtree.row_label_tree,
            secondary_tree=self.full_tabtree.column_header_tree,
            approach=approach,
        )
        value_str_2 = ValueStringGeneration.generate_string(
            node_2,  # type: ignore
            primary_tree=self.full_tabtree.column_header_tree,
            secondary_tree=self.full_tabtree.row_label_tree,
            approach=approach,
        )

        # Assert
        self.assertEqual(value_str_1, expected_string_1)
        self.assertEqual(value_str_2, expected_string_2)

    def test_value_string_with_context_intersection(self):
        # Arrange
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(
            3, 4
        )  # grades / 2023 / english (class & name) / C
        expected_string_1 = "grades > 2023 > class & name & english: C"
        node_2 = self.full_tabtree.row_label_tree.get_node_by_index(
            3, 2
        )  # A (grades & class) / John (grades & name) / 17
        expected_string_2 = "grades & class & A > grades & name & John: 17"

        approach = NodeApproach(
            approach=ValueNodeApproach.BASE, include_context_intersection=True
        )

        # Act
        value_str_1 = ValueStringGeneration.generate_string(
            node_1,  # type: ignore
            primary_tree=self.full_tabtree.row_label_tree,
            secondary_tree=self.full_tabtree.column_header_tree,
            approach=approach,
        )
        value_str_2 = ValueStringGeneration.generate_string(
            node_2,  # type: ignore
            primary_tree=self.full_tabtree.column_header_tree,
            secondary_tree=self.full_tabtree.row_label_tree,
            approach=approach,
        )

        # Assert
        self.assertEqual(value_str_1, expected_string_1)
        self.assertEqual(value_str_2, expected_string_2)

    def test_value_string_text(self):
        # Arrange
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(
            3, 4
        )  # grades / 2023 / english / C; row label: John
        expected_string_1 = "The value of the row label John and the column header combination grades, 2023, english is C."
        node_2 = self.full_tabtree.row_label_tree.get_node_by_index(
            3, 2
        )  # A / John / 17; column header: age
        expected_string_2 = "The value of the column header age and the row label combination A, John is 17."

        approach = NodeApproach(
            approach=ValueNodeApproach.TEXT, include_context_intersection=False
        )

        # Act
        value_str_1 = ValueStringGeneration.generate_string(
            node_1,  # type: ignore
            primary_tree=self.full_tabtree.row_label_tree,
            secondary_tree=self.full_tabtree.column_header_tree,
            approach=approach,
        )
        value_str_2 = ValueStringGeneration.generate_string(
            node_2,  # type: ignore
            primary_tree=self.full_tabtree.column_header_tree,
            secondary_tree=self.full_tabtree.row_label_tree,
            approach=approach,
        )

        # Assert
        self.assertEqual(value_str_1, expected_string_1)
        self.assertEqual(value_str_2, expected_string_2)

    def test_value_string_text_with_intersection(self):
        # Arrange
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(
            3, 4
        )  # grades / 2023 / english (class & name) / C; row label: John
        expected_string_1 = "The value of the row label John and the column header combination grades, 2023, class & name & english is C."
        node_2 = self.full_tabtree.row_label_tree.get_node_by_index(
            3, 2
        )  # A (grades & class) / John (grades & name) / 17; column header: age
        expected_string_2 = "The value of the column header age and the row label combination grades & class & A, grades & name & John is 17."

        approach = NodeApproach(
            approach=ValueNodeApproach.TEXT, include_context_intersection=True
        )

        # Act
        value_str_1 = ValueStringGeneration.generate_string(
            node_1,  # type: ignore
            primary_tree=self.full_tabtree.row_label_tree,
            secondary_tree=self.full_tabtree.column_header_tree,
            approach=approach,
        )

        value_str_2 = ValueStringGeneration.generate_string(
            node_2,  # type: ignore
            primary_tree=self.full_tabtree.column_header_tree,
            secondary_tree=self.full_tabtree.row_label_tree,
            approach=approach,
        )

        # Assert
        self.assertEqual(value_str_1, expected_string_1)
        self.assertEqual(value_str_2, expected_string_2)

    def test_value_string_text_augmented(self):
        # Arrange
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(
            3, 4
        )  # grades / 2023 / english / C; row label: A / John
        expected_string_1 = "The value of the row label combination A, John and the column header combination grades, 2023, english is C."

        node_2 = self.full_tabtree.row_label_tree.get_node_by_index(
            3, 2
        )  # A / John / 17; column header: age
        expected_string_2 = "The value of the column header combination grades, null, age and the row label combination A, John is 17."

        approach = NodeApproach(
            approach=ValueNodeApproach.TEXT_AUGMENTED,
            include_context_intersection=False,
        )

        # Act
        value_str_1 = ValueStringGeneration.generate_string(
            node_1,  # type: ignore
            primary_tree=self.full_tabtree.row_label_tree,
            secondary_tree=self.full_tabtree.column_header_tree,
            approach=approach,
        )
        value_str_2 = ValueStringGeneration.generate_string(
            node_2,  # type: ignore
            primary_tree=self.full_tabtree.column_header_tree,
            secondary_tree=self.full_tabtree.row_label_tree,
            approach=approach,
        )

        # Assert
        self.assertEqual(value_str_1, expected_string_1)
        self.assertEqual(value_str_2, expected_string_2)

    def test_value_string_text_augmented_with_intersection(self):
        # Arrange
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(
            3, 4
        )  # grades / 2023 / english (class & name) / C; row label: A (grades & class) / John
        expected_string_1 = "The value of the row label combination grades & class & A, grades & name & John and the column header combination grades, 2023, class & name & english is C."
        node_2 = self.full_tabtree.row_label_tree.get_node_by_index(
            3, 2
        )  # A (grades & class) / John (grades & name) / 17; column header: age
        expected_string_2 = "The value of the column header combination grades, null, class & name & age and the row label combination grades & class & A, grades & name & John is 17."

        approach = NodeApproach(
            approach=ValueNodeApproach.TEXT_AUGMENTED, include_context_intersection=True
        )

        # Act
        value_str_1 = ValueStringGeneration.generate_string(
            node_1,  # type: ignore
            primary_tree=self.full_tabtree.row_label_tree,
            secondary_tree=self.full_tabtree.column_header_tree,
            approach=approach,
        )
        value_str_2 = ValueStringGeneration.generate_string(
            node_2,  # type: ignore
            primary_tree=self.full_tabtree.column_header_tree,
            secondary_tree=self.full_tabtree.row_label_tree,
            approach=approach,
        )

        # Assert
        self.assertEqual(value_str_1, expected_string_1)
        self.assertEqual(value_str_2, expected_string_2)

    def test_real_world_example_awk_1(self):
        # Arrange
        df = self.parsed_df[4]
        df.set_headers(1, 0, override=True)

        # generate model
        self.full_tabtree = self.tabtree_service.generate_full_tabtree(df)

        # Check whether the model was generated correctly
        self.assertIsNotNone(self.full_tabtree)

        # Select nodes
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(3, 2)  # 2,143
        expected_str_1 = [
            "The value of the column header Revenue and the row label Residential is 2,143.",
            "The value of the column header combination 2023, Revenue and the row label Residential is 2,143.",
            "The value of the column header combination 2023, (In millions) & Revenue and the row label (In millions) & Residential is 2,143.",
        ]
        self.assertIsInstance(node_1, ValueNode)
        self.assertEqual(node_1.value, "2,143")  # type: ignore

        node_2 = self.full_tabtree.row_label_tree.get_node_by_index(7, 7)  # 267
        expected_str_2 = [
            "The value of the row label Public and other water (a) and the column header combination 2022, Revenue is 267.",
            "The value of the row label Public and other water (a) and the column header combination 2022, Revenue is 267.",
            "The value of the row label (In millions) & Public and other water (a) and the column header combination 2022, (In millions) & Revenue is 267.",
        ]
        self.assertIsInstance(node_2, ValueNode)
        self.assertEqual(node_2.value, "267")  # type: ignore

        node_3 = self.full_tabtree.row_label_tree.get_node_by_index(3, 2)  # 2,143
        expected_str_3 = [
            "The value of the row label Residential and the column header combination 2023, Revenue is 2,143.",
            "The value of the row label Residential and the column header combination 2023, Revenue is 2,143.",
            "The value of the row label (In millions) & Residential and the column header combination 2023, (In millions) & Revenue is 2,143.",
        ]
        self.assertIsInstance(node_3, ValueNode)

        approaches = [
            NodeApproach(
                approach=ValueNodeApproach.TEXT, include_context_intersection=False
            ),
            NodeApproach(
                approach=ValueNodeApproach.TEXT_AUGMENTED,
                include_context_intersection=False,
            ),
            NodeApproach(
                approach=ValueNodeApproach.TEXT_AUGMENTED,
                include_context_intersection=True,
            ),
        ]

        # Act
        value_str_1 = [
            ValueStringGeneration.generate_string(
                node_1,  # type: ignore
                primary_tree=self.full_tabtree.column_header_tree,
                secondary_tree=self.full_tabtree.row_label_tree,
                approach=approach,
            )
            for approach in approaches
        ]
        value_str_2 = [
            ValueStringGeneration.generate_string(
                node_2,  # type: ignore
                primary_tree=self.full_tabtree.row_label_tree,
                secondary_tree=self.full_tabtree.column_header_tree,
                approach=approach,
            )
            for approach in approaches
        ]
        value_str_3 = [
            ValueStringGeneration.generate_string(
                node_3,  # type: ignore
                primary_tree=self.full_tabtree.row_label_tree,
                secondary_tree=self.full_tabtree.column_header_tree,
                approach=approach,
            )
            for approach in approaches
        ]

        # Assert
        for i in range(len(approaches)):
            self.assertEqual(value_str_1[i], expected_str_1[i])
            self.assertEqual(value_str_2[i], expected_str_2[i])
            self.assertEqual(value_str_3[i], expected_str_3[i])

    def test_real_world_example_awk_2(self):
        # Arrange
        df = self.parsed_df[5]
        df.set_headers(0, -1, override=True)

        # generate model
        self.full_tabtree = self.tabtree_service.generate_full_tabtree(df)

        # Check whether the model was generated correctly
        self.assertIsNotNone(self.full_tabtree)

        # Select value nodes
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(
            3, 2
        )  # MD, MO, NJ, WV
        expected_str_1 = (
            "The value of the column header States Allowed is MD, MO, NJ, WV."
        )

        self.assertIsInstance(node_1, ValueNode)
        self.assertEqual(node_1.value, "MD, MO, NJ, WV")  # type: ignore

        node_2 = self.full_tabtree.row_label_tree.get_node_by_index(4, 0)  # 267
        expected_str_2 = "The value of the column header Regulatory Practices is Utility plant recovery mechanisms."
        self.assertIsInstance(node_2, ValueNode)
        self.assertEqual(node_2.value, "Utility plant recovery mechanisms")  # type: ignore

        approaches = [
            NodeApproach(
                approach=ValueNodeApproach.TEXT, include_context_intersection=False
            ),
            NodeApproach(
                approach=ValueNodeApproach.TEXT_AUGMENTED,
                include_context_intersection=False,
            ),
            NodeApproach(
                approach=ValueNodeApproach.TEXT_AUGMENTED,
                include_context_intersection=True,
            ),
        ]

        # Act
        value_str_1 = [
            ValueStringGeneration.generate_string(
                node_1,  # type: ignore
                primary_tree=self.full_tabtree.column_header_tree,
                secondary_tree=self.full_tabtree.row_label_tree,
                approach=approach,
            )
            for approach in approaches
        ]
        value_str_2 = [
            ValueStringGeneration.generate_string(
                node_2,  # type: ignore
                primary_tree=self.full_tabtree.row_label_tree,
                secondary_tree=self.full_tabtree.column_header_tree,
                approach=approach,
            )
            for approach in approaches
        ]

        # Assert
        for i in range(len(approaches)):
            self.assertEqual(value_str_1[i], expected_str_1)
            self.assertEqual(value_str_2[i], expected_str_2)

    def test_real_world_example_awk_3(self):
        # Arrange
        df = self.parsed_df[6]
        df.set_headers(0, 0, override=True)

        # generate model
        self.full_tabtree = self.tabtree_service.generate_full_tabtree(df)

        # Check whether the model was generated correctly
        self.assertIsNotNone(self.full_tabtree)

        # Select nodes
        node_1 = self.full_tabtree.column_header_tree.get_node_by_index(1, 1)
        expected_str_1 = "The value of the column header Surface Water and the row label New Jersey is 74%."

        self.assertIsInstance(node_1, ValueNode)
        self.assertEqual(node_1.value, "74%")  # type: ignore

        approaches = [
            NodeApproach(
                approach=ValueNodeApproach.TEXT, include_context_intersection=False
            ),
            NodeApproach(
                approach=ValueNodeApproach.TEXT_AUGMENTED,
                include_context_intersection=False,
            ),
            NodeApproach(
                approach=ValueNodeApproach.TEXT_AUGMENTED,
                include_context_intersection=True,
            ),
        ]

        # Act
        value_str_1 = [
            ValueStringGeneration.generate_string(
                node_1,  # type: ignore
                primary_tree=self.full_tabtree.column_header_tree,
                secondary_tree=self.full_tabtree.row_label_tree,
                approach=approach,
            )
            for approach in approaches
        ]

        # Assert
        for i in range(len(approaches)):
            self.assertEqual(value_str_1[i], expected_str_1)

    def test_full_serialization(self):
        context_string_approach = (
            NodeApproach(
                approach=ContextNodeApproach.TEXT, include_context_intersection=False
            ),
        )
        value_string_approach = (
            NodeApproach(
                approach=ValueNodeApproach.TEXT, include_context_intersection=False
            ),
        )

        approaches = (
            context_string_approach[0],
            value_string_approach[0],
        )

        full_str = self.tabtree_service.generate_serialized_string(
            tabtree=self.full_tabtree,
            primary_colour=NodeColor.YELLOW,
            approaches=approaches,
        )

        full_str_2 = self.tabtree_service.generate_serialized_string(
            tabtree=self.full_tabtree,
            primary_colour=NodeColor.BLUE,
            approaches=approaches,
        )

        expected_str = """
    The table captures grades as its main column header.
    The column header grades has no siblings. The children of grades are null, 2023, 2024.
    The column header null has siblings 2023, 2024. The children of null are age.
    The values of the column header age are:
    The value of the column header age and the row label combination B, Michael is 17.
    The value of the column header age and the row label combination A, Tiffany is 16.
    The value of the column header age and the row label combination A, John is 17.
    The column header 2023 has siblings null, 2024. The children of 2023 are math, english.
    The column header math has siblings english. The values of math are:
    The value of the column header math and the row label combination B, Michael is D.
    The value of the column header math and the row label combination A, Tiffany is B.
    The value of the column header math and the row label combination A, John is A.
    The column header english has siblings math. The values of english are:
    The value of the column header english and the row label combination B, Michael is D.
    The value of the column header english and the row label combination A, Tiffany is B.
    The value of the column header english and the row label combination A, John is C.
    The column header 2024 has siblings null, 2023. The children of 2024 are math, english.
    The column header math has siblings english. The values of math are:
    The value of the column header math and the row label combination B, Michael is D.
    The value of the column header math and the row label combination A, Tiffany is C.
    The value of the column header math and the row label combination A, John is A.
    The column header english has siblings math. The values of english are:
    The value of the column header english and the row label combination B, Michael is D.
    The value of the column header english and the row label combination A, Tiffany is B.
    The value of the column header english and the row label combination A, John is B.
    """

        pattern = re.compile(r"\s+")
        full_str_mod = re.sub(pattern, "", full_str)
        expected_str_mod = re.sub(pattern, "", expected_str)
        self.assertEqual(full_str_mod, expected_str_mod)

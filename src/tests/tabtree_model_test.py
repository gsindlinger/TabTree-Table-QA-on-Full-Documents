import copy
import unittest

from matplotlib import pyplot as plt

from ..model.tabtree.tabtree_model import CellNode, NodeColor
from ..model.tabtree.tabtree_service import TabTreeService
from .testutils import AbstractTableTests


class TestTabTreeModel(unittest.TestCase, AbstractTableTests):

    def setUp(self):
        self.setup_parse_and_clean()
        self.tabtree_service = TabTreeService()

    def test_split_table_on_context(self):
        ### Arrange
        df = self.parsed_df[2]
        df.set_headers(1, 2)

        cells = [(0, 0), (0, 3), (4, 0), (5, 4)]

        cell_1_copy = copy.deepcopy(df.get_cell(*cells[0]))
        cell_2_copy = copy.deepcopy(df.get_cell(*cells[1]))
        cell_3_copy = copy.deepcopy(df.get_cell(*cells[2]))
        cell_4_copy = copy.deepcopy(df.get_cell(*cells[3]))

        # same code as for regular parsing
        self.assertEqual(cell_1_copy.colspan, (0, 6))
        self.assertEqual(cell_1_copy.rowspan, (0, 0))

        self.assertEqual(cell_2_copy.colspan, (3, 3))
        self.assertEqual(cell_2_copy.rowspan, (0, 0))

        self.assertEqual(cell_3_copy.colspan, (0, 0))
        self.assertEqual(cell_3_copy.rowspan, (1, 1))

        self.assertEqual(cell_4_copy.colspan, (1, 0))
        self.assertEqual(cell_4_copy.rowspan, (0, 1))

        ### Act
        df.split_cells_on_headers()

        ### Assert
        cell_1 = df.get_cell(*cells[0])
        cell_2 = df.get_cell(*cells[1])
        cell_3 = df.get_cell(*cells[2])
        cell_4 = df.get_cell(*cells[3])

        self.assertEqual(cell_1.colspan, (0, 1))
        self.assertEqual(cell_1.rowspan, (0, 0))

        self.assertEqual(cell_2.colspan, (1, 3))
        self.assertEqual(cell_2.rowspan, (0, 0))

        self.assertEqual(cell_3.colspan, (0, 0))
        self.assertEqual(cell_3.rowspan, (1, 1))

        self.assertEqual(cell_4.colspan, (0, 0))
        self.assertEqual(cell_4.rowspan, (0, 0))

        self.assertNotEqual(cell_1, cell_1_copy)
        self.assertNotEqual(cell_2, cell_2_copy)
        # for third cell nothing changed
        self.assertEqual(cell_3, cell_3_copy)

        self.assertEqual(cell_1.value, cell_1_copy.value)
        self.assertEqual(cell_2.value, cell_2_copy.value)

    def test_model_generation(self):
        ### Arrange
        df = self.parsed_df[3]
        df.set_headers(1, 2)

        # grades cell
        cell_1 = df.get_cell(0, 0)  # context-intersection
        cell_2 = df.get_cell(0, 2)  # column header

        # 2024
        cell_3 = df.get_cell(1, 5)  # column header
        cell_4 = df.get_cell(1, 6)  # should not exist in tree

        # name
        cell_5 = df.get_cell(2, 1)  # context-intersection

        # A
        cell_6 = df.get_cell(0, 3)  # row label
        cell_7 = df.get_cell(0, 4)  # should not exist in tree

        # 16
        cell_7 = df.get_cell(4, 2)  # value cell
        # Tiffany / C
        cell_8 = df.get_cell(4, 5)  # value cell

        ### Act
        tabtree = self.tabtree_service.generate_full_tabtree(df)

        ### Assert
        # check if all nodes are in the tree
        self.assertIsNotNone(
            tabtree.column_header_tree.get_node_by_index(
                cell_1.row_index, cell_1.col_index
            )
        )
        self.assertIsNotNone(
            tabtree.column_header_tree.get_node_by_index(
                cell_2.row_index, cell_2.col_index
            )
        )
        self.assertIsNotNone(
            tabtree.column_header_tree.get_node_by_index(
                cell_3.row_index, cell_3.col_index
            )
        )
        self.assertIsNone(
            tabtree.column_header_tree.get_node_by_index(
                cell_4.row_index, cell_4.col_index
            )
        )
        self.assertIsNotNone(
            tabtree.column_header_tree.get_node_by_index(
                cell_5.row_index, cell_5.col_index
            )
        )
        self.assertIsNone(
            tabtree.column_header_tree.get_node_by_index(
                cell_6.row_index, cell_6.col_index
            )
        )
        self.assertIsNotNone(
            tabtree.column_header_tree.get_node_by_index(
                cell_7.row_index, cell_7.col_index
            )
        )
        self.assertIsNotNone(
            tabtree.column_header_tree.get_node_by_index(
                cell_8.row_index, cell_8.col_index
            )
        )

        # check for correct colours
        self.assertEqual(
            tabtree.column_header_tree.get_node_by_index(
                cell_1.row_index, cell_1.col_index
            ).colour,  # type: ignore
            NodeColor.ORANGE,
        )

        self.assertEqual(
            tabtree.column_header_tree.get_node_by_index(
                cell_2.row_index, cell_2.col_index
            ).colour,  # type: ignore
            NodeColor.YELLOW,
        )

        self.assertEqual(
            tabtree.column_header_tree.get_node_by_index(
                cell_3.row_index, cell_3.col_index
            ).colour,  # type: ignore
            NodeColor.YELLOW,
        )

        self.assertEqual(
            tabtree.column_header_tree.get_node_by_index(
                cell_5.row_index, cell_5.col_index
            ).colour,  # type: ignore
            NodeColor.ORANGE,
        )

        self.assertEqual(
            tabtree.column_header_tree.get_node_by_index(
                cell_7.row_index, cell_7.col_index
            ).colour,  # type: ignore
            NodeColor.GRAY,
        )

        self.assertEqual(
            tabtree.column_header_tree.get_node_by_index(
                cell_8.row_index, cell_8.col_index
            ).colour,  # type: ignore
            NodeColor.GRAY,
        )

        # check for correct edges

        # for cell_1
        self.assertTrue(
            tabtree.column_header_tree.has_edge(
                tabtree.column_header_tree.get_node_by_index(
                    cell_1.row_index, cell_1.col_index
                ),  # type: ignore
                tabtree.column_header_tree.get_node_by_index(
                    cell_2.row_index, cell_2.col_index
                ),  # type: ignore
            )
        )

        for col_index in [2, 3, 5]:
            self.assertTrue(
                tabtree.column_header_tree.has_edge(
                    tabtree.column_header_tree.get_node_by_index(
                        cell_2.row_index, cell_2.col_index
                    ),  # type: ignore
                    tabtree.column_header_tree.get_node_by_index(
                        cell_2.row_index + 1, col_index
                    ),  # type: ignore
                )
            )

        # 2024
        self.assertTrue(
            tabtree.column_header_tree.has_edge(
                tabtree.column_header_tree.get_node_by_index(
                    cell_5.row_index, cell_5.col_index
                ),  # type: ignore
                tabtree.column_header_tree.get_node_by_index(
                    cell_3.row_index + 1, cell_3.col_index
                ),  # type: ignore
            )
        )

        for col_index in [5, 6]:
            self.assertTrue(
                tabtree.column_header_tree.has_edge(
                    tabtree.column_header_tree.get_node_by_index(
                        cell_3.row_index, cell_3.col_index
                    ),  # type: ignore
                    value_cell := tabtree.column_header_tree.get_node_by_index(
                        cell_3.row_index + 1, col_index
                    ),  # type: ignore
                )
            )
            self.assertTrue(value_cell.colour == NodeColor.GRAY)  # type: ignore

        # name
        self.assertTrue(
            tabtree.column_header_tree.has_edge(
                context_interaction_cell := tabtree.column_header_tree.get_node_by_index(
                    cell_5.row_index, cell_5.col_index - 1
                ),  # type: ignore
                tabtree.column_header_tree.get_node_by_index(
                    cell_5.row_index, cell_5.col_index
                ),  # type: ignore
            )
        )
        self.assertTrue(context_interaction_cell.colour == NodeColor.ORANGE)  # type: ignore

        # check in-degree for value cells
        self.assertEqual(
            tabtree.column_header_tree.in_degree(CellNode.generate_id(cell_7.row_index, cell_7.col_index)),  # type: ignore
            1,
        )
        self.assertEqual(
            tabtree.column_header_tree.in_degree(CellNode.generate_id(cell_8.row_index, cell_8.col_index)),  # type: ignore
            1,
        )

        ### Draw tree for visual inspection
        import networkx as nx
        from networkx.drawing.nx_agraph import write_dot, graphviz_layout
        import matplotlib.pyplot as plt

        write_dot(tabtree.column_header_tree, "column_header_tree.dot")
        pos = graphviz_layout(tabtree.column_header_tree, prog="dot")

        colors = [node["colour"] for node in tabtree.column_header_tree.nodes.values()]
        nx.draw(
            tabtree.column_header_tree,
            pos,
            with_labels=True,
            arrows=True,
            node_color=colors,
        )
        plt.show()

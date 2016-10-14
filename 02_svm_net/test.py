from unittest import TestCase

import numpy as np

from main import AdditionNode


class TestAdditionNode(TestCase):
    def setUp(self):
        self.input_1 = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        self.input_2 = np.array([
            [4, 5, 6],
            [7, 8, 9]
        ])
        self.node = AdditionNode()

    def test_forward(self):
        result = self.node.forward(self.input_1, self.input_2)

        np.testing.assert_allclose(
            result,
            np.array([
                [5, 7, 9],
                [11, 13, 15]
            ])
        )

    def test_local_gradients(self):
        self.node.forward(self.input_1, self.input_2)

        gradients = self.node.gradients()
        self.assertIsInstance(gradients, list)
        self.assertEqual(len(gradients), 2)

        np.testing.assert_allclose(
            gradients[0],
            np.ones([2, 3])
        )
        np.testing.assert_allclose(
            gradients[1],
            np.ones([2, 3])
        )

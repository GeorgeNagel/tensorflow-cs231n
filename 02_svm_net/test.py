from unittest import TestCase

import numpy as np

from main import AdditionNode, numerical_gradient


class TestNumericalGradient(TestCase):
    def test_sumall_gradient(self):
        input_vector = np.ones([3, 5])
        fn_to_optimize = lambda x: np.sum(x)

        actual_gradient = numerical_gradient(fn_to_optimize, input_vector)

        expected_gradient = np.ones([3, 5])
        np.testing.assert_allclose(expected_gradient, actual_gradient)

    def test_sumall_gradient_negative_input(self):
        input_vector = -1 * np.ones([3, 5])
        fn_to_optimize = lambda x: np.sum(x)

        actual_gradient = numerical_gradient(fn_to_optimize, input_vector)

        expected_gradient = np.ones([3, 5])
        np.testing.assert_allclose(expected_gradient, actual_gradient)


class TestAdditionNode(TestCase):
    def setUp(self):
        self.input_1 = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ], dtype=np.float64)
        self.input_2 = np.array([
            [4, 5, 6],
            [7, 8, 9]
        ],  dtype=np.float64)
        self.node = AdditionNode()

    def test_forward(self):
        result = self.node.forward(self.input_1, self.input_2)

        np.testing.assert_allclose(
            result,
            np.array([
                [5, 7, 9],
                [11, 13, 15]
            ], dtype=np.float64)
        )

    def test_local_gradients_analytic(self):
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

    def test_local_gradients_numerical(self):
        def loss_1(input_1):
            out = self.node.forward(input_1, self.input_2)
            return np.sum(out)

        def loss_2(input_2):
            out = self.node.forward(self.input_1, input_2)
            return np.sum(out)
        gradient_1 = numerical_gradient(loss_1, self.input_1)
        gradient_2 = numerical_gradient(loss_2, self.input_2)

        np.testing.assert_allclose(
            gradient_1,
            np.ones([2, 3])
        )
        np.testing.assert_allclose(
            gradient_2,
            np.ones([2, 3])
        )

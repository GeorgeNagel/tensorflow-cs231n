from unittest import TestCase

import numpy as np

from main import numerical_gradient
from net.nodes import (
    AdditionNode, MultiplicationNode, SVMLossNode)


class TestNumericalGradient(TestCase):
    def test_sumall_gradient(self):
        input_vector = np.ones([3, 5])

        actual_gradient = numerical_gradient(lambda x: np.sum(x), input_vector)

        expected_gradient = np.ones([3, 5])
        np.testing.assert_allclose(expected_gradient, actual_gradient)

    def test_sumall_gradient_negative_input(self):
        input_vector = -1 * np.ones([3, 5])

        actual_gradient = numerical_gradient(lambda x: np.sum(x), input_vector)

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
        self.expected_grad_1 = np.ones([2, 3])
        self.expected_grad_2 = np.ones([2, 3])

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
            self.expected_grad_1
        )
        np.testing.assert_allclose(
            gradients[1],
            self.expected_grad_2
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
            self.expected_grad_1
        )
        np.testing.assert_allclose(
            gradient_2,
            self.expected_grad_2
        )


class TestMultiplicationNode(TestCase):
    def setUp(self):
        self.input_1 = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ], dtype=np.float64)
        self.input_2 = np.array([
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ], dtype=np.float64)
        self.node = MultiplicationNode()
        self.expected_grad_1 = np.array([
            [24, 33, 42],
            [24, 33, 42]
        ], dtype=np.float64)
        self.expected_grad_2 = np.array([
            [5, 5, 5],
            [7, 7, 7],
            [9, 9, 9]
        ], dtype=np.float64)

    def test_local_gradients_numerical(self):
        step_size = .1 ** 6

        def loss_1(input_1):
            out = self.node.forward(input_1, self.input_2)
            return np.sum(out)

        def loss_2(input_2):
            out = self.node.forward(self.input_1, input_2)
            return np.sum(out)
        gradient_1 = numerical_gradient(
            loss_1, self.input_1, step_size=step_size)
        gradient_2 = numerical_gradient(
            loss_2, self.input_2, step_size=step_size)

        np.testing.assert_allclose(
            gradient_1,
            self.expected_grad_1
        )
        np.testing.assert_allclose(
            gradient_2,
            self.expected_grad_2
        )

    def test_local_gradients_analytical(self):
        self.node.forward(self.input_1, self.input_2)
        gradient_1, gradient_2 = self.node.gradients()

        np.testing.assert_allclose(
            gradient_1,
            self.expected_grad_1
        )
        np.testing.assert_allclose(
            gradient_2,
            self.expected_grad_2
        )


class TestSVMLoss(TestCase):
    def test_single_point_loss_correct_class(self):
        node = SVMLossNode()
        scores = np.array([
            [10, 0],
            [10, 0]
        ])
        correct_class_indexes = np.array([0, 0])
        loss = node.forward(scores, correct_class_indexes)
        self.assertEqual(loss, 0)

from unittest import TestCase

import numpy as np

from main import numerical_gradient, vectorized_loss, analytic_gradient


class TestGradient(TestCase):
    def test_numerical_gradient(self):
        """ Test the analytic gradient by calculating a numerical gradient. """
        W = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([1, 2], dtype=np.float64)
        x = np.array([1, 2, 3], dtype=np.float64)
        correct_class_index = 0

        fn_to_optimize = lambda W: vectorized_loss(
            x, correct_class_index, W, b)
        grad_W = numerical_gradient(fn_to_optimize, W)
        expected_grad_W = np.array([
            [-1, -2, -3],
            [1, 2, 3]
        ], dtype=np.float64)
        self.assertTrue(np.allclose(grad_W, expected_grad_W))

        fn_to_optimize = lambda b: vectorized_loss(
            x, correct_class_index, W, b)
        grad_b = numerical_gradient(fn_to_optimize, b)
        expected_grad_b = np.array([-1, 1], dtype=np.float64)
        self.assertTrue(np.allclose(grad_b, expected_grad_b))


class TestVectorizedLoss(TestCase):
    def test_loss(self):
        W = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([1, 2], dtype=np.float64)
        x = np.array([1, 2, 3], dtype=np.float64)
        correct_class_index = 0

        loss = vectorized_loss(x, correct_class_index, W, b)
        self.assertTrue(np.isclose(loss, 20.0))


class TestAnalyticGradient(TestCase):
    def test_gradient(self):
        pass

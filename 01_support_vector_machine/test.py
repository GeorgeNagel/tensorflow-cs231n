from unittest import TestCase

import numpy as np

from main import (
    numerical_gradient, vectorized_loss, analytic_gradient, single_point_loss,
    analytic_gradient_vectorized)


class TestGradient(TestCase):
    def test_numerical_gradient(self):
        """ Test the numerical gradient estimate. """
        W = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([1, 2], dtype=np.float64)
        x = np.array([1, 2, 3], dtype=np.float64)
        correct_class_index = 0

        fn_to_optimize = lambda W: vectorized_loss(
            x, correct_class_index, W, b)
        grad_W = numerical_gradient(fn_to_optimize, W)
        expected_grad_W = np.array([
            [0, -1, -2],
            [2, 3, 4]
        ], dtype=np.float64)
        np.testing.assert_allclose(grad_W, expected_grad_W)

        fn_to_optimize = lambda b: vectorized_loss(
            x, correct_class_index, W, b)
        grad_b = numerical_gradient(fn_to_optimize, b)
        expected_grad_b = np.array([-1, 1], dtype=np.float64)
        np.testing.assert_allclose(grad_b, expected_grad_b)

    def test_numerical_gradient_vectorized(self):
        """ Test the numerical gradient for multiple inputs X. """
        W = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([1, 2], dtype=np.float64)
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        Y = np.array([[1, 0], [0, 1]], dtype=np.float64)

        fn_to_optimize = lambda W: vectorized_loss(
            X, Y, W, b)
        grad_W = numerical_gradient(fn_to_optimize, W)
        expected_grad_W = np.array([
            [2.5, 2.5, 2.5],
            [1, 1, 1]
        ], dtype=np.float64)
        np.testing.assert_allclose(grad_W, expected_grad_W)

        fn_to_optimize = lambda b: vectorized_loss(
            X, Y, W, b)
        grad_b = numerical_gradient(fn_to_optimize, b)
        expected_grad_b = np.array([-0.5, 0.5], dtype=np.float64)
        np.testing.assert_allclose(grad_b, expected_grad_b)


class TestLoss(TestCase):
    def test_single_point_loss(self):
        W = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([1, 2], dtype=np.float64)
        x = np.array([1, 2, 3], dtype=np.float64)
        correct_class_index = 0

        loss = single_point_loss(x, correct_class_index, W, b)
        np.testing.assert_almost_equal(loss, 41)

    def test_vectorized_loss(self):
        W = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([1, 2], dtype=np.float64)
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        Y = np.array([[1, 0], [0, 1]], dtype=np.float64)

        loss = vectorized_loss(X, Y, W, b)
        np.testing.assert_almost_equal(loss, 31)

    def test_vectorized_loss_single_data_point(self):
        """ Run the vectorized_loss function for a single data point. """
        W = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([1, 2], dtype=np.float64)
        x = np.array([1, 2, 3], dtype=np.float64)
        correct_class_index = 0

        loss = vectorized_loss(x, correct_class_index, W, b)
        np.testing.assert_almost_equal(loss, 41)


class TestAnalyticGradient(TestCase):
    def test_gradient(self):
        W = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([1, 2], dtype=np.float64)
        x = np.array([1, 2, 3], dtype=np.float64)
        correct_class_index = 0

        grad_W, grad_b = analytic_gradient(W, x, b, correct_class_index)
        expected_grad_W = np.array([
            [0, -1, -2],
            [2, 3, 4]
        ], dtype=np.float64)
        np.testing.assert_allclose(grad_W, expected_grad_W)

        expected_grad_b = np.array([-1, 1], dtype=np.float64)
        np.testing.assert_allclose(grad_b, expected_grad_b)

    def test_vectorized_gradient(self):
        W = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([1, 2], dtype=np.float64)
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        Y = np.array([[1, 0], [0, 1]], dtype=np.float64)

        grad_W, grad_b = analytic_gradient_vectorized(W, X, b, Y)

        expected_grad_W = np.array([
            [2.5, 2.5, 2.5],
            [1, 1, 1]
        ], dtype=np.float64)
        np.testing.assert_allclose(grad_W, expected_grad_W)

        expected_grad_b = np.array([-0.5, 0.5], dtype=np.float64)
        np.testing.assert_allclose(grad_b, expected_grad_b)

from unittest import TestCase

import numpy as np

from main import (
    numerical_gradient, vectorized_loss, analytic_gradient, single_point_loss,
    analytic_gradient_vectorized)


def single_point_incorrect_test_data():
    W = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=np.float64)
    b = np.array([1, 2], dtype=np.float64)
    x = np.array([1, 2, 3], dtype=np.float64)
    correct_class_index = 0
    grad_W = np.array([
        [0, -1, -2],
        [2, 3, 4]
    ], dtype=np.float64)
    grad_b = np.array([-1, 1], dtype=np.float64)
    return W, x, b, correct_class_index, grad_W, grad_b


def single_point_correct_test_data():
    W = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=np.float64)
    b = np.array([1, 2], dtype=np.float64)
    x = np.array([1, 2, 3], dtype=np.float64)
    correct_class_index = 1
    grad_W = np.array([
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.float64)
    grad_b = np.array([0, 0], dtype=np.float64)
    return W, x, b, correct_class_index, grad_W, grad_b


def vectorized_test_data():
    W = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=np.float64)
    b = np.array([1, 2], dtype=np.float64)
    X = np.array([
        [1, 2, 3],
        [1, 2, 3]
    ], dtype=np.float64)
    Y = np.array([
        [1, 0],
        [0, 1]
    ], dtype=np.float64)
    grad_W = np.array([
        [2.5, 2.5, 2.5],
        [1, 1, 1]
    ], dtype=np.float64)
    grad_b = np.array([-0.5, 0.5], dtype=np.float64)
    return W, X, b, Y, grad_W, grad_b


def vectorized_test_data_all_incorrect():
    W = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=np.float64)
    b = np.array([1, 2], dtype=np.float64)
    X = np.array([
        [1, 2, 3],
        [1, 2, 3]
    ], dtype=np.float64)
    Y = np.array([
        [1, 0],
        [1, 0]
    ], dtype=np.float64)
    grad_W = np.array([
        [0, -1, -2],
        [2, 3, 4]
    ], dtype=np.float64)
    grad_b = np.array([-1, 1], dtype=np.float64)
    return W, X, b, Y, grad_W, grad_b


def vectorized_test_data_all_correct():
    W = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=np.float64)
    b = np.array([1, 2], dtype=np.float64)
    X = np.array([
        [1, 2, 3],
        [1, 2, 3]
    ], dtype=np.float64)
    Y = np.array([
        [0, 1],
        [0, 1]
    ], dtype=np.float64)
    grad_W = np.array([
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.float64)
    grad_b = np.array([0, 0], dtype=np.float64)
    return W, X, b, Y, grad_W, grad_b


class TestSinglePointNumericalGradient(TestCase):
    def test_numerical_gradient_incorrect(self):
        """ Test the numerical gradient on an incorrect data point. """
        W, x, b, correct_class_index, expected_grad_W, expected_grad_b = single_point_incorrect_test_data()  # noqa

        fn_to_optimize = lambda W: vectorized_loss(
            x, correct_class_index, W, b)
        grad_W = numerical_gradient(fn_to_optimize, W)
        np.testing.assert_allclose(grad_W, expected_grad_W)

        fn_to_optimize = lambda b: vectorized_loss(
            x, correct_class_index, W, b)
        grad_b = numerical_gradient(fn_to_optimize, b)
        np.testing.assert_allclose(grad_b, expected_grad_b)

    def test_numerical_gradient_correct(self):
        """
        Test the numerical gradient on a correct data point.
        The only gradient contribution should be from the regularization.
        """
        W, x, b, correct_class_index, expected_grad_W, expected_grad_b = single_point_correct_test_data()  # noqa

        fn_to_optimize = lambda W: vectorized_loss(
            x, correct_class_index, W, b)
        grad_W = numerical_gradient(fn_to_optimize, W)
        np.testing.assert_allclose(grad_W, expected_grad_W)

        fn_to_optimize = lambda b: vectorized_loss(
            x, correct_class_index, W, b)
        grad_b = numerical_gradient(fn_to_optimize, b)
        np.testing.assert_allclose(grad_b, expected_grad_b)

    def test_numerical_gradient_vectorized(self):
        """ Test the numerical gradient for multiple inputs X. """
        W, X, b, Y, expected_grad_W, expected_grad_b = vectorized_test_data()

        fn_to_optimize = lambda W: vectorized_loss(
            X, Y, W, b)
        grad_W = numerical_gradient(fn_to_optimize, W)
        np.testing.assert_allclose(grad_W, expected_grad_W)

        fn_to_optimize = lambda b: vectorized_loss(
            X, Y, W, b)
        grad_b = numerical_gradient(fn_to_optimize, b)
        np.testing.assert_allclose(grad_b, expected_grad_b)

    def test_numerical_gradient_vectorized_all_incorrect(self):
        """ Test the numerical gradient for multiple inputs X. """
        W, X, b, Y, expected_grad_W, expected_grad_b = vectorized_test_data_all_incorrect()  # noqa

        fn_to_optimize = lambda W: vectorized_loss(
            X, Y, W, b)
        grad_W = numerical_gradient(fn_to_optimize, W)
        np.testing.assert_allclose(grad_W, expected_grad_W)

        fn_to_optimize = lambda b: vectorized_loss(
            X, Y, W, b)
        grad_b = numerical_gradient(fn_to_optimize, b)
        np.testing.assert_allclose(grad_b, expected_grad_b)

    def test_numerical_gradient_vectorized_all_correct(self):
        """ Test the numerical gradient for multiple inputs X. """
        W, X, b, Y, expected_grad_W, expected_grad_b = vectorized_test_data_all_correct()  # noqa

        fn_to_optimize = lambda W: vectorized_loss(
            X, Y, W, b)
        grad_W = numerical_gradient(fn_to_optimize, W)
        np.testing.assert_allclose(grad_W, expected_grad_W)

        fn_to_optimize = lambda b: vectorized_loss(
            X, Y, W, b)
        grad_b = numerical_gradient(fn_to_optimize, b)
        np.testing.assert_allclose(grad_b, expected_grad_b)


class TestLoss(TestCase):
    def test_single_point_loss_incorrect(self):
        W, x, b, correct_class_index, grad_W, grad_b = single_point_incorrect_test_data()  # noqa

        loss = single_point_loss(x, correct_class_index, W, b)
        np.testing.assert_almost_equal(loss, 41)

    def test_single_point_loss_correct(self):
        W, x, b, correct_class_index, grad_W, grad_b = single_point_correct_test_data()  # noqa

        loss = single_point_loss(x, correct_class_index, W, b)
        np.testing.assert_almost_equal(loss, 21)

    def test_vectorized_loss(self):
        W, X, b, Y, grad_W, grad_b = vectorized_test_data()

        loss = vectorized_loss(X, Y, W, b)
        np.testing.assert_almost_equal(loss, 31)

    def test_vectorized_loss_single_data_point_incorrect(self):
        """ Run the vectorized_loss function for a single data point. """
        W, x, b, correct_class_index, grad_W, grad_b = single_point_incorrect_test_data()  # noqa

        loss = vectorized_loss(x, correct_class_index, W, b)
        np.testing.assert_almost_equal(loss, 41)

    def test_vectorized_loss_single_data_point_correct(self):
        """ Run the vectorized_loss function for a single data point. """
        W, x, b, correct_class_index, grad_W, grad_b = single_point_correct_test_data()  # noqa

        loss = vectorized_loss(x, correct_class_index, W, b)
        np.testing.assert_almost_equal(loss, 21)

    def test_vectorized_loss_incorrect(self):
        W, X, b, Y, grad_W, grad_b = vectorized_test_data_all_incorrect()

        loss = vectorized_loss(X, Y, W, b)
        np.testing.assert_almost_equal(loss, 41)

    def test_vectorized_loss_correct(self):
        W, X, b, Y, grad_W, grad_b = vectorized_test_data_all_correct()

        loss = vectorized_loss(X, Y, W, b)
        np.testing.assert_almost_equal(loss, 21)


class TestAnalyticGradient(TestCase):
    def test_single_point_incorrect(self):
        W, x, b, correct_class_index, expected_grad_W, expected_grad_b = single_point_incorrect_test_data()  # noqa

        grad_W, grad_b = analytic_gradient(W, x, b, correct_class_index)
        np.testing.assert_allclose(grad_W, expected_grad_W)
        np.testing.assert_allclose(grad_b, expected_grad_b)

    def test_single_point_correct(self):
        W, x, b, correct_class_index, expected_grad_W, expected_grad_b = single_point_correct_test_data()  # noqa

        grad_W, grad_b = analytic_gradient(W, x, b, correct_class_index)
        np.testing.assert_allclose(grad_W, expected_grad_W)
        np.testing.assert_allclose(grad_b, expected_grad_b)

    # def test_vectorized(self):
    #     W, X, b, Y, expected_grad_W, expected_grad_b = vectorized_test_data()

    #     grad_W, grad_b = analytic_gradient_vectorized(W, X, b, Y)

    #     np.testing.assert_allclose(grad_W, expected_grad_W)
    #     np.testing.assert_allclose(grad_b, expected_grad_b)

    def test_vectorized_all_incorrect(self):
        W, X, b, Y, expected_grad_W, expected_grad_b = vectorized_test_data_all_incorrect()  # noqa

        grad_W, grad_b = analytic_gradient_vectorized(W, X, b, Y)

        np.testing.assert_allclose(grad_W, expected_grad_W)
        np.testing.assert_allclose(grad_b, expected_grad_b)

    # def test_vectorized_all_correct(self):
    #     W, X, b, Y, expected_grad_W, expected_grad_b = vectorized_test_data_all_correct()  # noqa

    #     grad_W, grad_b = analytic_gradient_vectorized(W, X, b, Y)

    #     np.testing.assert_allclose(grad_W, expected_grad_W)
    #     np.testing.assert_allclose(grad_b, expected_grad_b)

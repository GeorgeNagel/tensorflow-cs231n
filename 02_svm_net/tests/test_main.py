from unittest import TestCase

import numpy as np

from main import numerical_gradient, indexes_to_one_hot


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


class TestIndexesToOneHot(TestCase):
    def test_indexes_to_one_hot(self):
        indexes = np.array([0, 1, 2, 3])
        number_of_columns = 4
        expected_one_hot_arr = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        actual_one_hot_arr = indexes_to_one_hot(indexes, number_of_columns)
        np.testing.assert_allclose(
            expected_one_hot_arr,
            actual_one_hot_arr
        )

    def test_should_pad_with_zeros(self):
        indexes = np.array([0, 0, 0])
        number_of_columns = 3
        expected_one_hot_arr = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0]
        ])
        actual_one_hot_arr = indexes_to_one_hot(indexes, number_of_columns)
        np.testing.assert_allclose(
            expected_one_hot_arr,
            actual_one_hot_arr
        )

from unittest import TestCase

import numpy as np

from main import numerical_gradient


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

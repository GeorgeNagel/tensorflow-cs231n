import numpy as np

from net.nodes import SVMLossNode
from utils import numerical_gradient

for i in range(100):
    arr = np.random.random([2, 3])*4 - 2
    correct_class_indexes = np.array([0, 0])
    number_of_classes = 3

    node = SVMLossNode()
    node.forward(arr, correct_class_indexes, number_of_classes)

    analytical_gradient = node.gradients()[0]

    def fn_to_optimize(_arr):
        return np.sum(node.forward(_arr, correct_class_indexes, number_of_classes))
    num_grad = numerical_gradient(fn_to_optimize, arr)

    print("arr: %s" % arr)
    print("Numerical: %s" % num_grad)
    print("Analytical: %s" % analytical_gradient)
    print(
        "analytical==numerical %s" % (
            np.allclose(analytical_gradient, num_grad)
        )
    )

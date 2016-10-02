# This file trains a Support Vector Machine (SVM) on the iris dataset
import numpy as np


def analytic_gradient(W, x, b, y_expected):
    """
    Return the gradient for W and b given an input x and expected y
    """
    # Forward pass
    W_dot_x = np.dot(W, x)
    y_actual = np.add(W_dot_x, b)

    l_1_norm = np.linalg.norm(W, ord=1)
    correct_class_index = np.argmax(y_expected)
    # Get the score for the correct class
    sj = y_expected[correct_class_index]
    # Create an array with the value of the correct score
    sj_array = np.mult(np.ones(3), sj)

    s_diff = np.subtract(sj_array, y_actual)
    s_diff_biased = np.add(s_diff, 1)
    s_diff_biased_clamped = np.maximum(s_diff_biased, 0)
    score_loss = np.sum(s_diff_biased_clamped)
    loss = score_loss + l_1_norm

    # Back propogation
    d_loss_d_loss = 1
    # Calculate the gradient contribution of the regularization step

    return W_grad, b_grad


def numerical_gradient(func, x):
    """
    Estimate the gradient of function func at point x (a numpy vector).
    """
    # Evaluate the funcation at the original point.
    initial = func(x)
    grad_x = np.zeros(x.shape)
    h = 0.0001

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # Evaluate loss at x + h
        ix = it.multi_index
        old_value = x[ix]
        # Increment by h
        x[ix] = old_value + h
        # Evaluate loss(x + h)
        func_xh = func(x)
        # Restore to previous value
        x[ix] = old_value

        # Compute the partial derivative
        grad_x[ix] = (func_xh - initial) / h
        # Step to next dimension
        it.iternext()
    return grad_x

def vectorized_loss(x, correct_class_index, W, b):
    scores = np.dot(W, x) + b
    margins = np.maximum(0, scores - scores[correct_class_index] + 1)
    # Setting the correct class margin to zero after the fact
    # makes it easier to take advantage of the vectorized approach as above
    margins[correct_class_index] = 0
    loss = np.sum(margins)
    return loss


def predict(W, x, b):
    linear_result = np.dot(W, x)
    linear_with_bias = linear_result + b
    best_class = np.argmax(linear_with_bias)
    y_predicted = np.zeros(3)
    return y_predicted


if __name__ == '__main__':
    from tensorflow.contrib import learn
    # The output y will be calculated via
    # y = W * x + b
    iris = learn.datasets.load_dataset('iris')

    # The loss function for an SVM
    # Li = Sum<j!=yi>(max(0, sj - sy + 1)) + L1norm(W)

    # Each x sample has four dimensions,
    # and belongs to one of three possible classes
    # By dimensional analysis, since W must be
    # multiplied by x and added to an array to get y, W must be 4x3.
    # 3x4 * 4x1 + 3x1 => 3x1
    W_initial = np.zeros([3, 4])
    # Similarly, b must be a 3x1 array
    b_initial = np.zeros([3, 1])

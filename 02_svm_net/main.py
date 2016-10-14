import numpy as np


def numerical_gradient(func, x, step_size=0.0001):
    """
    Estimate the gradient of function func at point x (a numpy vector).
    """
    # Evaluate the funcation at the original point.
    initial = func(x)
    grad_x = np.zeros(x.shape)

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # Evaluate loss at x + h
        ix = it.multi_index
        old_value = x[ix]
        # Increment by h
        x[ix] = old_value + step_size
        # Evaluate loss(x + h)
        func_xh = func(x)
        # Restore to previous value
        x[ix] = old_value

        # Compute the partial derivative
        grad_x[ix] = (func_xh - initial) / step_size
        # Step to next dimension
        it.iternext()
    return grad_x


class AdditionNode(object):
    def forward(self, input_1, input_2):
        self.input_1 = input_1
        self.input_2 = input_2
        return np.add(input_1, input_2)

    def gradients(self):
        gradient_1 = np.ones(self.input_1.shape)
        gradient_2 = np.ones(self.input_2.shape)
        return [gradient_1, gradient_2]


class MultiplicationNode(object):
    def forward(self, input_1, input_2):
        self.input_1 = input_1
        self.input_2 = input_2
        return np.dot(self.input_1, self.input_2)

    def gradients(self):
        return [self.input_2, self.input_1]

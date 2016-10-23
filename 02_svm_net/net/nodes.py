import numpy as np


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
        grad_1_row = np.sum(self.input_2, axis=1)
        grad_1 = np.vstack(
            [grad_1_row for i in range(self.input_1.shape[0])]
        )
        grad_2_column = np.sum(self.input_1, axis=0)
        grad_2 = np.vstack(
            [grad_2_column for i in range(self.input_2.shape[1])]
        ).T
        return [grad_1, grad_2]


class SumNode(object):
    def forward(self, arr):
        self.arr = arr
        return np.sum(arr)

    def gradients(self):
        return [np.ones(self.arr.shape)]


class MaxNode(object):
    def forward(self, arr, clamp_value):
        self.clamp_value = clamp_value
        self.arr = arr
        return np.maximum(arr, clamp_value)

    def gradients(self):
        grad = np.ones(self.arr.shape)
        grad[self.arr < self.clamp_value] = 0
        return [grad]


class ScalarMultiplyNode(object):
    def forward(self, arr, scalar):
        self.arr = arr
        self.scalar = scalar
        return arr * scalar

    def gradients(self):
        return [np.ones(self.arr.shape) * self.scalar]


class ScalarAddNode(object):
    def forward(self, arr, scalar):
        self.arr = arr
        return arr + scalar

    def gradients(self):
        return [np.ones(self.arr.shape)]

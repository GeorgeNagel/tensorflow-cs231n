import numpy as np

from utils import indexes_to_one_hot


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


class SelectNode(object):
    """ Select one value for each row in the input array. """

    def forward(self, arr, select):
        self.select = select
        self.arr = arr
        number_of_rows = arr.shape[0]
        self.number_of_columns = arr.shape[1]
        row_values = arr[select == 1].reshape([number_of_rows, 1])
        selected_array = np.hstack(
            [row_values for i in range(self.number_of_columns)]
        )
        return selected_array

    def gradients(self):
        gradient = np.zeros(self.arr.shape)
        gradient[self.select == 1] = self.number_of_columns
        return [gradient]


class SVMLossNode(object):

    def forward(self, arr, correct_class_indexes, number_of_classes):
        correct_class_arr = indexes_to_one_hot(correct_class_indexes, number_of_classes)
        correct_scores_arr = SelectNode().forward(arr, correct_class_arr)
        negated_correct_scores_arr = ScalarMultiplyNode().forward(correct_scores_arr, -1)
        diffed_scores_arr = AdditionNode().forward(arr, negated_correct_scores_arr)

        margined_diffed_scores_arr = ScalarAddNode().forward(diffed_scores_arr, 1)
        negative_one_hot = ScalarMultiplyNode().forward(correct_class_arr, -1)
        margined_normalized_arr = ScalarAddNode().forward(negative_one_hot, margined_diffed_scores_arr)

        clamped_diffed_scores = MaxNode().forward(margined_normalized_arr, 0)
        loss = SumNode().forward(clamped_diffed_scores)
        return loss

    def gradient(self):
        pass

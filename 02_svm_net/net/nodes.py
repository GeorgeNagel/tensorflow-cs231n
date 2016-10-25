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
        # 0. Setup
        correct_class_arr = indexes_to_one_hot(correct_class_indexes, number_of_classes)
        self.select_node = SelectNode()
        correct_scores_arr = self.select_node.forward(arr, correct_class_arr)

        # 1. scores - correct_class_scores
        self.scalar_mult_neg_one_first = ScalarMultiplyNode()
        negated_correct_scores_arr = self.scalar_mult_neg_one_first.forward(correct_scores_arr, -1)
        self.addition_node_first = AdditionNode()
        diffed_scores_arr = self.addition_node_first.forward(arr, negated_correct_scores_arr)

        # 2. diffed_scores + 1
        self.add_one_node = ScalarAddNode()
        margined_diffed_scores_arr = self.add_one_node.forward(diffed_scores_arr, 1)
        self.scalar_mult_neg_one_second = ScalarMultiplyNode()
        negative_one_hot = self.scalar_mult_neg_one_second.forward(correct_class_arr, -1)
        self.addition_node_second = AdditionNode()
        margined_normalized_arr = self.addition_node_second.forward(negative_one_hot, margined_diffed_scores_arr)

        # 3. clamp to 0
        self.clamp_node = MaxNode()
        clamped_diffed_scores = self.clamp_node.forward(margined_normalized_arr, 0.0)
        self.sum_node = SumNode()
        loss = self.sum_node.forward(clamped_diffed_scores)
        return loss

    def gradients(self):
        # 3
        d_clamp_diffed_scores_d_loss = self.sum_node.gradients()[0]
        d_margined_normalized_d_loss = self.clamp_node.gradients()[0] * d_clamp_diffed_scores_d_loss

        # 2
        d_margined_diffed_d_loss = self.addition_node_second.gradients()[1] * d_margined_normalized_d_loss
        d_diffed_scores_d_loss = self.add_one_node.gradients()[0] * d_margined_diffed_d_loss

        # 1
        d_residual_arr_d_loss = self.addition_node_first.gradients()[0] * d_diffed_scores_d_loss
        d_negated_correct_d_loss = self.addition_node_first.gradients()[1] * d_diffed_scores_d_loss
        d_correct_scores_d_loss = self.scalar_mult_neg_one_first.gradients()[0] * d_negated_correct_d_loss

        # 0
        d_correct_class_d_loss = self.select_node.gradients()[0] * d_correct_scores_d_loss
        d_arr_d_loss = d_correct_class_d_loss + d_residual_arr_d_loss
        return [d_arr_d_loss]

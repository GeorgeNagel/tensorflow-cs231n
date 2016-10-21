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


class SVMLossNode(object):
    def forward(self, scores, correct_class_indexes):
        expected_out = np.zeros(scores.shape)
        for expected, index in zip(expected_out, correct_class_indexes):
            expected[index] = 1
        correct_class_scores = [
            score[index] for score, index in zip(scores, correct_class_indexes)
        ]
        scores_diffed = scores - correct_class_scores
        scores_diffed_margined = scores_diffed + 1
        scores_clamped = np.maximum(scores_diffed_margined, 0)
        scores_clamped[expected_out == 1] = 0
        loss = np.sum(scores_clamped)
        return loss

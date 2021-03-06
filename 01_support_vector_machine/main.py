# This file trains a Support Vector Machine (SVM) on the iris dataset
import numpy as np


def analytic_gradient_vectorized(W, X, b, Y):
    """
    Return the gradient for W and b given a matrix of inputs X and outputs Y
    """
    if X.ndim > 1:
        number_of_data_points = X.shape[0]
    else:
        number_of_data_points = 1
    # Forward pass
    W_dot_X = np.dot(W, X.T).T
    scores = (W_dot_X + b)
    # scores_diffed = (scores.T - scores[Y == 1]).T
    # scores_diffed_biased = scores_diffed + 1
    # scores_diffed_biased = np.where(Y == 1, 0, scores_diffed_biased)
    correct_class_scores = scores[Y == 1]
    scores_diffed = (scores.T - correct_class_scores).T
    scores_diffed_biased = scores_diffed + 1
    scores_diffed_biased = np.where(Y == 1, 0, scores_diffed_biased)
    clamped_scores = np.maximum(scores_diffed_biased, 0)

    hinge_loss = np.sum(clamped_scores)/number_of_data_points
    regularization_loss = np.sum(np.abs(W))
    loss = hinge_loss + regularization_loss

    # Back propogation
    d_loss_d_loss = np.ones(b.shape)
    # Calculate the gradient contribution of the regularization step
    d_reg_d_loss = np.ones(W.shape)
    # Calculate the gradient contribution of the clamping
    d_clamp_in_d_loss = np.where(
        scores_diffed_biased > 0, d_loss_d_loss, np.zeros(d_loss_d_loss.shape))

    # Calculate the gradient contribute of the score difference
    d_score_diff_d_loss = np.ones(b.shape) * d_clamp_in_d_loss
    # The correct values get a -1 contribution for each incorrect class above the clamp threshold
    incorrect_class_contributions_to_correct_grad = np.where(clamped_scores > 0, -1, 0)
    correct_class_values = np.sum(incorrect_class_contributions_to_correct_grad, axis=1)
    correct_class_values_matrix = np.broadcast_to(correct_class_values, (Y == 1).T.shape).T
    d_score_diff_d_loss = np.where(Y == 1, correct_class_values_matrix, d_score_diff_d_loss)
    # Calculate the gradient contribution of W dot X
    d_W_dot_X_d_loss = np.ones(b.shape) * d_score_diff_d_loss
    # Calculate the gradient contribution of W
    W_grad = np.dot(d_W_dot_X_d_loss.T, X)/number_of_data_points + d_reg_d_loss
    # Calculate the gradient contribution of b
    b_grad = np.sum((np.ones(b.shape) * d_score_diff_d_loss)/number_of_data_points, axis=0)
    return W_grad, b_grad, loss


def analytic_gradient(W, x, b, correct_class_index):
    """
    Return the gradient for W and b given an input x and expected y
    """
    # Forward pass
    W_dot_x = np.dot(W, x)
    y_actual = np.add(W_dot_x, b)

    # l_1_norm = np.sum(np.abs(W))
    # Get the score for the correct class
    sy = y_actual[correct_class_index]
    # Create an array with the value of the correct score
    sy_array = np.multiply(np.ones(y_actual.shape), sy)

    score_diff = np.subtract(y_actual, sy_array)
    score_diff_biased = np.add(score_diff, 1)
    # Account for the fact that we don't calculate loss on sj-sj comparisons
    score_diff_biased[correct_class_index] = 0
    # score_diff_biased_clamped = np.maximum(score_diff_biased, 0)
    # score_loss = np.sum(score_diff_biased_clamped)
    # loss = score_loss + l_1_norm

    # Back propogation
    d_loss_d_loss = np.ones(b.shape)
    # Calculate the gradient contribution of the regularization step
    d_reg_d_loss = np.ones(W.shape)
    # Calculate the gradient contribution of the clamping
    d_clamp_in_d_loss = np.where(
        score_diff_biased > 0, d_loss_d_loss, np.zeros(d_loss_d_loss.shape))
    # Calculate the gradient contribution of the bias
    d_bias_in_d_loss = np.ones(b.shape) * d_clamp_in_d_loss
    # Calculate the gradient contribute of the score difference
    d_y_d_loss = np.ones(b.shape) * d_bias_in_d_loss
    d_y_d_loss[correct_class_index] = -1 * np.sum(score_diff_biased > 0)
    # Calculate the gradient contribution of W dot x
    d_W_dot_x_d_loss = np.ones(b.shape) * d_y_d_loss
    # Calculate the gradient contribution of W
    W_grad = np.outer(d_W_dot_x_d_loss, x) + d_reg_d_loss
    # Calculate the gradient contribution of b
    b_grad = np.ones(b.shape) * d_y_d_loss
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


def single_point_loss(x, correct_class_index, W, b):
    scores = np.dot(W, x) + b
    margins = np.maximum(0, scores - scores[correct_class_index] + 1)
    # Setting the correct class margin to zero after the fact
    # makes it easier to take advantage of the vectorized approach as above
    margins[correct_class_index] = 0
    loss = np.sum(margins)

    regularization_cost = np.sum(np.abs(W))
    loss += regularization_cost
    return loss


def vectorized_loss(X, Y, W, b):
    """ Fully-vectorized loss function. """
    if X.ndim > 1:
        number_of_data_points = X.shape[0]
    else:
        number_of_data_points = 1
    W_dot_X = np.dot(W, X.T).T
    scores = (W_dot_X + b)
    correct_class_scores = scores[Y == 1]
    scores_diffed = (scores.T - correct_class_scores).T
    margins = np.maximum(0, scores_diffed + 1)
    # Set the correct class margin to zero after the fact
    margins[Y == 1] = 0

    hinge_loss = np.sum(margins)/number_of_data_points
    regularization_loss = np.sum(np.abs(W))
    loss = hinge_loss + regularization_loss
    return loss


def predict(W, x, b):
    linear_result = np.dot(W, x)
    linear_with_bias = linear_result + b
    best_class = np.argmax(linear_with_bias)
    return best_class


def accuracy(W, X_test, b, Y_test):
    total_seen = 0
    total_correct = 0
    for x, y in zip(X_test, Y_test):
        best_class = predict(W, x, b)
        expected = np.argmax(y)
        print("Best: %s. Expected: %s" % (best_class, expected))
        total_seen += 1
        if best_class == expected:
            total_correct += 1
    return float(total_correct) / float(total_seen)


def test_train_split(iris_data, train_fraction):
    number_of_samples = len(iris_data[1])
    number_of_classes = 3

    X = iris_data[0]
    np.random.shuffle(X)
    correct_classes = iris_data[1]
    Y = np.zeros([number_of_samples, number_of_classes])
    Y[np.arange(number_of_samples), correct_classes] = 1
    np.random.shuffle(Y)

    number_of_samples_train = round(number_of_samples * train_fraction)

    X_train = X[:number_of_samples_train]
    Y_train = Y[:number_of_samples_train]
    X_test = X[number_of_samples_train:]
    Y_test = Y[number_of_samples_train:]

    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    from tensorflow.contrib import learn
    # The output y will be calculated via
    # y = W * x + b
    iris_data = learn.datasets.load_dataset('iris')

    X_train, Y_train, X_test, Y_test = test_train_split(iris_data, .8)

    # The loss function for an SVM
    # Li = Sum<j!=yi>(max(0, sj - sy + 1)) + L1norm(W)

    # Each x sample has four dimensions,
    # and belongs to one of three possible classes
    # By dimensional analysis, since W must be
    # multiplied by x and added to an array to get y, W must be 4x3.
    # 3x4 * 4x1 + 3x1 => 3x1
    W = np.random.uniform(-1.0, 1.0, [3, 4])
    # Similarly, b must be a 3x1 array
    b = np.random.uniform(-1.0, 1.0, (3,))

    learning_rate = .1 ** 2
    print "INITIAL W: %s" % W

    for i in range(10000):
        grad_W, grad_b, loss = analytic_gradient_vectorized(W, X_train, b, Y_train)
        print("W: %s" % W)
        print("GRADW: %s" % grad_W)
        print("loss: %s" % loss)
        W = W - grad_W*learning_rate
        b = b - grad_b*learning_rate

    accy = accuracy(W, X_test, b, Y_test)
    print "ACCURACY: %s" % accy

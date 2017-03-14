import numpy as np
import math

def predict(x, θ):
    """
    Purpose
    =======
    Our linear model in vectorized form.

    Parameters
    ==========
    :θ: (m x 1) matrix containing the weights of our model
    :x: (1 x n) matrix of features

    Summary
    =======
    Linear algebra is everywhere in machine learning. It allows use
    to compute values quickly an efficiently and with less code.

    The vectorized form of our hypothesis function looks like:

        y_hat = θ' * x

    Here we take the transpose of θ and perform matrix multiplication
    with x. Programmatically we will need to take the transpose of x as
    well just to make sure the dimensions line up properly.

    This assumes θ is a column vector and x is a (1 x n) dimensional matrix.
    In this scenario, x will have a bias term of 1. The purpose of the bias
    term is to help predict a naive value even when no learning has occurred.
    """
    return θ.T.dot(x.T)

def cost_function(X, y, θ):
    """
    Purpose
    =========
    Used by gradient descent to measure the quality of our linear model

    Parameters
    ==========
    :X: (m x n) matrix of features
    :y: (m x 1) matrix of target labels
    :θ: (m x 1) matrix containing the weights of our model

    Summary
    =======
    Training a linear model means setting its parameters (θ)
    so that the model best fits the data. For this reason we
    need a measure of how poorly our model is fitting the data.

    A popular measure in linear regression is the mean squared error (MSE):

        MSE(X, hθ) = 1/m * ∑((predicted - actual)^2)
                   = 1/m * ∑((h(θ, X[i]) - y[i])^2)

    In statistics, the MSE measures the average of the squares of the errors -
    that is, the difference between an estimator and what is estimated. As
    a result, it can be used to measure the quality of an estimator such as
    our linear model.

    In math, multiplying (1/m) by the sum is the same as dividing the sum by m,
    which is how we traditionally obtain the mean value. This tripped me up
    until I realized it because I'm used to seeing the mean calculated as
    (sum / m) where m in the number of items being summed together.

        sum = 55 = [1 + 2 + ... + 10]
        5.5 = sum / 10
        5.5 = 1/10 * sum

    The algorithm we will implement is going to be slightly different than
    the one given above. It's formula is:

        MSE(X, hθ) = 1/(2 * m) * ∑((predicted - actual)^2)
                   = 1/(2 * m) * squared_error_sum

                   Can be rewritten as...

                   = 0.5 * (squared_error_sum / m)
                   = 0.5 * mean_of_sum_of_squared_errors

    To break it apart, it is 1/2 * mean(X) where mean(X) is the mean of the
    difference between the predicted value and the actual value. The mean is
    halved (1/2) as a convenience for the computation of the gradient descent,
    as the derivative term of the square function will cancel out the 1/2 term.
    """
    # A vectorized implementation of the above mean squared error cost function
    m     = len(X)      # Number of examples
    preds = X.dot(θ)    # Predicted values using matrix multiplication
    error = preds - y   # Difference between predicted and actual values
    square_error       = np.power(error, 2)    # Negative values not allowed
    square_error_sum   = np.sum(square_error)  # Add everything together
    mean_squared_error = (1 / m) * square_error_sum # Find the mean

     # Mean is halved to help gradient descent converge
    return 0.5 * mean_squared_error


def step_gradient(X, y, θ, alpha):
    """
    Purpose
    =======
    Perform an update the parameters θ

    Parameters
    ==========
    :X: (m x n + 1 matrix of feature
    :y: (m x 1) matrix of target labels
    :θ: (n x 1) matrix of weights
    :alpha: The learning rate determines the magnitude of the update

    Summary
    =======
    When applied to the case of linear regression, the gradient descent
    equation takes the form of:

        REPEAT UNTIL CONVERGENCE:

            θ0 = θ0 - α * 1/m ∑(hθ(x_i) - y_i)
            θ1 = θ1 - α * 1/m ∑(hθ(x_i) - y_i) * x_i
            ...
            θn = θn - α * 1/m ∑(hθ(x_n) - y_n) * x_n

    Here m is the size of the training set. θ are the weights of our model
    which will be continuously updated. Note that for n > 0 we are multiplying
    by x_i at the end due to the derivative.

    The point of all this is that if we start with a guess for our hypothesis
    and then repeatedly apply these gradient descent equations, our hypothesis
    will become more and more accurate.

    Note that we can also compute this in one LOC:

        return sθ - (alpha * ((2/m) * X.T.dot(X.dot(θ) - y)))

    """
    # This is a vectorized implementation of the gradient descent algorithm.
    m = len(X)              # Number of training examples
    n_features = X.shape[1] # Number of features in training set

    preds     = X.dot(θ)   # Predicted values using matrix multiplication
    error     = preds - y  # Predicted values vs actual values

    # NOTE
    # To perform the summation we will compute the dot product of
    # X and our error. For each row in X, the dot product will take
    # each column in the error matrix and perform an element-wise
    # multiplication followed by a summation.
    #
    # To vizualize, first imagine a (2 x 3) matrix (A):
    #
    #    matrix([[1, 2, 3],
    #            [4, 5, 6]])
    #
    # and then a (3 x 2) matrix (B):
    #
    #   matrix([[ 7,  8],
    #           [ 9, 10],
    #           [11, 12]])
    #
    # For each row in matrix A, we perform an element-wise multiplication with
    # each column of matrix B. Then we sum each product together to get the
    # final result. The result is then stored in a matrix (C) which has the
    # same number of rows as matrix A and the same number of columns as
    # matrix B.
    #
    # (1, 2, 3) • (7, 9, 11)  = 1×7 + 2×9  + 3×11  = 58
    # (1, 2, 3) • (8, 10, 12) = 1×8 + 2×10 + 3×12  = 64
    # (4, 5, 6) • (7, 9, 11)  = 4×7 + 5×9  + 6×11  = 139
    # (4, 5, 6) • (8, 10, 12) = 4×8 + 5×10 + 6×12  = 154
    #
    #   A = np.matrix([[1,2,3], [4,5,6]])
    #   B = np.matrix([[7, 8], [9, 10], [11, 12]])
    #
    #   A.dot(B) =>
    #       matrix([[ 58,  64],
    #               [139, 154]])
    #
    #
    error_sum = X.T.dot(error)        # Compute the dot product
    delta     = ((1/m) * error_sum).T # The magnitude of change, or gradient
    θ_update  = θ.T - (alpha * delta) # Make θ go down by α using subtraction

    return θ_update.T

def gradient_descent_runner(X, y, θ, alpha, n_iters):
    θ_update = θ
    j_hist   = np.array([], dtype=np.float32)
    θ_hist   = np.array([], dtype=np.float32)
    for i in range(n_iters):
        θ_update = step_gradient(X, y, θ_update, alpha)
        j_hist   = np.append(j_hist, cost_function(X, y, θ_update))

        # Watch gradient descent find the best fit
        if i % 10 == 0:
            θ_hist = np.append(θ_hist, θ_update)

    return (θ_update, j_hist, θ_hist)

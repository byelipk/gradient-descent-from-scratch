from algorithms import *

def cost_function_test_one():
    X = np.matrix('1 2; 1 3; 1 4; 1 5')
    y = np.matrix('7;6;5;4')
    theta  = np.matrix('0.1;0.2')
    result = cost_function(X, y, theta)
    print(result) # Should equal 11.9450

def cost_function_test_two():
    X = np.matrix('1 2 3; 1 3 4; 1 4 5; 1 5 6')
    y = np.matrix('7;6;5;4')
    theta  = np.matrix('0.1;0.2;0.3')
    result = cost_function(X, y, theta)
    print(result) # Should equal 7.0175

def gradient_descent_test_one():
    X = np.matrix('1 5; 1 2; 1 4; 1 5')
    y = np.matrix('1 6 4 2').T
    theta = np.matrix('0 0').T
    alpha = 0.01;
    num_iters = 1000;
    result = gradient_descent_runner(X, y, theta, alpha, num_iters)
    print(result) # 5.2148, -0.5733

def gradient_descent_test_two():
    X = np.matrix(' 2 1 3; 7 1 9; 1 8 1; 3 7 4')
    y = np.matrix('2 ; 5 ; 5 ; 6')
    theta = np.matrix('0 0 0').T
    alpha = 0.01;
    num_iters = 100;
    result = gradient_descent_runner(X, y, theta, alpha, num_iters)
    print(result) # 0.23680, 0.56524, 0.31248


cost_function_test_one()
cost_function_test_two()

gradient_descent_test_one()
gradient_descent_test_two()

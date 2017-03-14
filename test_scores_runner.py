import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from algorithms import *

filename = "data/test_scores.csv"
df   = pd.read_csv(filename)
data = np.asmatrix(df.as_matrix())
X    = data[:, 0]  # All rows from the first column
y    = data[:, 1]  # All rows from the second column

# Create a scatter plot
plt.scatter(X, y)
plt.figtext(.5,.95,'Hours studying v Test score', fontsize=18, ha='center')
plt.figtext(.5,.9,'Does studying more make my grades go up?',fontsize=10,ha='center')

plt.xlabel("Hours spent studying")
plt.ylabel("Test score")
plt.show()


# Perform feature scaling on X
scaler = StandardScaler()
X      = scaler.fit_transform(X)

# Add intercept term to X
X = np.c_[np.ones((len(X), 1)), X]

# Find the number of training examples and the number of features
m, n = X.shape

# Run gradient descent
theta  = np.random.randn(n).reshape(-1, 1)
alpha  = 0.01
epochs = 1000;

new_theta, j_hist = gradient_descent_runner(X, y, theta, alpha, epochs)
results = "Weights: θ0 = {0}, θ1 = {1}".format(
    new_theta[0,0], new_theta[1,0])
print(results)


print("Plotting cost function J")
iterations = np.arange(1, epochs + 1, 1)
plt.title("Cost function analysis")
plt.plot(iterations, j_hist)
plt.show()

def predict(hours):
    str = "Predicted test score after studying {0} hours:".format(hours)
    print(str)
    # Before we make a prediction we have to normalize the input data!
    hours_norm = (hours - scaler.mean_[0]) / scaler.scale_[0]
    x = np.array([[1, hours_norm]])
    print("{}".format(x.dot(new_theta)[0,0]))

predict(10)
predict(50)
predict(100)

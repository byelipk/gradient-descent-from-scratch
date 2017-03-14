import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from algorithms import *

print("Testing gradient descent implementation using Housing Prices dataset.")

# Step 1 - Collect our data
filename = "data/housing_prices.csv"
df   = pd.read_csv(filename)
data = np.asmatrix(df.as_matrix())
X    = data[:, :2]  # All rows from the first two columns
y    = data[:, 2]  # All rows from the third

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
epochs = 400;

new_theta, j_hist = gradient_descent_runner(X, y, theta, alpha, epochs)
results = "Weights: θ0 = {0}, θ1 = {1}, θ2 = {2}".format(new_theta[0,0], new_theta[1,0], new_theta[2,0])
print(results)

print("Plotting cost function J")
iterations = np.arange(1, epochs + 1, 1)
plt.title("Cost function analysis")
plt.plot(iterations, j_hist)
plt.show()

print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):")
# Before we make a prediction we have to normalize the input data!
sq_feet_norm = (1650 - scaler.mean_[0]) / scaler.scale_[0]
n_rooms_norm = (3 - scaler.mean_[1]) / scaler.scale_[1]
x = np.array([[1, sq_feet_norm, n_rooms_norm]])
print("${}".format(x.dot(new_theta)[0,0]))

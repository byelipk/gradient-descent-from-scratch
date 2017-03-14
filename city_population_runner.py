import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from algorithms import *

print("Testing gradient descent implementation using Food Truck dataset.")

# Step 1 - Collect our data
filename = "data/food_trucks.csv"
df   = pd.read_csv(filename)
data = np.asmatrix(df.as_matrix())
X    = data[:, 0]  # All rows from the first column
y    = data[:, 1]  # All rows from the second column

# Create a scatter plot
plt.scatter(X, y)
plt.figtext(.5,.95,'Population versus Profit', fontsize=18, ha='center')
plt.figtext(.5,.9,'Does population size cause our profit to go up?',fontsize=10,ha='center')

plt.xlabel("Population (10,000s)")
plt.ylabel("Profit ($10,000s)")
plt.show()

# Perform feature scaling on X
scaler = StandardScaler()
X      = scaler.fit_transform(X)

# Add intercept term to X
X = np.c_[np.ones((len(X), 1)), X]

# Find the number of training examples and the number of features
m, n = X.shape

# Run gradient descent
theta = np.zeros((n, 1))
alpha  = 0.01
epochs = 1000;

# Expect to see a cost function around 32
print(cost_function(X, y, theta))

new_theta, j_hist, theta_hist = gradient_descent_runner(
    X, y, theta, alpha, epochs)
results = "Weights: θ0 = {0}, θ1 = {1}".format(new_theta[0,0], new_theta[1,0])
print(results)

# Plot linear fit and cost function.
# Remember to remove the intercept term!
X_minus_b = X[:,1].reshape(-1,1)

plt.figtext(.5,.95,'Plotting gradient descent progress', fontsize=14, ha='center')
plt.plot(X_minus_b, y, "r+")
for model in theta_hist:
    plt.plot(X_minus_b, X.dot(model), "b--")
plt.show()

print("Plotting the best fit linear model")
plt.figtext(.5,.95,'Linear Model', fontsize=14, ha='center')
plt.figtext(.5,.9, results, fontsize=10, ha='center')
plt.plot(X_minus_b, y, "r+")
plt.plot(X_minus_b, X.dot(new_theta), "b-")
plt.show()

print("Plotting cost function J")
iterations = np.arange(1, epochs + 1, 1)
plt.title("Cost function analysis")
plt.plot(iterations, j_hist)
plt.show()

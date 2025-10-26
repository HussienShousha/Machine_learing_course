import pandas as pd
import numpy as np

#
# Final Notes may be notice cost not tend to zero,
# but actually it decreases alot if you need to see
# more accurate cost so scale y as well as x (Features)


data = pd.read_csv('data.csv')

X = data.iloc[:, :-1]  # all features
Y = data.iloc[:, -1]  # target

m = len(Y)  # training examples

X_range = np.max(X, axis=0) - np.min(X, axis=0)

X_mean = np.mean(X, axis=0)

X = (X - X_mean) / X_range  # feature scaling(norm)

X = np.c_[np.ones(m), X]

n = X.shape[1]

theta = np.zeros(n)

alpha = 0.1  # like lec note best learning rate between 0.001 and 1

iterations = 1500

J_history = []

for i in range(iterations):

    h = X @ theta

    error = h - Y

    gradient = (1 / m) * (X.T @ error)

    theta = theta - alpha * gradient

    cost = (1 / (2 * m)) * np.sum(error ** 2)
    J_history.append(cost)

    if i % 100 == 0:
        print(f"Iteration {i}: Cost = {cost}")

# Final Notes may be notice cost not tend to zero
# but actually it decreases alot if you need to see
# more accurate cost so scale y as well as x (Features)

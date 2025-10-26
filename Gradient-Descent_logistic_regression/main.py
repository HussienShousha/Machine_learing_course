
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

m = len(Y)


X_range = np.max(X, axis=0) - np.min(X, axis=0)

X_mean = np.mean(X, axis=0)

X = (X - X_mean) / X_range

X = np.c_[np.ones(m), X]

n = X.shape[1]

theta = np.zeros(n)


alpha = 0.1
iterations = 1500

J_history = []

def sigmoid(z):
    return 1 / (1 + np.exp(-z))




def cal_cost(X, Y, theta):
    m = len(Y)


    h = sigmoid(X @ theta)

    epsilon = 1e-5

    cost = -(1/m) * np.sum(Y * np.log(h + epsilon) + (1 - Y) * np.log(1 - h + epsilon))

    return cost



for iteration in range(iterations):

    h = sigmoid(X @ theta)

    error = h - Y

    gradient = (1 / m) * (X.T @ error)

    theta = theta - alpha * gradient

    cost = cal_cost(X, Y, theta)

    J_history.append(cost)




from random import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





data  = pd.read_csv('data.csv')

data.plot(kind='scatter', x='population', y='profit')

plt.show()

x = data['population'].values

y = data['profit'].values

m = len(y)


theta_0 = 0
theta_1 = 0

alpha = 0.01

iterations = 1500





for i in range(iterations):



    old_theta_0 = theta_0
    old_theta_1 = theta_1

    h = old_theta_0 + old_theta_1 * x

    error = h - y

    d_theta_0 = (1/m) * np.sum(error)
    d_theta_1 = (1/m) * np.sum(error*x)

    theta_0 = old_theta_0 - alpha * d_theta_0
    theta_1 = old_theta_1 - alpha * d_theta_1




print("theta_0", theta_0)
print("theta_1", theta_1)


y_pred = theta_0 + theta_1 * x

plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label='Fitted line') 
plt.xlabel('Population')
plt.ylabel('Profit')
plt.legend()
plt.show()



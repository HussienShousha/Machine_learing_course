from random import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('data.csv')

data.plot(kind = 'scatter', x='population', y='profit')


plt.show()


data = data.sample(frac=1, random_state=42).reset_index(drop=True)



x = data['population'].to_numpy().reshape(-1, 1)
y = data['profit'].to_numpy().reshape(-1, 1)


x_train = x[0:80]


y_train = y[0:80]


x_test = x[80:]
y_test = y[80:]


model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


print('Coefficients: \n', model.coef_, ' ', model.intercept_)

print('Mean squared error: %.2f', mean_squared_error(y_test, y_pred))



# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 23:50:57 2021

@author: Aditya Mishra
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as Sk_linear_reg
from sklearn.metrics import mean_squared_error
from my_linear_model import LinearRegression

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print('Describe Dataset\n', df.describe())

epoch = 10_000
my_clf = LinearRegression(max_iter=epoch, optimizer='bgd')
reg_clf = Sk_linear_reg()

x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'], axis=1), df['target'], test_size=0.3)
_, _, errors = my_clf.fit(x_train, y_train)
reg_clf.fit(x_train, y_train)

my_pred = my_clf.predict(x_test)
reg_pred = reg_clf.predict(x_test)

print(f"My Model's MSE: {my_clf.mse(y_test, my_pred):.3f}")
print(f"Sklearn Model's MSE: {mean_squared_error(y_test, reg_pred):.3f}")

# Error Plot
plt.title("Error in K-epochs")
plt.plot(range(epoch), errors, 'r-')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

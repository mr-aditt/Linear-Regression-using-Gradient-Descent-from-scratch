# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 23:50:57 2021

@author: Aditya Mishra
"""

import numpy as np
from math import sqrt


class LinearRegression:

    def __init__(self, lr=0.01, max_iter=100_000, optimizer='bgd', batch_size=128, copy_x=True):
        """

        :param lr: float, Learning Rate
        :param max_iter: int, Max number of iteration or epochs
        :param optimizer: str, 'bgd' <- Batch Gradient Descent
                                'sgd' <- Stochastic Gradient Descent
        :param batch_size: int, Partition training set into small batch size. Applicable only when
                                optimizer = 'sgd'
        :param copy_x: int, Make a copy of predictors
        """

        self.coef_ = 0.0 
        self.intercept_ = 0.0
        self.error_ = list()
        self.lr = lr
        self.max_iter = max_iter
        self.optimizer = optimizer.lower()
        self.batch_size = batch_size if self.optimizer == 'sgd' else None
        self.copy_x = copy_x
    
    @staticmethod
    def mse(y_actual, y_predicted):
        """

        :param y_actual: Array, True response
        :param y_predicted: Array, Predicted response
        :return: float, Amount of error
        """
        return (1/len(y_actual))*sum((y_actual-y_predicted)**2)
    
    @staticmethod
    def rmse(y_actual, y_predicted):
        """

        :param y_actual: Array, True response
        :param y_predicted: Array, Predicted response
        :return: float, Square root of amount of error
        """
        print(type(y_actual))
        return sqrt((1/len(y_actual))*sum((y_actual-y_predicted)**2))
    
    def bgd(self, x, y):
        """
        This is Batch Gradient Descent.
        :param x: Array [n_instances, n_features], Predictors
        :param y: Array [n_instances,], Response
        """
        # 1. Predict y for the dataset
        y_hat = np.dot(x, self.coef_) + self.intercept_
        
        # 2. Calculate Loss
        loss = self.mse(y, y_hat)
        self.error_.append(loss)
        
        # 3. Calculate the amount to change in 
        # coefficients and intercept needed to 
        # reduce the loss.
        j_coef = (-2 / x.shape[0]) * np.dot(x.T, (y - y_hat))
        j_intercept = (-2 / x.shape[0]) * np.sum(y - y_hat)

        # 4. Update the coefficients and intercept
        #  with that change 
        self.coef_ -= self.lr * j_coef
        self.intercept_ -= self.lr * j_intercept
    
    def sgd(self, x, y):
        """
        This is Stochastic Gradient Descent. Per iteration, sgd runs for (n_instances by batch_size) times
        :param x: Array [n_instances, n_features], Predictors
        :param y: Array [n_instances,], Response
        """
        start = 0
        end = self.batch_size
        j_coef, j_intercept, loss = 0, 0, 0
        while end < x.shape[0]:
            y_hat = np.dot(x[start:end, :], self.coef_) + self.intercept_
            loss += self.mse(y[start:end], y_hat)
            j_coef += (-2 / self.batch_size) * np.dot(x[start:end, :].T, (y[start:end] - y_hat))
            j_intercept += (-2 / self.batch_size) * np.sum(y[start:end] - y_hat)
            start = end
            end += self.batch_size
        self.error_.append(loss)    
        self.coef_ -= self.lr * j_coef
        self.intercept_ -= self.lr * j_intercept
    
    def fit(self, x, y):
        """
        This method fits the Regression line on dataset by calling either bgd or sgd for max_iter times.
        :param x: Array [n_instances, n_features], Predictors
        :param y: Array [n_instances,], Response
        :return: Tuple, (coefficients, y-intercept, errors)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        x_copy = x.copy() if self.copy_x is True else x
        # Run for 'e' max_iter and calculate optimal coefficients and intercept values
        self.coef_ = np.zeros(x_copy.shape[1])
        for e in range(self.max_iter):
            if self.optimizer == 'bgd':
                self.bgd(x_copy, y)
            elif self.optimizer == 'sgd':
                self.sgd(x_copy, y)
            
        return self.coef_, self.intercept_, self.error_
    
    def predict(self, x):
        """
        This method predict the response for predictors based on fitted line's
        coefficients and intercept.
        :param x: Array, Predictor
        :return: Array, Predicted response
        """
        x = np.asarray(x)
        return np.dot(x, self.coef_) + self.intercept_

    def __repr__(self):
        return f'\n\nLinearRegression{self.coef_, self.intercept_, self.lr, self.max_iter, self.optimizer}'
    
    def __str__(self):
        return '\n\nClassifier: Linear Regression\nAttributes:\n\tcoefficient: {} \n\tintercept: {:.2f} ' \
               '\n\tlearning_rate: {} \n\tepochs: {:,} \n\toptimizer: {}'.format(self.coef_, self.intercept_, self.lr,
                                                                                 self.max_iter, self.optimizer)

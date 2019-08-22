# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:47:07 2019

@author: Brian Kenji Aguilar
"""

# Importing necessary libraries and classes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# Reading the dataset
data_set = pd.read_excel('slr02.xls')
# Independent Variable
chirps = data_set.iloc[:,:-1].values
# Dependent Variable
temperature = data_set.iloc[:,len(chirps[0])].values

# Create the training and test set
X_train, X_test, Y_train, Y_test = train_test_split(chirps,temperature, test_size = 0.2, random_state = 1)

# Fit our linear regression model to our training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Test our linear regression model
Y_predict = regressor.predict(X_test)

# Plot our results for the training and test sets
plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Cricket Chirps vs Temperature (Train)")
plt.xlabel("Cricket Chirps")
plt.ylabel("Temperature")
plt.show()


plt.scatter(X_test,Y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Cricket Chirps vs Temperature (Test)")
plt.xlabel("Cricket Chirps")
plt.ylabel("Temperature")
plt.show()
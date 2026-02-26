#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:05:30 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Learning Objective: Observe OLS breakdown in the Overfitting Regime (n approx m).

# 1. Load and Expand Data
print('Loading Wine Quality data...')
# TASK: Load data and use PolynomialFeatures(degree=2) to expand features to m=77.

wine = fetch_openml('wine-quality-red', version=1, as_frame=True)
X = wine.data
y = wine.target.astype(float)

print(f'Features (m): {X.shape[1]}')

poly = PolynomialFeatures(degree=2)
X_expanded = poly.fit_transform(X)
X_expanded = StandardScaler().fit_transform(X_expanded)


# 2. Unstable Split
# TASK: Use a very small training fraction (e.g., train_size=0.05 or n=80) 
# to make n approach m.
X_train, X_test, y_train, y_test = train_test_split(X_expanded, y, train_size=0.05, random_state=24)

print(f'Training samples (n): {X_train.shape[0]}')
print(f'Features (m): {X_train.shape[1]}')

# 3. Fit OLS
# TASK: Fit LinearRegression on the expanded features.

model = LinearRegression().fit(X_train, y_train)

# 4. Evaluate
# TASK: Calculate and print Training MSE and Test MSE.
# Note if the test error "explodes" compared to the stable regime.

train_mse = mean_squared_error(y_train, y_pred=model.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred=model.predict(X_test))

print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# 5. Visualization
# TASK: Create a bar chart (use log scale for y-axis if error is very high).
plt.figure(figsize=(6, 4))
plt.bar(['Train MSE', 'Test MSE'], [train_mse, test_mse], color=['blue', 'orange'])
plt.title('Training vs Test MSE for OLS in Overfitting Regime')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:06:27 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Learning Objective: Observe OLS behavior in the Stable Regime (n >> m).
np.random.seed(24)

# 1. Load Data
print('Loading Wine Quality data...')
# TASK: Load wine-quality-red, scale features, and cast target to float.

wine = fetch_openml('wine-quality-red', version=1, as_frame=True)
X = wine.data
y = wine.target.astype(float)

X_scaled = StandardScaler().fit_transform(X)

# 2. Stable Split
# TASK: Create a split with 80% training data (Large n, Small m).
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=24)

# 3. Fit OLS
# TASK: Use LinearRegression.
model = LinearRegression().fit(X_train, y_train)

# 4. Evaluate
# TASK: Calculate and print Training MSE and Test MSE.
# Compare the generalization gap.

train_mse = mean_squared_error(y_train, y_pred=model.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred=model.predict(X_test))

print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Ratio of samples to features: {X_scaled.shape[0]/X_scaled.shape[1]:.2f}")

# 5. Visualization
# TASK: Create a bar chart comparing Train vs Test MSE.
plt.figure(figsize=(6, 4))
plt.bar(['Train MSE', 'Test MSE'], [train_mse, test_mse], color=['blue', 'orange'])
plt.title('Training vs Test MSE for OLS')
plt.ylabel('Mean Squared Error')
plt.show()
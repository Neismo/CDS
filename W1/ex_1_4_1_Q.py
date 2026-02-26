#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 16:47:23 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

np.random.seed(24)

# --- SECTION 1: Data Loading ---
print('Fetching Wine Quality data...')
# TASK: Load the 'wine-quality-red' dataset from OpenML.
# Ensure the target is cast to float.

df = fetch_openml('wine-quality-red', version=1, as_frame=True)
X = df.data
y = df.target.values.astype(float)

# --- SECTION 2: Preprocessing ---
# TASK: 
# 1. Scale the features using StandardScaler.
# 2. Perform a standard 70/30 train/test split.

X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=24)

# --- SECTION 3: Complexity Sweep ---
# TASK: Sweep through different values of lambda (Ridge alpha).
# Calculate both Training MSE and Test MSE for each.

lambdas = np.logspace(-3, 5, 25)
train_errors = []
test_errors = []

for l in lambdas:
   model = Ridge(alpha=l).fit(X_train, y_train)
   train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
   test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

# --- SECTION 4: Visualization ---
# TASK: Plot Training vs. Test Error.
# Invert the x-axis so complexity (low lambda) increases to the right.

plt.figure(figsize=(10, 6))
plt.plot(lambdas, train_errors, label="Train MSE")
plt.plot(lambdas, test_errors, label="Test MSE")
plt.legend()
plt.xscale('log')
plt.ylabel("MSE")
plt.xlabel("Complexity (as function of lambda)")
plt.gca().invert_xaxis()
plt.show()
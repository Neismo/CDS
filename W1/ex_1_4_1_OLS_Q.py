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

# 1. Load Data
print('Loading Wine Quality data...')
# TASK: Load wine-quality-red, scale features, and cast target to float.

# 2. Stable Split
# TASK: Create a split with 80% training data (Large n, Small m).
# X_train, X_test, y_train, y_test = ...

# 3. Fit OLS
# TASK: Use LinearRegression.

# 4. Evaluate
# TASK: Calculate and print Training MSE and Test MSE.
# Compare the generalization gap.

# 5. Visualization
# TASK: Create a bar chart comparing Train vs Test MSE.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 16:43:08 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Learning Objective: Observe coefficient instability and its link to collinearity.

# --- SECTION 1: Parameters ---
np.random.seed(42) 
n_samples = 100
n_simulations = 500
sigma = 1.0
rho = 0.98
beta_true = np.array([2, 0])
x_test = np.array([[1, 1]])
target_val = (x_test @ beta_true)[0]

def generate_data(n, rho, sigma):
    '''
    TASK: Generate synthetic data.
    1. Create X with two features correlated by rho (use multivariate_normal).
    2. Generate y = X @ beta_true + noise.
    '''
    X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]], size=n)
    noise = np.random.normal(0, sigma, n)
    y = X @ beta_true + noise
    return X, y

# --- SECTION 2: Simulation ---
# TASK: Run a loop for n_simulations.
all_betas = []
all_preds = []

print('Running simulations...')
for _ in range(n_simulations):
    X_train, y_train = generate_data(n_samples, rho, sigma)
    model = LinearRegression().fit(X=X_train, y=y_train)
    all_betas.append(model.coef_)
    all_preds.append(model.predict(X=x_test)[0])

# --- SECTION 3: Calculations ---
# TASK: Calculate the following metrics:
# 1. Mean and Variance of the estimated coefficients (betas).
# 2. The Bias^2 at x_test.
# 3. The Variance at x_test.

bias_sq = (np.mean(all_preds) - target_val)**2
variance = np.var(all_preds)

print(f"Bias²: {bias_sq:.5f}, Variance: {variance:.5f}")

# --- SECTION 4: Visualization ---
# TASK: Create a histogram of estimated Beta 1 and Beta 2.
plt.hist(x=[beta[0] for beta in all_betas], label="Beta[0]")
plt.hist(x=[beta[1] for beta in all_betas], label="Beta[1]")
plt.legend()
plt.show()
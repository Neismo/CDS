#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 16:45:31 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Learning Objective: Quantify the Bias-Variance tradeoff using Ridge lambda.

# --- SECTION 1: Parameters ---
np.random.seed(42)
n_samples = 100
n_simulations = 500
sigma = 1.0          # Noise level
beta_true = np.array([2, 0])
x_eval = np.array([[1, 1]])
target_val = (x_eval @ beta_true)[0]
lambdas = np.logspace(-2, 5, 25)

# Fixed X for consistent theoretical/empirical comparison
mean = [0, 0]
cov = [[1, 0.98], [0.98, 1]] # High correlation
X_fixed = np.random.multivariate_normal(mean, cov, n_samples)

# --- SECTION 2: Simulation ---
# TASK: Iterate through lambdas and simulations.
# 1. Generate y = X_fixed @ beta_true + noise.
# 2. Fit Ridge(alpha=l).
# 3. Predict at x_eval.
# 4. Calculate empirical Bias^2 and Variance.

lambda_preds = []
bias_sq = []
variance = []

print('Simulating Ridge complexity sweep...')
for l in lambdas:
    all_preds = []
    for _ in range(n_simulations):
        y = X_fixed @ beta_true + np.random.normal(0, sigma, n_samples)
        ridge_coeff = (X_fixed.T @ X_fixed + np.eye(2) * l)**-1 @ X_fixed.T @ y
        pred = x_eval @ ridge_coeff
        all_preds.append(pred[0])
    lambda_preds.append(all_preds)
    bias_sq.append((np.mean(all_preds) - target_val)**2)
    variance.append(np.var(all_preds))

print(variance[5], bias_sq[5])

# --- SECTION 3: Visualization ---
# TASK: Plot Bias^2, Variance, and Total (Bias^2 + Var) vs Lambdas.
# Remember to invert the x-axis for complexity.

plt.figure(figsize=(10, 6))
plt.plot(lambdas, bias_sq, label='Bias²', marker='o')
plt.plot(lambdas, variance, label='Variance', marker='o')
plt.plot(lambdas, np.array(bias_sq) + np.array(variance), label='Total Error', marker='o')
# Plot vertical line a lowest total error
min_total_idx = np.argmin(np.array(bias_sq) + np.array(variance))
plt.axvline(x=lambdas[min_total_idx], color='gray', linestyle='--', label='Min Total Error')
plt.xscale('log')
plt.gca().invert_xaxis()
plt.legend()
plt.show()
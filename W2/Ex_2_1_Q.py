import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DTU Colors
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

def info_leakage_audit():
    np.random.seed(42)
    N, M = 50, 1000  # Small sample, massive feature space (perfect for leakage)
    X = np.random.randn(N, M)
    y = np.random.randn(N)

    print('--- Workflow A: Leakage ---')
    # 1. Scaling the WHOLE dataset (The first sin)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Correlation on the WHOLE dataset (The second sin)
    # We calculate correlation of each column in X with y
    corrs = np.array([np.abs(np.corrcoef(X_scaled[:, i], y)[0, 1]) for i in range(M)])
    top_10_indices = np.argsort(corrs)[-10:][::-1]
    
    # 3. Selection and Splitting
    X_selected = X_scaled[:, top_10_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.1, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2_a = model.score(X_test, y_test)
    corrs_a = corrs[top_10_indices] 
    print(f'Workflow A (Leaky) Test R^2: {r2_a:.3f}')

    print('\n--- Workflow B: The Audit (No Leakage) ---')
    # 1. Split FIRST (The Golden Rule)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # 2. Scale based on Train only
    scaler_b = StandardScaler()
    X_train_scaled = scaler_b.fit_transform(X_train_raw)
    X_test_scaled = scaler_b.transform(X_test_raw)
    
    # 3. Find correlations using ONLY training data
    corrs_train = np.array([np.abs(np.corrcoef(X_train_scaled[:, i], y_train)[0, 1]) for i in range(M)])
    top_10_indices_b = np.argsort(corrs_train)[-10:][::-1]
    
    # 4. Select features
    X_train_selected = X_train_scaled[:, top_10_indices_b]
    X_test_selected = X_test_scaled[:, top_10_indices_b]
    
    model_b = LinearRegression()
    model_b.fit(X_train_selected, y_train)
    r2_b = model_b.score(X_test_selected, y_test)
    corrs_b = corrs_train[top_10_indices_b]
    print(f'Workflow B (Non-leaky) Test R^2: {r2_b:.3f}')

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(10), corrs_a, color=DTU_RED)
    plt.title('Leaky: Correlations (Whole Set)', color=DTU_NAVY)
    plt.ylabel('Abs. Correlation with y')
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    plt.bar(range(10), corrs_b, color=DTU_NAVY)
    plt.title('Honest: Correlations (Train Set Only)', color=DTU_NAVY)
    plt.ylabel('Abs. Correlation with y')
    plt.ylim(0, 1)
    plt.show()

if __name__ == '__main__':
    info_leakage_audit()
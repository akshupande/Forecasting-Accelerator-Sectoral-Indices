# Databricks notebook source
# NIFTY 50 Forecasting using DWT and Multiple ML Models
# This notebook demonstrates forecasting NIFTY 50 closing prices using:
# - Multiple features (price, volume, volatility, lags, moving averages)
# - Wavelet denoising (Haar)
# - Random Forest, XGBoost, Ridge, Lasso models
# - Train: Apr 2023 – Mar 2025, Forecast: Apr 2025 – Sep 2025



# ==============================
# 1. Problem Overview
# ==============================
"""
Objective: Forecast NIFTY 50 index closing prices for Apr 2025 – Sep 2025
using historical data from Apr 2023 – Mar 2025.

Dataset:
- train1: Apr 2023 - Mar 2024
- train2: Apr 2024 - Mar 2025  
- test: Apr 2025 - Oct 2025

Approach:
1. Feature engineering (lags, moving averages, volatility, ATR, returns)
2. Wavelet denoising (Haar) for smoothing
3. Train multiple ML models (RandomForest, XGBoost, Lasso, Ridge)
4. Evaluate models and select best performer
5. Visualize results and save predictions
"""

# ==============================
# 2. Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pywt
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 3. Load and Preprocess Data
# ==============================
# Load datasets
train1 = pd.read_csv("NIFTY 50-01-04-2023-to-31-03-2024.csv")
train2 = pd.read_csv("NIFTY 50-01-04-2024-to-31-03-2025.csv")
test = pd.read_csv("NIFTY 50-01-04-2025-to-30-09-2025.csv")

# Combine training data
train = pd.concat([train1, train2], ignore_index=True)

# Standardize column names and convert numeric columns
def preprocess_data(df):
    df = df.copy()
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert date column
    if 'Date' not in df.columns:
        raise ValueError("Date column not found in DataFrame")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # Convert numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    return df.sort_values('Date').reset_index(drop=True)

train = preprocess_data(train)
test = preprocess_data(test)

print(f"Training data: {train['Date'].min()} to {train['Date'].max()}")
print(f"Test data: {test['Date'].min()} to {test['Date'].max()}")

# ==============================
# 4. Wavelet Denoising Function
# ==============================
def wavelet_denoise(data, wavelet='haar', level=1):
    """
    Apply Haar wavelet denoising with soft thresholding
    """
    coeff = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeff[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]
    reconstructed = pywt.waverec(coeff, wavelet)
    
    # Ensure same length as input
    return reconstructed[:len(data)]


# ==============================
# 5. Feature Engineering
# ==============================
def create_features(df):
    df = df.copy()
    
    # Basic price features
    df['HL_Ratio'] = df['High'] / df['Low']
    df['OC_Ratio'] = df['Open'] / df['Close']
    
    # Returns
    df['Returns'] = df['Close'].pct_change()
    df['Returns_1'] = df['Returns'].shift(1)
    df['Returns_2'] = df['Returns'].shift(2)
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    
    # Moving averages
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    
    # Exponential moving averages
    df['EMA_5'] = df['Close'].ewm(span=5).mean()
    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    
    # Volatility
    df['Volatility_5'] = df['Returns'].rolling(5).std()
    df['Volatility_10'] = df['Returns'].rolling(10).std()
    
    # ATR (Average True Range)
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR_5'] = df['TR'].rolling(5).mean()
    df['ATR_10'] = df['TR'].rolling(10).mean()
    
    # Price position relative to range
    df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    return df

# Apply feature engineering
train_fe = create_features(train)
test_fe = create_features(test)

# Apply wavelet denoising to Close price
train_fe['Close_DWT'] = wavelet_denoise(train_fe['Close'].dropna().values)
test_fe['Close_DWT'] = wavelet_denoise(test_fe['Close'].dropna().values)

# Fill missing values
train_fe = train_fe.fillna(method='bfill').fillna(method='ffill')
test_fe = test_fe.fillna(method='bfill').fillna(method='ffill')

train_fe['Close_DWT'] = pd.Series(wavelet_denoise(train_fe['Close'].values), index=train_fe.index)
test_fe['Close_DWT'] = pd.Series(wavelet_denoise(test_fe['Close'].values), index=test_fe.index)

# ==============================
# 6. Prepare Training and Test Sets
# ==============================
# Feature columns (excluding date and target)
feature_cols = [col for col in train_fe.columns if col not in ['Index Name', 'Date', 'Close']]

X_train = train_fe[feature_cols]
y_train = train_fe['Close']
X_test = test_fe[feature_cols]
y_test = test_fe['Close']

print(f"Training features: {X_train.shape}")
print(f"Test features: {X_test.shape}")

# ==============================
# 7. Train Multiple Models
# ==============================
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': 'xgb',  # Will use alternative if not available
    'Lasso': Lasso(alpha=0.01, random_state=42),
    'Ridge': Ridge(alpha=1.0, random_state=42)
}

# Try to import XGBoost, fallback to RandomForest
try:
    from xgboost import XGBRegressor
    models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
except ImportError:
    print("XGBoost not available, using RandomForest instead")
    models['XGBoost'] = RandomForestRegressor(n_estimators=100, random_state=42)

# Train all models
trained_models = {}
predictions = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    predictions[name] = model.predict(X_test)

# ==============================
# 8. Evaluate Models
# ==============================
results = []

for name, y_pred in predictions.items():
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    })

results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print(results_df)

# Identify best model
best_model_name = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
best_model = trained_models[best_model_name]
best_predictions = predictions[best_model_name]

print(f"\nBest Model: {best_model_name}")

# ==============================
# 9. Visualizations
# ==============================
plt.style.use('ggplot')  # Built-in style, no extra installs
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance & Data Insights', fontsize=18, fontweight='bold')

# Colors for clarity
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# Plot 1: Actual vs Predicted for all models
for idx, (name, y_pred) in enumerate(predictions.items()):
    axes[0, 0].plot(test_fe['Date'], y_pred, label=name,
                    color=colors[idx], linewidth=2)

axes[0, 0].plot(test_fe['Date'], y_test, label='Actual',
                color='black', linewidth=3)
axes[0, 0].set_title('Actual vs Predicted Prices - All Models', fontsize=14)
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Close Price')
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Best model detailed comparison
axes[0, 1].plot(test_fe['Date'], y_test, label='Actual',
                color='black', linewidth=3)
axes[0, 1].plot(test_fe['Date'], best_predictions,
                label=f'Predicted ({best_model_name})',
                color='tab:orange', linestyle='--', linewidth=2)
axes[0, 1].set_title(f'Best Model: {best_model_name}', fontsize=14)
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Close Price')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Residuals for best model
residuals = y_test - best_predictions
axes[1, 0].scatter(test_fe['Date'], residuals, alpha=0.7,
                   color='purple', s=40)
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_title(f'Residuals - {best_model_name}', fontsize=14)
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Original vs DWT Denoised series
axes[1, 1].plot(train_fe['Date'], train_fe['Close'],
                label='Original', color='tab:blue', linewidth=2)
axes[1, 1].plot(train_fe['Date'], train_fe['Close_DWT'],
                label='DWT Denoised', color='tab:green', linewidth=2)
axes[1, 1].set_title('Original vs Wavelet Denoised Series', fontsize=14)
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Close Price')
axes[1, 1].legend()
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ==============================
# 10. Save Predictions
# ==============================
# Create results dataframe
results_data = {
    'Date': test_fe['Date'],
    'Actual_Close': y_test
}

# Add predictions from all models
for name, y_pred in predictions.items():
    results_data[f'Predicted_{name}'] = y_pred

# Create final results dataframe
predictions_df = pd.DataFrame(results_data)

# Save to CSV
predictions_df.to_csv('NIFTY_50_Forecast_Predictions.csv', index=False)
print("Predictions saved to 'NIFTY_50_Forecast_Predictions.csv'")

# Display first few rows of predictions
print("\nFirst 10 rows of predictions:")
print(predictions_df.head(10))
